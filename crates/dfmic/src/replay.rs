//! Replay a recorded network trace through the DSP.
//!
//! `dfmic replay-dsp` feeds gains/coefs captured from upstream libDF into our
//! signal processing, so a parity failure can be attributed to the DSP rather
//! than to the network. The trace is produced by building libDF with
//! `DF_DUMP=<path>` (see the README) and contains one `FRAME` line per hop.
//!
//! The trace also records *which* stages libDF ran for each frame: a gains list
//! with no coefficients means "ERB mask only", an empty list for both means
//! "nothing", and so on. Those decisions are replayed as they were recorded.
//! Re-deriving them from our own thresholds would let this tool invent
//! divergences of its own — a frame libDF did not mask because its thresholds
//! differ would be zeroed here, and the parity comparison would blame the DSP
//! for a difference the trace never had.
//!
//! This is a development tool: it exists to keep the parity gate honest.

use std::path::Path;

use dfn_dsp::{
    DfnPipeline, EncoderOutput, ModelRuntime, Params, Stages, DF_ORDER, HOP_SIZE, NB_DF, NB_ERB,
};
use num_complex::Complex32;

use crate::wav;

/// One frame of recorded network output.
struct Frame {
    lsnr: f32,
    /// Empty when libDF applied no mask to this frame.
    gains: Vec<f32>,
    /// Empty when libDF skipped the deep filter for this frame.
    coefs: Vec<f32>,
    /// The stages libDF ran, recovered from which lists the trace carried.
    stages: Stages,
}

struct Replay {
    frames: Vec<Frame>,
    next: usize,
    current: usize,
}

impl Replay {
    fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut frames = Vec::new();
        for line in text.lines() {
            // Traces are often captured alongside other logging; ignore the rest.
            let Some(rest) = line.strip_prefix("FRAME ") else {
                continue;
            };
            let (lsnr, rest) = rest
                .split_once(" G[")
                .ok_or_else(|| format!("bad trace line: {line}"))?;
            let (gains, coefs) = rest
                .split_once("] C[")
                .ok_or_else(|| format!("bad trace line: {line}"))?;
            let coefs = coefs.trim_end_matches(']');
            let lsnr: f32 = lsnr
                .trim()
                .parse()
                .map_err(|_| format!("bad lsnr: {line}"))?;
            let gains = parse_list(gains)?;
            let coefs = parse_list(coefs)?;
            // A mask that is present but the wrong length would otherwise be
            // copied into the start of the gain array and leave the tail holding
            // whatever the previous frame put there.
            if !gains.is_empty() && gains.len() != NB_ERB {
                return Err(format!(
                    "trace frame has {} gains, the model wants {NB_ERB}: {line}",
                    gains.len()
                ));
            }
            if !coefs.is_empty() && coefs.len() != NB_DF * DF_ORDER * 2 {
                return Err(format!(
                    "trace frame has {} coefficients, the model wants {}: {line}",
                    coefs.len(),
                    NB_DF * DF_ORDER * 2
                ));
            }
            // libDF dumps a zero mask for a frame it zeroed and no mask at all
            // for one it passed through, so all-zero gains and no gains are
            // genuinely different cases and both are recoverable from the trace.
            let stages = Stages {
                gains: !gains.is_empty() && gains.iter().any(|gain| *gain != 0.0),
                zero: !gains.is_empty() && gains.iter().all(|gain| *gain == 0.0),
                deep_filter: !coefs.is_empty(),
            };
            frames.push(Frame {
                lsnr,
                gains,
                coefs,
                stages,
            });
        }
        if frames.is_empty() {
            return Err("empty trace".into());
        }
        Ok(Self {
            frames,
            next: 0,
            current: 0,
        })
    }
}

fn parse_list(s: &str) -> Result<Vec<f32>, String> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|x| {
            x.trim()
                .parse::<f32>()
                .map_err(|_| format!("bad number {x:?}"))
        })
        .collect()
}

impl ModelRuntime for Replay {
    fn encode_into(&mut self, _feat_erb: &[f32], _feat_spec: &[f32], out: &mut EncoderOutput) {
        self.current = self.next;
        self.next = (self.next + 1) % self.frames.len();
        out.lsnr = self.frames[self.current].lsnr;
    }

    fn stages(&self, _lsnr: f32) -> Option<Stages> {
        // The trace recorded which stages libDF ran; replaying them is the whole
        // point, so the pipeline's thresholds never get a say.
        Some(self.frames[self.current].stages)
    }

    fn erb_mask(&mut self, _enc: &EncoderOutput) -> [f32; dfn_dsp::NB_ERB] {
        let g = &self.frames[self.current].gains;
        // Lengths were checked at load, so a short list cannot leave the tail of
        // the mask holding the previous frame's values.
        let mut gains = [0.0f32; dfn_dsp::NB_ERB];
        gains.copy_from_slice(&g[..NB_ERB]);
        gains
    }

    fn df_coefs(&mut self, _enc: &EncoderOutput, out: &mut [Complex32; NB_DF * DF_ORDER]) {
        let c = &self.frames[self.current].coefs;
        for (coefficient, pair) in out.iter_mut().zip(c.as_chunks::<2>().0) {
            *coefficient = Complex32::new(pair[0], pair[1]);
        }
    }

    fn reset(&mut self) {
        self.next = 0;
        self.current = 0;
    }
}

pub fn run(trace: &Path, input: &Path, output: &Path) -> Result<(), String> {
    let audio = wav::read(input)?;
    if audio.sample_rate != 48_000 {
        return Err(format!("{} is not 48 kHz", input.display()));
    }
    if audio.channels > 1 {
        eprintln!(
            "dfmic: note: {} has {} channels, downmixed to mono",
            input.display(),
            audio.channels
        );
    }
    let replay = Replay::load(trace)?;
    let mut pipeline = DfnPipeline::new(replay, Params::default());

    let mut out = Vec::with_capacity(audio.samples.len());
    let mut block = vec![0.0f32; HOP_SIZE];
    let mut enhanced = vec![0.0f32; HOP_SIZE];
    for chunk in audio.samples.chunks(HOP_SIZE) {
        block.fill(0.);
        block[..chunk.len()].copy_from_slice(chunk);
        pipeline.process(&block, &mut enhanced);
        out.extend_from_slice(&enhanced);
    }
    out.truncate(audio.samples.len());
    wav::write_f32(output, 48_000, &out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trace frame's stage decisions come from the lists it carries, not from
    /// our own thresholds: a replay must run exactly what libDF ran.
    #[test]
    fn stages_are_recovered_from_the_trace() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("dfmic-replay-{}.trace", std::process::id()));
        let gains: Vec<String> = (0..NB_ERB)
            .map(|i| format!("{}", 0.5 + i as f32 * 0.01))
            .collect();
        let coefs: Vec<String> = (0..NB_DF * DF_ORDER * 2)
            .map(|i| format!("{}", i as f32 * 1e-3))
            .collect();
        let text = format!(
            "FRAME -30 G[{}] C[{}]\nFRAME 5 G[] C[]\nFRAME -20 G[{}] C[]\n",
            gains.join(","),
            coefs.join(","),
            vec!["0"; NB_ERB].join(","),
        );
        std::fs::write(&path, text).unwrap();
        let mut replay = Replay::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let mut enc = EncoderOutput::default();
        for expected in [Stages::ALL, Stages::NONE, Stages::ZERO] {
            replay.encode_into(&[], &[], &mut enc);
            assert_eq!(
                replay.stages(enc.lsnr),
                Some(expected),
                "a replay reports the recorded stages"
            );
        }
    }

    /// A trace whose lists are the wrong length is rejected at load, rather than
    /// silently leaving the tail of the reused mask buffer stale.
    #[test]
    fn short_lists_are_rejected() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("dfmic-replay-short-{}.trace", std::process::id()));
        std::fs::write(&path, "FRAME 0 G[0.5,0.5] C[]\n").unwrap();
        let error = match Replay::load(&path) {
            Ok(_) => panic!("a short gains list was accepted"),
            Err(error) => error,
        };
        let _ = std::fs::remove_file(&path);
        assert!(
            error.contains("gains"),
            "expected a gains length error, got {error:?}"
        );
    }
}
