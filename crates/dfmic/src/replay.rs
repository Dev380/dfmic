//! Replay a recorded network trace through the DSP.
//!
//! `dfmic replay-dsp` feeds gains/coefs captured from upstream libDF into our
//! signal processing, so a parity failure can be attributed to the DSP rather
//! than to the network. The trace is produced by building libDF with
//! `DF_DUMP=<path>` (see the README) and contains one `FRAME` line per hop.
//!
//! This is a development tool: it exists to keep the parity gate honest.

use std::path::Path;

use dfn_dsp::{DfnPipeline, EncoderOutput, ModelRuntime, Params, DF_ORDER, HOP_SIZE, NB_DF};
use num_complex::Complex32;

use crate::wav;

/// One frame of recorded network output.
struct Frame {
    lsnr: f32,
    gains: Vec<f32>,
    coefs: Vec<f32>,
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
            frames.push(Frame {
                lsnr: lsnr
                    .trim()
                    .parse()
                    .map_err(|_| format!("bad lsnr: {line}"))?,
                gains: parse_list(gains)?,
                coefs: parse_list(coefs)?,
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
    if s.is_empty() {
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

    fn erb_mask(&mut self, _enc: &EncoderOutput) -> [f32; dfn_dsp::NB_ERB] {
        let mut gains = [0.0f32; dfn_dsp::NB_ERB];
        let g = &self.frames[self.current].gains;
        let n = g.len().min(gains.len());
        gains[..n].copy_from_slice(&g[..n]);
        gains
    }

    fn df_coefs(&mut self, _enc: &EncoderOutput) -> Vec<Complex32> {
        let c = &self.frames[self.current].coefs;
        c.chunks_exact(2)
            .map(|p| Complex32::new(p[0], p[1]))
            .take(NB_DF * DF_ORDER)
            .collect()
    }
}

pub fn run(trace: &Path, input: &Path, output: &Path) -> Result<(), String> {
    let audio = wav::read(input)?;
    if audio.sample_rate != 48_000 {
        return Err(format!("{} is not 48 kHz", input.display()));
    }
    let replay = Replay::load(trace)?;
    let mut pipeline = DfnPipeline::new(
        replay,
        Params {
            atten_lim_db: 100.,
            post_filter_beta: 0.,
            min_db_thresh: -15.,
            max_db_erb_thresh: 35.,
            max_db_df_thresh: 35.,
            min_buffer: 0,
            mask_reduce: dfn_dsp::MaskReduce::None,
        },
    );

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
