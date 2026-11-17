//! Offline enhancement: file in, file out.
//!
//! This is the same engine the daemon runs, driven from a file instead of
//! PipeWire. It doubles as the parity harness's front door.

use std::path::Path;
use std::time::Instant;

use dfn_dsp::{DfnPipeline, ModelRuntime, Params, HOP_SIZE};
use dfn_npu::NpuRuntime;

use crate::wav;

pub struct Options {
    pub input: std::path::PathBuf,
    pub output: std::path::PathBuf,
    pub models: std::path::PathBuf,
    pub device: String,
    pub params: Params,
}

pub fn run(opts: &Options) -> Result<(), String> {
    let audio = wav::read(&opts.input)?;
    if audio.sample_rate != 48_000 {
        return Err(format!(
            "{}: {} Hz; DeepFilterNet needs 48 kHz (convert with `ffmpeg -ar 48000`)",
            opts.input.display(),
            audio.sample_rate
        ));
    }
    if audio.channels > 1 {
        eprintln!(
            "dfmic: note: {} has {} channels, downmixed to mono",
            opts.input.display(),
            audio.channels
        );
    }

    let runtime = NpuRuntime::load(&opts.models, &opts.device, None)?;
    let mut pipeline = DfnPipeline::new(runtime, opts.params.clone());

    // `DFMIC_TRACE=1` prints per-frame local SNR, which is what upstream's
    // `deep-filter` logs at trace level: the cheapest way to bisect a parity
    // failure into "encoder" versus "DSP".
    let trace = std::env::var("DFMIC_TRACE").is_ok();
    let mut out = Vec::with_capacity(audio.samples.len());
    let mut block = vec![0.0f32; HOP_SIZE];
    let started = Instant::now();
    let mut frames = 0u64;

    for chunk in audio.samples.chunks(HOP_SIZE) {
        block.fill(0.);
        block[..chunk.len()].copy_from_slice(chunk);
        let mut enhanced = vec![0.0f32; HOP_SIZE];
        let lsnr = pipeline.process(&block, &mut enhanced);
        if trace {
            println!("{frames} {lsnr:.4}");
        }
        out.extend_from_slice(&enhanced);
        frames += 1;
    }
    out.truncate(audio.samples.len());

    let elapsed = started.elapsed().as_secs_f64();
    let audio_secs = audio.samples.len() as f64 / 48_000.0;
    eprintln!(
        "dfmic: {frames} frames, {audio_secs:.2} s audio in {elapsed:.2} s ({:.1}x realtime)",
        audio_secs / elapsed
    );

    wav::write_f32(&opts.output, 48_000, &out)
}

/// Check that the NPU and the model artifacts are usable, without touching audio.
pub fn doctor(models: &Path, device: &str) -> Result<(), String> {
    println!("openvino devices: {:?}", dfn_npu::available_devices()?);
    for name in ["enc", "erb_dec", "df_dec"] {
        let xml = models.join(format!("{name}.xml"));
        if !xml.exists() {
            return Err(format!("missing model: {}", xml.display()));
        }
    }
    println!("models: {}", models.display());

    let started = Instant::now();
    let runtime = NpuRuntime::load(models, device, None)?;
    println!("compiled on {device} in {:.2} s", started.elapsed().as_secs_f64());

    // One second of synthetic audio, so the timing is a real end-to-end figure.
    let mut pipeline = DfnPipeline::new(runtime, Params::default());
    let block = vec![0.01f32; HOP_SIZE];
    let mut out = vec![0.0f32; HOP_SIZE];
    let n = 100;
    let t0 = Instant::now();
    for _ in 0..n {
        pipeline.process(&block, &mut out);
    }
    let per_frame = t0.elapsed().as_secs_f64() / n as f64;
    println!(
        "steady state: {:.2} ms per {:.0} ms frame ({:.1}x realtime, {:.0}% duty)",
        per_frame * 1000.0,
        HOP_SIZE as f64 / 48.0,
        0.01 / per_frame,
        per_frame * 100.0
    );
    Ok(())
}

/// Dump the DSP features for the first few hops. Used to bisect parity failures
/// against upstream libDF; `libDF/examples/dump_features.rs` prints the same.
pub fn dump_features(input: &Path, models: Option<&Path>) -> Result<(), String> {
    use dfn_dsp::{DfState, HOP_SIZE, NB_DF, NB_ERB, N_FREQS};

    let audio = wav::read(input)?;
    let mut state = DfState::new(NB_ERB, 2);
    let alpha = dfn_dsp::calc_norm_alpha(1.0);
    let mut spec = vec![num_complex::Complex32::default(); N_FREQS];
    let mut erb = [0.0f32; NB_ERB];
    let mut cplx = vec![num_complex::Complex32::default(); NB_DF];

    let mut npu = match models {
        Some(dir) => Some(DfnPipeline::new(
            NpuRuntime::load(dir, "NPU", None)?,
            Params::default(),
        )),
        None => None,
    };

    println!("first8 {:?}", &audio.samples[..8]);
    println!("len {}", audio.samples.len());
    println!("erb {:?}", state.erb);
    let hops: usize = std::env::var("DFMIC_HOPS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    for hop in 0..hops {
        let start = hop * HOP_SIZE;
        let block = &audio.samples[start..start + HOP_SIZE];
        state.analysis(block, &mut spec);
        state.feat_erb(&spec, alpha, &mut erb);
        state.feat_cplx(&spec[..NB_DF], alpha, &mut cplx);
        if hop == 0 {
            println!("erb_feat {erb:?}");
            println!("cplx {:?}", &cplx[..4]);
            println!("spec0 {:?}", &spec[..4]);
        }
        // Machine-readable per-frame features for cross-checking the network path.
        let erb_csv: Vec<String> = erb.iter().map(|v| format!("{v:.7}")).collect();
        let spec_csv: Vec<String> = cplx.iter().flat_map(|c| [format!("{:.7}", c.re), format!("{:.7}", c.im)]).collect();
        println!("F {hop} E {} S {}", erb_csv.join(","), spec_csv.join(","));

        // Same features through the NPU graphs, for cross-checking against the
        // Python runtime in tools/dfnpy (which is verified against upstream ONNX).
        if let Some(pipeline) = npu.as_mut() {
            let mut feat_spec = vec![0.0f32; NB_DF * 2];
            for (i, c) in cplx.iter().enumerate() {
                feat_spec[i] = c.re;
                feat_spec[NB_DF + i] = c.im;
            }
            let enc = pipeline.runtime_mut().encode(&erb, &feat_spec);
            println!("N {hop} lsnr {:.6} emb {:?}", enc.lsnr, &enc.emb[..4]);
        }
    }
    Ok(())
}
