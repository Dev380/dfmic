//! DeepFilterNet3_ll signal processing.
//!
//! This is a port of libDF's `DFState` and the pipeline glue from
//! `DfTract::process` (DeepFilterNet by Hendrik Schröter, MIT/Apache-2.0 — see
//! NOTICE). It is ported rather than depended upon because libDF couples its DSP
//! to tract, and we run the neural network on the NPU instead.
//!
//! Fidelity notes:
//!   * frame analysis/synthesis, the vorbis window, the ERB filter bank, the mean
//!     and unit normalisations, the interpolated band mask, the 5-tap deep filter
//!     and the post filter all follow libDF line for line;
//!   * the rolling spectrogram buffers are primed with `df_order` silent frames,
//!     exactly as `DfTract::init` does, so the first frames behave like upstream;
//!   * the neural network is abstracted behind [`ModelRuntime`], implemented for
//!     the NPU in `dfn-npu`.

use std::f32::consts::PI;
use std::sync::Arc;

use num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// Sample rate the model was trained for.
pub const SR: usize = 48_000;
/// STFT size.
pub const FFT_SIZE: usize = 960;
/// Hop size, i.e. one model frame.
pub const HOP_SIZE: usize = 480;
/// ERB bands predicted by the first decoder.
pub const NB_ERB: usize = 32;
/// Frequency bins handled by the deep filter (and by the encoder's complex input).
pub const NB_DF: usize = 96;
/// Deep filter taps.
pub const DF_ORDER: usize = 5;
/// Frequency bins of a real STFT of `FFT_SIZE`.
pub const N_FREQS: usize = FFT_SIZE / 2 + 1;

const MEAN_NORM_INIT: [f32; 2] = [-60., -90.];
const UNIT_NORM_INIT: [f32; 2] = [0.001, 0.0001];

fn freq2erb(freq_hz: f32) -> f32 {
    9.265 * (freq_hz / (24.7 * 9.265)).ln_1p()
}

fn erb2freq(n_erb: f32) -> f32 {
    24.7 * 9.265 * ((n_erb / 9.265).exp() - 1.)
}

/// Number of STFT bins in each ERB band. Sums to `N_FREQS`.
pub fn erb_fb(nb_bands: usize, min_nb_freqs: usize) -> Vec<usize> {
    let nyq_freq = SR / 2;
    let freq_width = SR as f32 / FFT_SIZE as f32;
    let erb_low = freq2erb(0.);
    let erb_high = freq2erb(nyq_freq as f32);
    let mut erb = vec![0usize; nb_bands];
    let step = (erb_high - erb_low) / nb_bands as f32;
    let min_nb_freqs = min_nb_freqs as i32;
    let mut prev_freq = 0i32;
    let mut freq_over = 0i32;
    for i in 1..=nb_bands {
        let f = erb2freq(erb_low + i as f32 * step);
        let fb = (f / freq_width).round() as i32;
        let mut nb_freqs = fb - prev_freq - freq_over;
        if nb_freqs < min_nb_freqs {
            freq_over = min_nb_freqs - nb_freqs;
            nb_freqs = min_nb_freqs;
        } else {
            freq_over = 0;
        }
        erb[i - 1] = nb_freqs as usize;
        prev_freq = fb;
    }
    erb[nb_bands - 1] += 1; // the DC/Nyquist bin layout gives one extra
    let too_large = erb.iter().sum::<usize>() - N_FREQS;
    if too_large > 0 {
        erb[nb_bands - 1] -= too_large;
    }
    debug_assert_eq!(erb.iter().sum::<usize>(), N_FREQS);
    erb
}

/// Smoothing coefficient for the normalisation states.
pub fn calc_norm_alpha(tau: f32) -> f32 {
    let dt = HOP_SIZE as f32 / SR as f32;
    let alpha = f32::exp(-dt / tau);
    let mut a = 1.0;
    let mut precision = 3;
    while a >= 1.0 {
        a = (alpha * 10i32.pow(precision) as f32).round() / 10i32.pow(precision) as f32;
        precision += 1;
    }
    a
}

/// STFT analysis/synthesis plus the normalisation states, i.e. libDF's `DFState`.
pub struct DfState {
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    wnorm: f32,
    pub erb: Vec<usize>,
    analysis_mem: Vec<f32>,
    analysis_scratch: Vec<Complex32>,
    synthesis_mem: Vec<f32>,
    synthesis_scratch: Vec<Complex32>,
    mean_norm_state: Vec<f32>,
    unit_norm_state: Vec<f32>,
    fft_in: Vec<f32>,
    fft_scratch_out: Vec<f32>,
}

impl DfState {
    pub fn new(nb_bands: usize, min_nb_freqs: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(FFT_SIZE);
        let backward = planner.plan_fft_inverse(FFT_SIZE);
        let window_size_h = FFT_SIZE / 2;

        let mut window = vec![0.0f32; FFT_SIZE];
        for (i, w) in window.iter_mut().enumerate() {
            let sin = (0.5 * PI as f64 * (i as f64 + 0.5) / window_size_h as f64).sin();
            *w = (0.5 * PI as f64 * sin * sin).sin() as f32;
        }
        let wnorm = 1. / (FFT_SIZE.pow(2) as f32 / (2 * HOP_SIZE) as f32);

        let mut s = Self {
            analysis_scratch: forward.make_scratch_vec(),
            synthesis_scratch: backward.make_scratch_vec(),
            fft_in: forward.make_input_vec(),
            fft_scratch_out: backward.make_output_vec(),
            fft_forward: forward,
            fft_inverse: backward,
            window,
            wnorm,
            erb: erb_fb(nb_bands, min_nb_freqs),
            analysis_mem: vec![0.; FFT_SIZE - HOP_SIZE],
            synthesis_mem: vec![0.; FFT_SIZE - HOP_SIZE],
            mean_norm_state: Vec::new(),
            unit_norm_state: Vec::new(),
        };
        s.init_norm_states(NB_DF);
        s
    }

    pub fn init_norm_states(&mut self, nb_df_freqs: usize) {
        let (min, max) = (MEAN_NORM_INIT[0], MEAN_NORM_INIT[1]);
        let nb_erb = self.erb.len();
        let step = (max - min) / (nb_erb - 1) as f32;
        self.mean_norm_state = (0..nb_erb).map(|i| min + i as f32 * step).collect();

        let (min, max) = (UNIT_NORM_INIT[0], UNIT_NORM_INIT[1]);
        let step = (max - min) / (nb_df_freqs - 1) as f32;
        self.unit_norm_state = (0..nb_df_freqs).map(|i| min + i as f32 * step).collect();
    }

    pub fn reset(&mut self) {
        self.analysis_mem.fill(0.);
        self.synthesis_mem.fill(0.);
        self.init_norm_states(NB_DF);
    }

    /// Windowed STFT of one hop. `output` has `N_FREQS` bins.
    pub fn analysis(&mut self, input: &[f32], output: &mut [Complex32]) {
        debug_assert_eq!(input.len(), HOP_SIZE);
        debug_assert_eq!(output.len(), N_FREQS);

        let split = FFT_SIZE - HOP_SIZE;
        for i in 0..split {
            self.fft_in[i] = self.analysis_mem[i] * self.window[i];
        }
        for i in 0..HOP_SIZE {
            self.fft_in[split + i] = input[i] * self.window[split + i];
        }
        // libDF indexes the *memory* buffer here, which is `FFT_SIZE - HOP_SIZE`
        // long: with a 960/480 STFT that means the memory is fully replaced.
        let mem_split = self.analysis_mem.len() - HOP_SIZE;
        self.analysis_mem.rotate_left(HOP_SIZE);
        self.analysis_mem[mem_split..].copy_from_slice(input);

        self.fft_forward
            .process_with_scratch(&mut self.fft_in, output, &mut self.analysis_scratch)
            .expect("forward FFT failed");
        for x in output.iter_mut() {
            *x *= self.wnorm;
        }
    }

    /// Inverse STFT and overlap-add of one hop.
    pub fn synthesis(&mut self, input: &mut [Complex32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), N_FREQS);
        debug_assert_eq!(output.len(), HOP_SIZE);

        let mut x = std::mem::take(&mut self.fft_scratch_out);
        // A single-bin input is degenerate for realfft; ignore the error the way
        // libDF does instead of failing the audio path.
        let _ = self
            .fft_inverse
            .process_with_scratch(input, &mut x, &mut self.synthesis_scratch);
        for (xi, w) in x.iter_mut().zip(self.window.iter()) {
            *xi *= w;
        }
        for i in 0..HOP_SIZE {
            output[i] = x[i] + self.synthesis_mem[i];
        }
        let split = self.synthesis_mem.len() - HOP_SIZE;
        self.synthesis_mem.rotate_left(HOP_SIZE);
        for i in 0..split {
            self.synthesis_mem[i] += x[HOP_SIZE + i];
        }
        self.synthesis_mem[split..].copy_from_slice(&x[(split + HOP_SIZE)..]);
        self.fft_scratch_out = x;
    }

    /// Log-ERB band energies with a running mean removed, in place.
    pub fn feat_erb(&mut self, input: &[Complex32], alpha: f32, output: &mut [f32]) {
        debug_assert_eq!(input.len(), N_FREQS);
        debug_assert_eq!(output.len(), NB_ERB);

        let mut bcsum = 0usize;
        for (band_size, out_b) in self.erb.iter().zip(output.iter_mut()) {
            let k = 1. / *band_size as f32;
            let mut acc = 0.0f32;
            for j in 0..*band_size {
                let idx = bcsum + j;
                acc += (input[idx].re * input[idx].re + input[idx].im * input[idx].im) * k;
            }
            *out_b = (acc + 1e-10).log10() * 10.;
            bcsum += *band_size;
        }
        for (x, s) in output.iter_mut().zip(self.mean_norm_state.iter_mut()) {
            *s = *x * (1. - alpha) + *s * alpha;
            *x -= *s;
            *x /= 40.;
        }
    }

    /// Unit-normalised complex spectrum of the deep-filter bins, in place.
    pub fn feat_cplx(&mut self, input: &[Complex32], alpha: f32, output: &mut [Complex32]) {
        debug_assert_eq!(input.len(), NB_DF);
        debug_assert_eq!(output.len(), NB_DF);
        output.copy_from_slice(input);
        for (x, s) in output.iter_mut().zip(self.unit_norm_state.iter_mut()) {
            *s = x.norm() * (1. - alpha) + *s * alpha;
            *x /= s.sqrt();
        }
    }

    /// Multiply each STFT bin by its ERB band gain.
    pub fn apply_mask(&self, output: &mut [Complex32], gains: &[f32]) {
        debug_assert_eq!(output.len(), N_FREQS);
        debug_assert_eq!(gains.len(), NB_ERB);
        let mut bcsum = 0usize;
        for (band_size, gain) in self.erb.iter().zip(gains.iter()) {
            for j in 0..*band_size {
                output[bcsum + j] *= *gain;
            }
            bcsum += *band_size;
        }
    }
}

fn fmt_list(xs: &[f32]) -> String {
    let mut s = String::with_capacity(xs.len() * 12 + 2);
    s.push('[');
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{x:?}"));
    }
    s.push(']');
    s
}

/// libDF's post filter, applied when the post-filter beta is above zero.
///
/// Deliberately processes whole groups of four bins, which leaves a trailing
/// remainder (the Nyquist bin) untouched. That is what libDF's own `post_filter`
/// does, and the point of this port is to reproduce it — the dropped bin is
/// either DC or Nyquist and carries no speech. Changing it to `as_chunks` would
/// be the same arithmetic, so the loop stays as it is and the lints are silenced
/// with the reason.
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn post_filter(noisy: &[Complex32], enh: &mut [Complex32], beta: f32) {
    let beta_p1 = beta + 1.;
    let eps = 1e-12;
    for (n, e) in noisy.chunks_exact(4).zip(enh.chunks_exact_mut(4)) {
        let mut pf = [0.0f32; 4];
        for k in 0..4 {
            let g = (e[k].norm() / (n[k].norm() + eps)).clamp(eps, 1.);
            let g_sin = g * (g * PI / 2.0).sin();
            pf[k] = (beta_p1 * g / (1. + beta * (g / g_sin).powi(2))) / g;
        }
        for k in 0..4 {
            e[k] *= pf[k];
        }
    }
}

/// One frame of analysis, as the pipeline defines it: which of the network's
/// stages apply to a hop is decided by this, not by the runtime.
///
/// The flags travel from whoever computed the local SNR to the pipeline, so a
/// runtime that replays a recorded trace reproduces the *recorded* decisions
/// instead of re-deriving them from thresholds it did not use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stages {
    /// Estimate gains: the ERB decoder runs and its mask is applied.
    pub gains: bool,
    /// Apply a zero mask: the frame is noise by the caller's own definition.
    pub zero: bool,
    /// Run the deep filter and use its coefficients.
    pub deep_filter: bool,
}

impl Stages {
    /// Everything on: the frame is in the band where all three graphs run.
    pub const ALL: Self = Self {
        gains: true,
        zero: false,
        deep_filter: true,
    };

    /// Nothing on: pass the noisy spectrum through untouched.
    pub const NONE: Self = Self {
        gains: false,
        zero: false,
        deep_filter: false,
    };

    /// Silence the frame with a zero ERB mask, which is what the deep filter
    /// alone does not do (it only replaces the low bins).
    pub const ZERO: Self = Self {
        gains: false,
        zero: true,
        deep_filter: false,
    };
}

/// What the neural network provides to the DSP.
///
/// Implementations own all recurrence state and must be fed exactly one frame per
/// call to [`DfnPipeline::process`].
pub trait ModelRuntime {
    /// Encoder forward pass for one frame, written into `out`.
    ///
    /// Takes the destination rather than returning one: this runs 100 times a
    /// second and the encoder has six outputs, so returning them meant six
    /// allocations and six copies per hop. `out` is resized on the first call
    /// and reused thereafter, each buffer to its own length.
    fn encode_into(&mut self, feat_erb: &[f32], feat_spec: &[f32], out: &mut EncoderOutput);

    /// Which stages the frame just encoded needs, if the runtime knows better
    /// than the pipeline's own thresholds.
    ///
    /// `None` — the default — means "decide it from [`Params`]", which is what
    /// a real model wants: libDF's gates are a property of the pipeline, not of
    /// the network. A runtime that replays a recorded trace returns `Some`, so a
    /// frame libDF did not mask is not zeroed here by a threshold libDF never
    /// consulted, which would make this tool invent parity failures of its own.
    fn stages(&self, lsnr: f32) -> Option<Stages> {
        let _ = lsnr;
        None
    }

    /// First-stage mask for one frame: `NB_ERB` band gains.
    fn erb_mask(&mut self, enc: &EncoderOutput) -> [f32; NB_ERB];

    /// Deep-filter coefficients for one frame, `NB_DF * DF_ORDER` complex taps.
    ///
    /// Written into `out` rather than returned for the same reason as
    /// [`Self::encode_into`]: this runs 100 times a second.
    fn df_coefs(&mut self, enc: &EncoderOutput, out: &mut [Complex32; NB_DF * DF_ORDER]);

    /// Forget the model's recurrence state (GRU hidden states and the
    /// convolution history). Called when the stream is reset.
    fn reset(&mut self);
}

impl EncoderOutput {
    /// Drop every buffer's contents, keeping the allocations.
    ///
    /// Used by a stream reset: the sizes are the model's, and only the values
    /// are stale.
    pub fn clear(&mut self) {
        for slot in [
            &mut self.emb,
            &mut self.c0,
            &mut self.e0,
            &mut self.e1,
            &mut self.e2,
            &mut self.e3,
        ] {
            slot.fill(0.0);
        }
        self.lsnr = 0.0;
    }
}

impl Default for EncoderOutput {
    fn default() -> Self {
        Self {
            emb: Vec::new(),
            c0: Vec::new(),
            lsnr: 0.0,
            e0: Vec::new(),
            e1: Vec::new(),
            e2: Vec::new(),
            e3: Vec::new(),
        }
    }
}

/// Encoder outputs for a single frame.
pub struct EncoderOutput {
    pub emb: Vec<f32>,
    pub c0: Vec<f32>,
    pub lsnr: f32,
    pub e0: Vec<f32>,
    pub e1: Vec<f32>,
    pub e2: Vec<f32>,
    pub e3: Vec<f32>,
}

/// libDF's optional reduction of channel masks before applying them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaskReduce {
    #[default]
    None,
    Max,
    Mean,
}

/// Apply libDF's `ReduceMask` operation across equally sized channel masks.
/// Mono deliberately remains unchanged.
pub fn reduce_channel_masks(masks: &mut [&mut [f32]], reduce: MaskReduce) {
    if masks.len() < 2 || reduce == MaskReduce::None {
        return;
    }
    let bands = masks[0].len();
    assert!(masks.iter().all(|mask| mask.len() == bands));
    for band in 0..bands {
        let value = match reduce {
            MaskReduce::None => unreachable!(),
            MaskReduce::Max => masks
                .iter()
                .fold(f32::NEG_INFINITY, |value, mask| value.max(mask[band])),
            MaskReduce::Mean => {
                masks.iter().map(|mask| mask[band]).sum::<f32>() / masks.len() as f32
            }
        };
        for mask in masks.iter_mut() {
            mask[band] = value;
        }
    }
}

/// Runtime parameters, mirroring libDF's `RuntimeParams`.
#[derive(Clone, Debug)]
pub struct Params {
    /// Maximum noise attenuation in dB. `100` means "no limit".
    pub atten_lim_db: f32,
    /// Post-filter beta; `0` disables the post filter.
    pub post_filter_beta: f32,
    /// Below this local SNR a zero mask is applied (frame treated as noise).
    pub min_db_thresh: f32,
    /// Above this local SNR no processing is applied (frame treated as clean).
    pub max_db_erb_thresh: f32,
    /// Above this local SNR the deep filter is skipped.
    pub max_db_df_thresh: f32,
    /// Minimum number of processed hops retained before the queue is shortened.
    pub min_buffer: u32,
    /// Cross-channel ERB mask reduction. This pipeline is mono.
    pub mask_reduce: MaskReduce,
}

impl Default for Params {
    /// libDF's `RuntimeParams::default_with_ch(1)` — the LADSPA plugin's defaults,
    /// which is what upstream `deep-filter` ships.
    fn default() -> Self {
        Self {
            atten_lim_db: 100.,
            post_filter_beta: 0.,
            min_db_thresh: -10.,
            max_db_erb_thresh: 30.,
            max_db_df_thresh: 20.,
            min_buffer: 0,
            mask_reduce: MaskReduce::None,
        }
    }
}

impl Params {
    fn atten_lim(&self) -> Option<f32> {
        let lim = self.atten_lim_db.abs();
        if lim >= 100. {
            None
        } else if lim < 0.01 {
            // 0 dB would mean "no reduction at all"; libDF forces near-passthrough.
            Some(1.)
        } else {
            Some(10f32.powf(-lim / 20.))
        }
    }

    /// Whether to estimate gains, apply a zero mask, and apply the deep filter.
    ///
    /// libDF decides this from four thresholds, and so does the daemon. It lives
    /// here rather than in the trait so a runtime that replays a trace has the
    /// same implementation available if it ever wants it.
    pub fn stages_for(&self, lsnr: f32) -> Stages {
        if lsnr < self.min_db_thresh {
            Stages::ZERO
        } else if lsnr > self.max_db_erb_thresh {
            Stages::NONE
        } else if lsnr > self.max_db_df_thresh {
            Stages {
                gains: true,
                zero: false,
                deep_filter: false,
            }
        } else {
            Stages::ALL
        }
    }
}

/// One stream of DeepFilterNet3_ll: 480 samples in, 480 samples out.
pub struct DfnPipeline<R: ModelRuntime> {
    state: DfState,
    runtime: R,
    params: Params,
    alpha: f32,
    /// The last [`DF_ORDER`] noisy spectra, oldest first: the deep filter reads
    /// them as its taps.
    rolling_x: std::collections::VecDeque<Vec<Complex32>>,
    /// The current noisy spectrum, before any mask.
    spec: Vec<Complex32>,
    /// The enhanced spectrum of the current hop. One frame, not a history:
    /// only the newest frame of the y side is ever read.
    y_spec: Vec<Complex32>,
    /// Deep-filter coefficients for the current hop, written in place.
    coefs: [Complex32; NB_DF * DF_ORDER],
    feat_erb: [f32; NB_ERB],
    feat_spec: Vec<f32>,
    /// Scratch for the complex features. Kept here rather than allocated per
    /// hop: this runs 100 times a second, and the allocation showed up in the
    /// profile (`_int_malloc`, `to_vec`) as a measurable cost.
    cplx: Vec<Complex32>,
    /// Reused encoder output buffer: see [`ModelRuntime::encode_into`].
    enc: EncoderOutput,
    /// Whether to dump frames for `replay-dsp`. Read once: `std::env::var`
    /// allocates and locks the environment, which is far too expensive to do on
    /// the audio path (it was ~1% of the worker's CPU).
    dump: bool,
}

impl<R: ModelRuntime> DfnPipeline<R> {
    pub fn new(runtime: R, params: Params) -> Self {
        let mut state = DfState::new(NB_ERB, 2);
        state.init_norm_states(NB_DF);
        // libDF primes the noisy history with `df_order` silent frames.
        let silent = vec![Complex32::default(); N_FREQS];
        Self {
            state,
            runtime,
            alpha: calc_norm_alpha(1.0),
            rolling_x: std::collections::VecDeque::from(vec![silent; DF_ORDER]),
            spec: vec![Complex32::default(); N_FREQS],
            y_spec: vec![Complex32::default(); N_FREQS],
            coefs: [Complex32::default(); NB_DF * DF_ORDER],
            feat_erb: [0.0; NB_ERB],
            feat_spec: vec![0.0; NB_DF * 2],
            cplx: vec![Complex32::default(); NB_DF],
            enc: EncoderOutput::default(),
            dump: std::env::var_os("DFMIC_DUMP").is_some(),
            params,
        }
    }

    pub fn params(&self) -> &Params {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    pub fn runtime_mut(&mut self) -> &mut R {
        &mut self.runtime
    }

    /// Process one hop of 48 kHz mono audio. Returns the local SNR estimate.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> f32 {
        debug_assert_eq!(input.len(), HOP_SIZE);
        debug_assert_eq!(output.len(), HOP_SIZE);

        // Analysis runs for every hop, silent or not. It and the synthesis share
        // the overlap-add, so a hop that skipped one would splice the next frame
        // against a stale tail: on a pause, that is a click when speech resumes.
        // Only the network is skipped on silence.
        self.state.analysis(input, &mut self.spec);

        // The deep filter reads the last `DF_ORDER` noisy frames, so the history
        // advances on silent hops too: a gap has to be in the history *as*
        // silence, not as "the audio from before the gap". The oldest frame is
        // recycled rather than dropped, so this costs no allocation per hop.
        let mut recycled = self.rolling_x.pop_front().expect("history is primed");
        recycled.copy_from_slice(&self.spec);
        self.rolling_x.push_back(recycled);

        let lim = self.params.atten_lim();
        let energy: f32 = input.iter().map(|x| x * x).sum();
        let silent = energy / (input.len() as f32) < 1e-7;

        // Hops the network has nothing to do with: digital silence, and a user
        // who asked for no attenuation at all. Both are pass-throughs, and both
        // still go through the synthesis. That is not politeness: the frame
        // being synthesised carries the *previous* hop's audio (the analysis has
        // one hop of latency), so skipping the synthesis would drop that hop,
        // and masking it to zero would drop it just as surely. Either way the
        // next frame is spliced against a stale tail, which is a click.
        let passthrough_lsnr = if silent {
            Some(-15.)
        } else if lim == Some(1.) {
            Some(35.)
        } else {
            None
        };
        if let Some(lsnr) = passthrough_lsnr {
            self.y_spec.copy_from_slice(&self.spec);
            self.state.synthesis(&mut self.y_spec, output);
            return lsnr;
        }

        // Features and the network's first pass.
        self.state
            .feat_erb(&self.spec, self.alpha, &mut self.feat_erb);
        self.state
            .feat_cplx(&self.spec[..NB_DF], self.alpha, &mut self.cplx);
        // The encoder wants real/imaginary planes: [2, nb_df].
        for (i, c) in self.cplx.iter().enumerate() {
            self.feat_spec[i] = c.re;
            self.feat_spec[NB_DF + i] = c.im;
        }

        self.runtime
            .encode_into(&self.feat_erb, &self.feat_spec, &mut self.enc);
        let lsnr = self.enc.lsnr;
        let stages = self
            .runtime
            .stages(lsnr)
            .unwrap_or_else(|| self.params.stages_for(lsnr));

        let gains = if stages.gains {
            let mut gains = self.runtime.erb_mask(&self.enc);
            // This is the same operation libDF inserts in its multi-channel
            // decoder graph. One channel is intentionally a no-op.
            reduce_channel_masks(&mut [gains.as_mut_slice()], self.params.mask_reduce);
            Some(gains)
        } else if stages.zero {
            Some([0.0f32; NB_ERB])
        } else {
            None
        };
        let apply_df = stages.deep_filter;
        if apply_df {
            self.runtime.df_coefs(&self.enc, &mut self.coefs);
        }

        if self.dump {
            // Same format as libDF's DF_DUMP, so `replay-dsp` can consume either.
            let g: Vec<f32> = gains.map(|g| g.to_vec()).unwrap_or_default();
            let c: Vec<f32> = if apply_df {
                self.coefs.iter().flat_map(|z| [z.re, z.im]).collect()
            } else {
                Vec::new()
            };
            eprintln!("FRAME {lsnr} G{} C{}", fmt_list(&g), fmt_list(&c));
        }

        self.y_spec.copy_from_slice(&self.spec);

        // Stage 1: band gains on the current frame.
        if let Some(gains) = gains.as_ref() {
            self.state.apply_mask(&mut self.y_spec, gains);
        }

        // Stage 2: deep filter replaces the low bins using the last `DF_ORDER`
        // noisy frames, oldest first.
        if apply_df {
            self.y_spec[..NB_DF].fill(Complex32::default());
            for (tap, frame) in self.rolling_x.iter().enumerate() {
                for f in 0..NB_DF {
                    self.y_spec[f] += frame[f] * self.coefs[f * DF_ORDER + tap];
                }
            }
        }

        if stages.gains && self.params.post_filter_beta > 0. {
            let noisy = self.rolling_x.back().expect("history is primed");
            post_filter(noisy, &mut self.y_spec, self.params.post_filter_beta);
        }

        // Optional attenuation limit: mix the noisy spectrum back in.
        if let Some(lim) = lim {
            let noisy = self.rolling_x.back().expect("history is primed");
            for (e, n) in self.y_spec.iter_mut().zip(noisy.iter()) {
                *e = *e * (1. - lim) + n * lim;
            }
        }

        self.state.synthesis(&mut self.y_spec, output);
        lsnr
    }

    /// Forget everything about the stream so far, in this pipeline and in the
    /// model.
    ///
    /// Both halves matter: the STFT and normalisation states here, and the GRU
    /// and convolution history inside the runtime. A capture device that was
    /// switched, or a stream that restarted, has nothing to do with the audio
    /// that came before it.
    pub fn reset(&mut self) {
        self.state.reset();
        self.runtime.reset();
        self.enc.clear();
        for frame in self.rolling_x.iter_mut() {
            frame.fill(Complex32::default());
        }
        self.spec.fill(Complex32::default());
        self.y_spec.fill(Complex32::default());
        self.coefs = [Complex32::default(); NB_DF * DF_ORDER];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The STFT pair must reconstruct the signal with the vorbis window at 50%
    /// overlap. This is the single most load-bearing property of the DSP port: if
    /// the normalisation or the overlap-add is wrong, audio comes out quiet, loud
    /// or buzzing.
    #[test]
    fn analysis_synthesis_round_trip() {
        let mut state = DfState::new(NB_ERB, 2);
        let mut signal = Vec::new();
        let mut x = 0.0f32;
        for i in 0..HOP_SIZE * 20 {
            // A sweep, so a broken overlap-add cannot hide behind silence.
            let f = 200.0 + 3000.0 * (i as f32 / (HOP_SIZE * 20) as f32);
            x += 2.0 * PI * f / SR as f32;
            signal.push(0.5 * x.sin());
        }

        let mut out = Vec::new();
        for hop in signal.as_chunks::<HOP_SIZE>().0 {
            let mut spec = vec![Complex32::default(); N_FREQS];
            state.analysis(hop, &mut spec);
            let mut block = vec![0.0f32; HOP_SIZE];
            state.synthesis(&mut spec, &mut block);
            out.extend_from_slice(&block);
        }

        // Skip the first frames: the analysis memory starts at zero.
        let skip = HOP_SIZE * 4;
        let sig: f32 = signal[skip..].iter().map(|a| a * a).sum();
        let mut best = (f32::NEG_INFINITY, 0usize);
        for lag in 0..=(HOP_SIZE * 3) {
            let n = signal.len().saturating_sub(skip + lag);
            if n < HOP_SIZE * 4 {
                break;
            }
            let err: f32 = signal[skip..skip + n]
                .iter()
                .zip(out[skip + lag..skip + lag + n].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let sdr = 10.0 * (sig / err).log10();
            if sdr > best.0 {
                best = (sdr, lag);
            }
        }
        println!("best lag {} samples -> {:.1} dB", best.1, best.0);
        assert!(
            best.0 > 100.0,
            "round trip SDR only {:.1} dB at lag {}",
            best.0,
            best.1
        );
    }

    #[test]
    fn erb_bands_cover_every_bin() {
        let erb = erb_fb(NB_ERB, 2);
        assert_eq!(erb.len(), NB_ERB);
        assert_eq!(erb.iter().sum::<usize>(), N_FREQS);
        assert!(erb.iter().all(|&b| b >= 2));
    }

    #[test]
    fn norm_alpha_matches_upstream() {
        // libDF's rounding loop lands on 0.99 for tau=1 at 48 kHz / 480 hop.
        assert!((calc_norm_alpha(1.0) - 0.99).abs() < 1e-6);
    }

    /// A runtime that does nothing to the signal, so the pipeline is a plain
    /// STFT round trip: it reports every stage off, so no gains, no deep filter
    /// and no post filter are applied whatever the frame's SNR is.
    struct Passthrough;

    impl ModelRuntime for Passthrough {
        fn encode_into(&mut self, _feat_erb: &[f32], _feat_spec: &[f32], out: &mut EncoderOutput) {
            out.lsnr = 100.0;
        }

        /// Every stage off, whatever the frame's SNR is: the pipeline below is a
        /// plain STFT round trip.
        fn stages(&self, _lsnr: f32) -> Option<Stages> {
            Some(Stages::NONE)
        }

        fn erb_mask(&mut self, _enc: &EncoderOutput) -> [f32; NB_ERB] {
            [1.0; NB_ERB]
        }

        fn df_coefs(&mut self, _enc: &EncoderOutput, _out: &mut [Complex32; NB_DF * DF_ORDER]) {}

        fn reset(&mut self) {}
    }

    /// Best signal-to-distortion ratio between `input` and `output` over a range
    /// of lags, plus the lag that achieved it.
    fn best_sdr(input: &[f32], output: &[f32], skip: usize, max_lag: usize) -> (f32, usize) {
        let sig: f32 = input[skip..].iter().map(|a| a * a).sum();
        let mut best = (f32::NEG_INFINITY, 0usize);
        for lag in 0..=max_lag {
            let n = input.len().saturating_sub(skip + lag);
            if n < HOP_SIZE * 4 {
                break;
            }
            let err: f32 = input[skip..skip + n]
                .iter()
                .zip(output[skip + lag..skip + lag + n].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let sdr = 10.0 * (sig / err).log10();
            if sdr > best.0 {
                best = (sdr, lag);
            }
        }
        best
    }

    /// A silent hop must not break the overlap-add.
    ///
    /// Synthesis adds each hop to the tail of the one before it. A hop that
    /// skipped synthesis, or wrote zeros over it, leaves the next frame spliced
    /// against a stale tail: audible as a click when speech resumes after a
    /// pause. With a pass-through runtime the pipeline is a delayed copy of the
    /// input, so a gap in the input has to be a gap in the output and nothing
    /// more.
    #[test]
    fn silence_keeps_the_overlap_add_continuous() {
        let mut pipeline = DfnPipeline::new(Passthrough, Params::default());
        let hops = 24;
        let gap = 8..12;
        let mut signal = Vec::new();
        let mut x = 0.0f32;
        for hop in 0..hops {
            for _ in 0..HOP_SIZE {
                // A sweep, so a splice shows up as an error rather than as a
                // coincidentally continuous tone.
                let f = 200.0 + 3000.0 * (hop as f32 / hops as f32);
                x += 2.0 * PI * f / SR as f32;
                signal.push(if gap.contains(&hop) {
                    0.0
                } else {
                    0.5 * x.sin()
                });
            }
        }

        let mut out = Vec::new();
        let mut block = vec![0.0f32; HOP_SIZE];
        let mut enhanced = vec![0.0f32; HOP_SIZE];
        for hop in signal.as_chunks::<HOP_SIZE>().0 {
            block.copy_from_slice(hop);
            pipeline.process(&block, &mut enhanced);
            out.extend_from_slice(&enhanced);
        }

        let skip = HOP_SIZE * 4;
        let (sdr, lag) = best_sdr(&signal, &out, skip, HOP_SIZE * 3);
        println!("silence gap: {sdr:.1} dB at lag {lag}");
        assert!(
            sdr > 90.0,
            "a silent hop broke the overlap-add: only {sdr:.1} dB at lag {lag}"
        );
    }
}
