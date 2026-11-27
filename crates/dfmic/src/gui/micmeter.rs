//! The input meter, captured by the window itself.
//!
//! The daemon does not meter its input: measuring the microphone is only worth
//! doing while somebody is looking at a meter, and the window is the only thing
//! that ever looks. So it captures the physical microphone directly, exactly as
//! any other application would, and the daemon's capture stream can be paused
//! when nothing needs audio.
//!
//! Capture is `pw-record` writing raw `f32` to stdout; this module reads it in
//! fixed windows and keeps the loudest and the mean square of the latest one.
//! That is a few lines against a second PipeWire event loop inside the UI
//! process, and it cannot block the window: the reader thread only ever writes
//! to an atomic slot.
//!
//! It captures **two** channels and averages them here, which is the part that
//! makes the microphone bar comparable with "sent to apps". `pw-record
//! --channels 1` does not average the pair: it mixes with a coefficient of about
//! 0.707 per channel, so the result is hotter than the true mono mix by however
//! correlated the pair happens to be — measured on this microphone at +2.9 dB
//! when the two channels tracked each other closely and +1.2 dB when they did
//! not. The daemon's capture stream asks for MONO and gets the average, so a
//! meter fed by `--channels 1` read the microphone louder than the signal being
//! filtered, and the model made it worse by attenuating about 6 dB less on the
//! hotter copy, because its decision depends on level.

use std::io::Read;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Sample rate of the capture, and of the window the meter averages over.
///
/// A ~33 ms window matches the frame rate the meters are drawn at, so each frame
/// has fresh audio without the meter lagging behind it.
const RATE: usize = 48_000;
const WINDOW: usize = RATE / 30;

/// Channels captured, then averaged down to the mono level the meter shows.
const CHANNELS: usize = 2;

/// The latest level, written by the reader thread and read by the UI thread.
///
/// Lock-free on purpose: the UI thread polls this 30 times a second and must
/// never wait for the reader, which is blocked in a `read` most of the time.
pub struct InputLevel {
    peak_bits: AtomicU32,
    rms_bits: AtomicU32,
    clip: AtomicBool,
    /// When the last window was published, in milliseconds since `started`.
    ///
    /// Stored as a timestamp rather than as an age because the writer cannot know
    /// when the reader will look; `read` turns it into an age.
    at_ms: AtomicU32,
    started: Mutex<Instant>,
}

impl InputLevel {
    fn new() -> Self {
        Self {
            peak_bits: AtomicU32::new(0),
            rms_bits: AtomicU32::new(0),
            clip: AtomicBool::new(false),
            at_ms: AtomicU32::new(0),
            started: Mutex::new(Instant::now()),
        }
    }

    fn publish(&self, peak: f32, rms: f32, clip: bool) {
        self.peak_bits.store(peak.to_bits(), Ordering::Relaxed);
        self.rms_bits.store(rms.to_bits(), Ordering::Relaxed);
        self.clip.store(clip, Ordering::Relaxed);
        let ms = self
            .started
            .lock()
            .map(|s| s.elapsed().as_millis())
            .unwrap_or(0);
        // Stored as `ms + 1`, so that zero keeps meaning "nothing published
        // yet": a window published inside the first millisecond would otherwise
        // be indistinguishable from no window at all, and the meter would sit
        // blank until the second window arrived.
        let stored = ms.min(u32::MAX as u128 - 1) as u32 + 1;
        self.at_ms.store(stored, Ordering::Release);
    }

    /// The level, and how long ago it was measured.
    ///
    /// `None` until the first window has been published, so a freshly started
    /// capture reads as "no signal yet" rather than as silence.
    pub fn read(&self) -> Option<(Duration, f32, f32, bool)> {
        let published = self.at_ms.load(Ordering::Acquire);
        if published == 0 {
            return None;
        }
        let published = (published - 1) as u64;
        // Both clocks start at `started`, so the difference is the age.
        let now = self
            .started
            .lock()
            .map(|started| started.elapsed().as_millis() as u64)
            .unwrap_or(0);
        let age = now.saturating_sub(published);
        Some((
            Duration::from_millis(age),
            f32::from_bits(self.peak_bits.load(Ordering::Relaxed)),
            f32::from_bits(self.rms_bits.load(Ordering::Relaxed)),
            self.clip.load(Ordering::Relaxed),
        ))
    }
}

/// A running capture of one microphone.
pub struct MicMeter {
    child: Child,
    level: Arc<InputLevel>,
    target: String,
}

impl MicMeter {
    /// Start capturing `target`.
    ///
    /// An empty target means "whatever the system default is", which is what the
    /// window wants while no daemon is running: there is no device list to pick
    /// from, and the default source is still the real microphone. (While a daemon
    /// *is* running it may have taken the default over, which is why the caller
    /// passes the device name explicitly then.)
    pub fn start(target: &str) -> Option<Self> {
        if target == crate::proto::NODE_NAME {
            return None;
        }
        // SAFETY: the only unsafe call is `pre_exec` below, which runs
        // `prctl(PR_SET_PDEATHSIG)` — async-signal-safe, and touching nothing
        // this process owns.
        let mut child = unsafe {
            let mut command = Command::new("pw-record");
            // `--raw` is what makes stdout a plain sample stream. Without it
            // pw-record wraps stdout in an AU (`.snd`) container, whose header
            // would be read as audio — measured as a peak of 2.7e34.
            command.arg("--raw");
            // No `--target` at all is how pw-record is told to follow the default
            // source; an empty one would be a device name that does not exist.
            if !target.is_empty() {
                command.arg("--target").arg(target);
                // A target is not enough to pin a stream: WirePlumber re-links it
                // to the new default whenever that changes, and dfmic makes
                // itself the default. Without this the meter silently followed
                // the takeover and metered the *filtered* node, so both bars
                // showed the same signal and the microphone read as silence.
                // Measured: linked to `dfmic-source:capture_MONO` after takeover
                // without it, `alsa_input…:capture_FR` with it.
                command.arg("-P").arg("node.dont-reconnect=true");
            }
            command
                .arg("--rate")
                .arg(RATE.to_string())
                // Two channels, mixed in `read_loop`. See the module docs: asking
                // pw-record for one channel is not the same downmix as the
                // daemon's, and the difference reaches 3 dB.
                .arg("--channels")
                .arg(CHANNELS.to_string())
                .arg("--format")
                .arg("f32")
                .arg("-")
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                // If the window is killed the child must not outlive it: an orphan
                // keeps the microphone open and keeps costing CPU, and nothing is
                // left to notice. The kernel does this for us.
                .pre_exec(|| {
                    libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                    Ok(())
                })
                .spawn()
                .ok()?
        };
        let mut stdout = child.stdout.take()?;
        let level = Arc::new(InputLevel::new());
        let slot = Arc::clone(&level);
        std::thread::Builder::new()
            .name("dfmic-input-meter".into())
            .spawn(move || read_loop(&mut stdout, &slot))
            .ok()?;
        Some(Self {
            child,
            level,
            target: target.to_string(),
        })
    }

    pub fn level(&self) -> &Arc<InputLevel> {
        &self.level
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    /// Whether the capture is still alive.
    pub fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for MicMeter {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn read_loop(stdout: &mut impl Read, level: &InputLevel) {
    // One window of interleaved frames, read as bytes and reinterpreted:
    // `pw-record` writes native-endian `f32`, which is what `from_ne_bytes`
    // expects. `bytes` is a whole number of frames, so a short read can never
    // leave the next window starting mid-sample.
    let mut bytes = vec![0u8; WINDOW * CHANNELS * 4];
    loop {
        let mut filled = 0;
        while filled < bytes.len() {
            match stdout.read(&mut bytes[filled..]) {
                Ok(0) => return,
                Ok(n) => filled += n,
                Err(_) => return,
            }
        }
        let mut peak = 0.0f32;
        let mut sum = 0.0f64;
        // Clipping is judged on the channels, not on their average: one channel
        // pinned at full scale is a distorted microphone even if the mean is not.
        let mut clipped = false;
        for frame in bytes.as_chunks::<{ CHANNELS * 4 }>().0 {
            let mut mixed = 0.0f32;
            for channel in frame.as_chunks::<4>().0 {
                let sample = f32::from_ne_bytes([channel[0], channel[1], channel[2], channel[3]]);
                clipped |= sample.abs() >= 1.0;
                mixed += sample;
            }
            let sample = mixed / CHANNELS as f32;
            peak = peak.max(sample.abs());
            sum += (sample as f64) * (sample as f64);
        }
        let rms = (sum / WINDOW as f64).sqrt() as f32;
        level.publish(peak, rms, clipped || peak >= 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn stereo(left: f32, right: f32) -> Cursor<Vec<u8>> {
        let mut bytes = Vec::with_capacity(WINDOW * CHANNELS * 4);
        for _ in 0..WINDOW {
            for channel in [left, right] {
                bytes.extend_from_slice(&channel.to_ne_bytes());
            }
        }
        Cursor::new(bytes)
    }

    /// The meter must show the average of the pair, not either channel and not
    /// their sum.
    ///
    /// The two channels are deliberately different: with equal ones the average,
    /// either channel and the mono misreading of the interleaved stream all
    /// agree, and the test would pass however the samples were mixed. With
    /// `1.0` and `0.0` they do not — the average is 0.5, taking one channel is
    /// 1.0, and reading the stream as mono alternates between the two for an RMS
    /// of 0.707.
    #[test]
    fn the_meter_averages_the_channels() {
        let level = InputLevel::new();
        read_loop(&mut stereo(1.0, 0.0), &level);
        let (_, peak, rms, _) = level.read().expect("one window was published");
        assert!(
            (rms - 0.5).abs() < 1e-6,
            "rms was {rms}, not the 0.5 average"
        );
        assert!(
            (peak - 0.5).abs() < 1e-6,
            "peak was {peak}, not the 0.5 average"
        );
    }

    /// One channel at full scale is a clipped microphone even though the mean is
    /// only half of it.
    #[test]
    fn clipping_is_judged_on_the_channels() {
        let level = InputLevel::new();
        read_loop(&mut stereo(1.0, 0.0), &level);
        let (_, _, _, clip) = level.read().expect("one window was published");
        assert!(clip, "a channel at full scale was not reported as clipping");
    }
}
