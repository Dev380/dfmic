//! Meter ballistics, computed on the UI thread at a steady frame rate.
//!
//! Like a hardware meter: the bar jumps up instantly and falls at a fixed rate
//! in dB, the peak marker holds before falling, the numeric readout follows the
//! held peak (so it does not flicker), and clipping latches for a few seconds.

use std::time::{Duration, Instant};

/// How fast the bar falls once the signal drops.
const RELEASE_DB_PER_S: f32 = 24.0;
/// How long the peak marker holds before it falls, and how fast it then falls.
const HOLD: Duration = Duration::from_millis(1500);
const HOLD_RELEASE_DB_PER_S: f32 = 12.0;
/// How long a clip stays lit after the last clipped sample.
const CLIP_LATCH: Duration = Duration::from_secs(3);
/// The readout changes at most this often.
const READOUT_PERIOD: Duration = Duration::from_millis(200);
/// Where the bars bottom out; matches the drawn scale's floor with a margin so
/// a falling bar leaves the scale smoothly instead of stopping at its edge.
pub const BOTTOM_DB: f32 = -100.0;

pub fn to_db(linear: f64) -> f32 {
    if linear <= 0.0 {
        f32::NEG_INFINITY
    } else {
        (20.0 * linear.log10()) as f32
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Reading {
    /// Real samples with these linear amplitudes.
    Signal { peak: f64, rms: f64, clip: bool },
    /// Samples are flowing and every one is exactly zero.
    Silence,
    /// Nothing to show (no samples, not in use, no daemon).
    Nothing,
}

pub struct Ballistics {
    pub rms_db: f32,
    pub peak_db: f32,
    pub hold_db: f32,
    hold_until: Instant,
    clip_until: Option<Instant>,
    readout_db: Option<f32>,
    readout_at: Instant,
}

impl Ballistics {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            rms_db: BOTTOM_DB,
            peak_db: BOTTOM_DB,
            hold_db: BOTTOM_DB,
            hold_until: now,
            clip_until: None,
            readout_db: None,
            readout_at: now,
        }
    }

    pub fn step(&mut self, now: Instant, dt: f32, reading: Reading) {
        let (peak, rms) = match reading {
            Reading::Signal { peak, rms, clip } => {
                if clip {
                    self.clip_until = Some(now + CLIP_LATCH);
                }
                (to_db(peak).max(BOTTOM_DB), to_db(rms).max(BOTTOM_DB))
            }
            Reading::Silence | Reading::Nothing => (BOTTOM_DB, BOTTOM_DB),
        };

        let fall = RELEASE_DB_PER_S * dt;
        self.peak_db = if peak >= self.peak_db {
            peak
        } else {
            (self.peak_db - fall).max(peak)
        };
        self.rms_db = if rms >= self.rms_db {
            rms
        } else {
            (self.rms_db - fall).max(rms)
        };

        if peak >= self.hold_db {
            self.hold_db = peak;
            self.hold_until = now + HOLD;
        } else if now >= self.hold_until {
            self.hold_db = (self.hold_db - HOLD_RELEASE_DB_PER_S * dt).max(peak);
        }

        if now.duration_since(self.readout_at) >= READOUT_PERIOD || self.readout_db.is_none() {
            self.readout_at = now;
            self.readout_db = match reading {
                Reading::Signal { .. } => Some(self.hold_db),
                _ => None,
            };
        }
    }

    pub fn clipped(&self, now: Instant) -> bool {
        self.clip_until.is_some_and(|until| now < until)
    }

    pub fn reset_clip(&mut self) {
        self.clip_until = None;
    }

    /// The steady numeric readout, e.g. "−23.4 dBFS".
    pub fn readout(&self, reading: Reading) -> String {
        match (reading, self.readout_db) {
            (Reading::Silence, _) => "−∞ dBFS".into(),
            (Reading::Nothing, _) => "—".into(),
            (Reading::Signal { .. }, Some(db)) if db > BOTTOM_DB => format_dbfs(db),
            (Reading::Signal { .. }, _) => format!("< {} dBFS", minus(BOTTOM_DB as f64, 0)),
        }
    }
}

fn format_dbfs(db: f32) -> String {
    format!("{} dBFS", minus(db as f64, 1))
}

/// Format with a real minus sign, which is the width of a digit in the font.
pub fn minus(value: f64, decimals: usize) -> String {
    let text = format!("{:.*}", decimals, value);
    match text.strip_prefix('-') {
        // "-0.0" is noise; show it as zero.
        Some(rest) if rest.chars().all(|c| c == '0' || c == '.') => rest.to_string(),
        Some(rest) => format!("−{rest}"),
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minus_sign_and_negative_zero() {
        assert_eq!(minus(-23.44, 1), "−23.4");
        assert_eq!(minus(-0.01, 1), "0.0");
        assert_eq!(minus(6.0, 1), "6.0");
    }
}
