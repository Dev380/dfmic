//! Voice-gated levelling after the suppressor.
//!
//! The AGC answers one question — is the speech reaching the application at a
//! usable level — and it can only answer it from two things it can observe: how
//! loud the microphone hop was, and how loud the hop the suppressor produced is.
//! Everything here follows from keeping those two straight.
//!
//! **The target is measured on the post-suppression signal.** That is the signal
//! the gain is applied to, and therefore the only one the AGC can control.
//! Measuring the raw input instead treats the suppressor's attenuation as a
//! deficit and puts it back: measured over 11 s of a quiet room, suppression
//! alone gave -3.6 dB and the old input-referenced target turned that into
//! +12.4 dB of noise.
//!
//! **The gate is a level test, not an SNR test.** `lsnr >= 0` is not a voice
//! detector: over the same 11 s of a quiet room the model's local SNR has a
//! median of -9 dB, but it reaches +19 dB and is above zero in a third of hops.
//! No threshold on it separates speech from room. Level does separate them, so
//! the gate compares the input against a tracked room floor; `lsnr` is kept only
//! as a veto, at `min_db_thresh`, below which the DSP zero-masks the frame
//! itself — that is noise by the model's own definition.
//!
//! **The floor is the minimum of the smoothed input level over ten seconds.**
//! The smoothing matters: unsmoothed hop levels in that quiet room spread from
//! -55 dBFS (p01) to -35 dBFS (p99), and their minimum sits 8.6 dB below the
//! median, so any margin over it is mostly measuring hop-to-hop variance.
//! Smoothed, the minimum sits 2.7 dB below the median and a margin means what it
//! says. The window matters for the other direction: a minimum can jump *down*
//! the instant a quieter hop arrives but only rises when the oldest entry
//! expires, which is exactly the asymmetry a noise floor needs. A slow-rise
//! tracker would climb into the speaker during a sentence and shut the gate on
//! them.
//!
//! **The gain is held across the gaps inside a sentence and released to unity
//! after that.** Holding it forever — which is what this code used to do, since
//! the only write to `gain_db` was inside the gate — is what let one quiet
//! moment amplify the room for the rest of the session.

/// The level speech should reach, RMS.
const TARGET_DB: f32 = -18.0;

/// Bounds on the correction. A -45 dBFS room and an -18 dBFS target asks for
/// +27 dB, so the ceiling is what bounds how far the room could be lifted if the
/// gate ever opened on it. -12 dB is enough to pull a shout back to the target.
const MIN_GAIN_DB: f32 = -12.0;
const MAX_GAIN_DB: f32 = 18.0;

/// The loudest the amplified room may ever come out.
///
/// The gate is the first line of defence, but it is a binary decision made from
/// two estimates and it can be wrong — measured on this machine, a room at
/// -45 dBFS was lifted 12 dB and held there by an AGC whose gate was open a
/// third of the time. This is a ceiling that cannot be wrong, because it is
/// computed from the tracked floor rather than from a decision: whatever the
/// gate says, `gain + floor` never exceeds this.
///
/// WebRTC's AGC2 does the same thing for the same reason (`LimitGainByNoise`:
/// `gain + noise` may never exceed a fixed level), which is why a mis-gated AGC
/// there cannot amplify the room however confident its VAD is. It only ever
/// limits boost, never forces attenuation: in a room already above the ceiling
/// the correct response is "do not boost", not "push it down".
const NOISE_CEILING_DB: f32 = -30.0;

/// Gain smoothing, per hop. The asymmetry is the point: a hop that arrives too
/// loud is pulled down in about 0.2 s, so it cannot run into the limiter, while
/// a quiet one is brought up over about a second, so the gain does not chase the
/// dips inside a phrase. The other way round — boost fast, cut slowly — overshot
/// the target by 6 dB on real speech and left the limiter working on 13% of
/// hops.
const BOOST: f32 = 0.01;
const CUT: f32 = 0.05;

/// Measurement smoothing, about 100 ms. The correction follows the phrase, not
/// the hop: an unsmoothed hop RMS of speech swings 10 dB and the gain jitters
/// with it.
const LEVEL_SMOOTHING: f32 = 0.1;

/// The floor window: ten seconds, so any pause inside a sentence re-seats the
/// floor to the room.
const FLOOR_WINDOW_HOPS: usize = 1000;

/// How far above the floor the input must be for the gate to open. Speech sits
/// 15-20 dB above a quiet room, so this leaves room for a quiet speaker while
/// keeping the room itself shut.
const GATE_MARGIN_DB: f32 = 12.0;

/// A pause shorter than this is inside a sentence: between words is 100-200 ms,
/// between sentences 400-800 ms.
const HOLD_HOPS: u32 = 60;

/// Letting go once the hold has expired: 4 dB/s, so a full 18 dB in 4.5 s. Slow
/// enough to be a fade rather than a pump, fast enough that the gain does not
/// stay up for the rest of the session.
const RELEASE_DB_PER_HOP: f32 = 0.04;

/// An RMS at or below this (about -140 dBFS) is no signal at all: there is no
/// level to compute, and nothing to track a floor from.
const SILENCE: f32 = 1e-7;

/// What one hop tells the AGC about itself.
pub struct Hop {
    /// The model's local SNR for this hop.
    pub lsnr: f32,
    /// RMS of the hop as captured, before suppression.
    pub input_rms: f32,
    /// Below this local SNR the DSP zero-masks the frame, so the model itself
    /// calls it noise.
    pub min_db_thresh: f32,
}

pub struct Agc {
    gain_db: f32,
    limiter_gain: f32,
    previous: f32,
    input_level: Option<f32>,
    output_level: Option<f32>,
    floor: Vec<f32>,
    floor_next: usize,
    floor_db: f32,
    closed_hops: u32,
}

impl Agc {
    pub fn new() -> Self {
        Self {
            gain_db: 0.0,
            limiter_gain: 1.0,
            previous: 0.0,
            input_level: None,
            output_level: None,
            floor: vec![f32::INFINITY; FLOOR_WINDOW_HOPS],
            floor_next: 0,
            floor_db: f32::INFINITY,
            closed_hops: 0,
        }
    }

    /// Level one hop and apply the result. `samples` is the hop *after*
    /// suppression and before gain; the gain is measured on it and applied to it.
    /// Returns the gain in dB and whether the gate was open.
    pub fn process(&mut self, samples: &mut [f32], hop: Hop, enabled: bool) -> (f32, bool) {
        if !enabled {
            // Nothing is applied and nothing is remembered: switching the AGC
            // back on starts from unity and from an unknown room, not from
            // whatever the gain and the floor happened to be. Audio passes
            // through untouched — the limiter included, because a user who
            // turned the levelling off is not asking for a peak limiter either.
            self.forget();
            self.limiter_gain = 1.0;
            self.previous = 0.0;
            return (0.0, false);
        }

        // Track the floor from the *input*: the room is a property of the room,
        // not of what the suppressor did to it.
        let level =
            (hop.input_rms > SILENCE).then(|| smoothed(&mut self.input_level, db(hop.input_rms)));
        if let Some(level) = level {
            self.update_floor(level);
        }
        let voice = level.is_some_and(|level| {
            level >= self.floor_db + GATE_MARGIN_DB && hop.lsnr >= hop.min_db_thresh
        });

        if voice {
            self.closed_hops = 0;
            let measured = rms(samples);
            if measured > SILENCE {
                let level = smoothed(&mut self.output_level, db(measured));
                let desired = (TARGET_DB - level)
                    .clamp(MIN_GAIN_DB, MAX_GAIN_DB)
                    .min(self.noise_ceiling_gain());
                let coefficient = if desired > self.gain_db { BOOST } else { CUT };
                self.gain_db += (desired - self.gain_db) * coefficient;
            }
        } else {
            self.closed_hops = self.closed_hops.saturating_add(1);
            if self.closed_hops > HOLD_HOPS {
                // Let go, towards unity from whichever side the gain is on, and
                // stop exactly there rather than stepping past it.
                let step = RELEASE_DB_PER_HOP.min(self.gain_db.abs());
                self.gain_db -= step * self.gain_db.signum();
            }
        }

        let gain = 10f32.powf(self.gain_db / 20.0);
        self.apply_gain(samples, gain);
        (self.gain_db, voice)
    }

    /// Forget the level state, keeping only what the limiter needs to stay
    /// continuous.
    fn forget(&mut self) {
        self.gain_db = 0.0;
        self.closed_hops = 0;
        self.input_level = None;
        self.output_level = None;
        self.floor.fill(f32::INFINITY);
        self.floor_next = 0;
        self.floor_db = f32::INFINITY;
    }

    /// The most gain the tracked floor permits, whatever the gate decided.
    ///
    /// Zero before any level has been seen: with no floor there is no way to
    /// tell what the boost would be lifting, and silence is the wrong time to
    /// find out.
    fn noise_ceiling_gain(&self) -> f32 {
        (NOISE_CEILING_DB - self.floor_db).max(0.0)
    }

    fn update_floor(&mut self, level: f32) {
        self.floor[self.floor_next] = level;
        self.floor_next = (self.floor_next + 1) % FLOOR_WINDOW_HOPS;
        self.floor_db = self.floor.iter().copied().fold(f32::INFINITY, f32::min);
    }

    /// Scale by `gain` and hold the peak under -1 dBFS.
    ///
    /// Causal four-times-oversampled peak detection. The detector has 2 dB of
    /// headroom for intersample overshoot and the final clamp enforces the
    /// -1 dBFS sample ceiling even for an abrupt full-scale transition.
    ///
    /// Non-finite samples are replaced with silence before anything is measured.
    /// An `inf` input makes `required == 0`, which collapses the limiter, whose
    /// `inf * 0` is then `NaN` — and the limiter would stay collapsed for the
    /// length of its release, spreading the damage. A capture device that emits
    /// one such sample should cost one sample, not the next fifth of a second.
    fn apply_gain(&mut self, samples: &mut [f32], gain: f32) {
        let ceiling = 10f32.powf(-1.0 / 20.0);
        let detector_ceiling = 10f32.powf(-3.0 / 20.0);
        let mut previous = self.previous;
        let mut limiter_gain = self.limiter_gain;
        for sample in samples {
            if !sample.is_finite() {
                *sample = 0.0;
            }
            let gained = *sample * gain;
            let mut true_peak = gained.abs();
            for index in 1..4 {
                let fraction = index as f32 / 4.0;
                let interpolated = previous * (1.0 - fraction) + gained * fraction;
                true_peak = true_peak.max(interpolated.abs());
            }
            let required = if true_peak > detector_ceiling {
                detector_ceiling / true_peak
            } else {
                1.0
            };
            if required < limiter_gain {
                limiter_gain = required;
            } else {
                limiter_gain = (limiter_gain + 0.00005).min(1.0);
            }
            *sample = (gained * limiter_gain).clamp(-ceiling, ceiling);
            previous = gained;
        }
        self.previous = previous;
        self.limiter_gain = limiter_gain;
    }
}

impl Default for Agc {
    fn default() -> Self {
        Self::new()
    }
}

fn smoothed(previous: &mut Option<f32>, level: f32) -> f32 {
    let value = previous.map_or(level, |value| value + (level - value) * LEVEL_SMOOTHING);
    *previous = Some(value);
    value
}

/// RMS of a block of samples.
///
/// An empty block has no level, so it reports zero rather than `0/0`.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Amplitude to dBFS.
pub fn db(amplitude: f32) -> f32 {
    20.0 * amplitude.log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dfn_dsp::HOP_SIZE;

    const MIN_THRESH: f32 = -10.0;

    /// Deterministic uniform noise, so a scenario is the same on every run.
    struct Noise(u32);

    impl Noise {
        fn hop(&mut self, rms: f32, out: &mut [f32]) {
            for sample in out.iter_mut() {
                self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let unit = (self.0 >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
                // Uniform in [-1, 1] has an RMS of 1/sqrt(3).
                *sample = unit * rms * 3f32.sqrt();
            }
        }
    }

    struct Rig {
        agc: Agc,
        noise: Noise,
        block: Vec<f32>,
    }

    impl Rig {
        fn new() -> Self {
            Self {
                agc: Agc::new(),
                noise: Noise(0x5eed),
                block: vec![0.0; HOP_SIZE],
            }
        }

        /// Run hops of noise at `level_db` RMS. `suppression_db` is how much the
        /// suppressor removes before the AGC sees the hop.
        fn run(
            &mut self,
            hops: usize,
            level_db: f32,
            lsnr: f32,
            suppression_db: f32,
        ) -> (f32, f32) {
            let level = 10f32.powf(level_db / 20.0);
            let mut last = (0.0, 0.0);
            for _ in 0..hops {
                self.noise.hop(level, &mut self.block);
                let input_rms = rms(&self.block);
                for sample in self.block.iter_mut() {
                    *sample *= 10f32.powf(-suppression_db / 20.0);
                }
                let (gain, _) = self.agc.process(
                    &mut self.block,
                    Hop {
                        lsnr,
                        input_rms,
                        min_db_thresh: MIN_THRESH,
                    },
                    true,
                );
                last = (gain, db(rms(&self.block)));
            }
            last
        }
    }

    /// The bug: a quiet room, and the model calling it voice often enough to
    /// open the old `lsnr >= 0` gate a third of the time.
    #[test]
    fn quiet_room_is_never_amplified() {
        let mut rig = Rig::new();
        let (gain, _) = rig.run(1000, -45.0, 4.0, 3.6);
        assert!(
            gain <= 0.5,
            "the gate opened on room noise and lifted it by {gain:.2} dB"
        );
    }

    /// The ceiling, which is the only thing standing when the gate is wrong.
    ///
    /// The gate opens here — this is real speech 12 dB over the room, exactly
    /// what it is meant to accept — and the target asks for more gain than the
    /// room allows. The room must not be lifted past `NOISE_CEILING_DB` even so.
    #[test]
    fn the_room_never_exceeds_the_noise_ceiling() {
        let mut rig = Rig::new();
        // Ten seconds of room at -45, which is what the floor is tracked from.
        rig.run(400, -45.0, -14.0, 6.0);
        // Then speech at -33: 12 dB up, so the gate opens, and 6 dB is
        // suppressed, so the target asks for 21 dB and clamps to 18.
        let (gain, _) = rig.run(500, -33.0, 12.0, 6.0);
        let allowed = NOISE_CEILING_DB - -45.0;
        assert!(
            gain <= allowed + 0.5,
            "gain reached {gain:+.1} dB, which puts a -45 dBFS room at {:.1} dBFS \
             (ceiling {NOISE_CEILING_DB:.0}, so at most {allowed:+.0} dB of gain)",
            -45.0 + gain
        );
    }

    /// Speech well below the target is lifted to it, measured on what the
    /// suppressor produced rather than on the microphone.
    #[test]
    fn speech_is_levelled_to_the_target() {
        let mut rig = Rig::new();
        rig.run(300, -50.0, -14.0, 0.0);
        let (gain, output_db) = rig.run(400, -30.0, 12.0, 4.0);
        assert!(
            (output_db - TARGET_DB).abs() < 2.0,
            "speech landed at {output_db:.1} dBFS, not {TARGET_DB:.0} (gain {gain:+.1} dB)"
        );
    }

    /// The release that was missing entirely: after speech stops, the gain must
    /// come back to unity instead of holding at the clamp for ever.
    #[test]
    fn gain_releases_to_unity_after_speech() {
        let mut rig = Rig::new();
        rig.run(300, -50.0, -14.0, 0.0);
        rig.run(400, -30.0, 12.0, 4.0);
        let (peak, _) = rig.run(1, -50.0, -14.0, 0.0);
        assert!(
            peak > 8.0,
            "the gain never rose, so there is nothing to release"
        );
        let (gain, output_db) = rig.run(700, -50.0, -14.0, 0.0);
        assert!(
            gain.abs() < 0.5,
            "gain held at {gain:+.1} dB seven seconds after speech stopped"
        );
        assert!(
            output_db < -48.0,
            "the room came out at {output_db:.1} dBFS, so the gain was still applied"
        );
    }

    /// A pause between words is inside the sentence: the gain must not move.
    #[test]
    fn gain_is_held_across_a_short_pause() {
        let mut rig = Rig::new();
        rig.run(300, -50.0, -14.0, 0.0);
        let (before, _) = rig.run(400, -30.0, 12.0, 4.0);
        let (after, _) = rig.run(20, -30.0, -20.0, 4.0);
        assert!(
            (after - before).abs() < 0.05,
            "a 200 ms pause moved the gain from {before:+.2} to {after:+.2} dB"
        );
    }

    /// Digital silence has no level, so it must neither open the gate nor drag
    /// the floor to minus infinity and open it by comparison.
    #[test]
    fn digital_silence_keeps_the_gate_shut() {
        let mut agc = Agc::new();
        let mut block = vec![0.0f32; HOP_SIZE];
        for _ in 0..500 {
            let (gain, voice) = agc.process(
                &mut block,
                Hop {
                    lsnr: 20.0,
                    input_rms: 0.0,
                    min_db_thresh: MIN_THRESH,
                },
                true,
            );
            assert!(!voice, "the gate opened on digital silence");
            assert_eq!(gain, 0.0);
        }
    }

    /// Switching the AGC off is unity gain and leaves the limiter state alone.
    #[test]
    fn disabled_is_unity_and_forgets() {
        let mut rig = Rig::new();
        rig.run(300, -50.0, -14.0, 0.0);
        rig.run(400, -30.0, 12.0, 4.0);
        assert!(rig.agc.gain_db > 8.0);

        rig.noise.hop(10f32.powf(-40.0 / 20.0), &mut rig.block);
        let before = rig.block.clone();
        let (gain, voice) = rig.agc.process(
            &mut rig.block,
            Hop {
                lsnr: 12.0,
                input_rms: 0.01,
                min_db_thresh: MIN_THRESH,
            },
            false,
        );
        assert_eq!((gain, voice), (0.0, false));
        for (after, before) in rig.block.iter().zip(&before) {
            assert!(
                (after - before).abs() < 1e-6,
                "disabled AGC changed the audio"
            );
        }
        assert_eq!(rig.agc.gain_db, 0.0);
    }
}
