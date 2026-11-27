//! The PipeWire bridge, in Rust.
//!
//! Three streams on one thread loop: a capture stream reading a physical
//! microphone, an output stream exported as the `dfmic-source` node (the
//! virtual microphone), and an optional monitor stream that plays the same
//! audio back so the user can hear themselves.
//!
//! Process callbacks run on PipeWire's *data loop*, a third thread that the loop
//! lock does not cover, and only touch the lock-free rings and atomics below.
//! Main-loop and registry callbacks run on the loop thread under the lock, which
//! is what makes the proxies and listeners safe to touch from both.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pipewire as pw;
use pw::properties::properties;
use pw::spa;

use dfn_dsp::HOP_SIZE;

use crate::proto::{Device, Levels, MonitorTarget, NODE_NAME};

const RATE: u32 = 48_000;

/// The graph quantum we ask for while filtering, in samples: 2.67 ms.
///
/// PipeWire sizes the whole graph from the smallest latency any client asks
/// for, so this is not a private buffer. Measured end to end on this machine
/// (median over 150 s runs, with the worker woken by a futex):
/// 1024 = 102 ms, 480 = 60 ms, 256 = 31 ms, 128 = 26 ms.
///
/// A smaller quantum is better on both axes here, not just latency: the
/// consumer asks for less audio per callback, so it needs less in reserve and
/// underruns drop too (0-2 per 150 s at 128, against 4 at 256). It does mean
/// every other application wakes more often, which is why this is a setting
/// (`graph-quantum`) and only requested while the filter is running.
const ACTIVE_QUANTUM: u32 = 128;

/// What the streams ask for when we are not filtering: PipeWire's own default,
/// which leaves the graph exactly as it would be without dfmic running.
const NEUTRAL_QUANTUM: u32 = 1024;

/// How long a depth must hold before another shrink is attempted. libDF uses a
/// callback-counted equivalent; a wall-clock window is easier to reason about
/// and immune to quantum changes.
const SHRINK_HYSTERESIS_MS: u64 = 5_000;

/// Wait before retrying a shrink that just failed, and the ceiling that wait
/// doubles up to. Reaching the ceiling means this machine has settled.
const SHRINK_BACKOFF_MIN_MS: u64 = 5_000;
const SHRINK_BACKOFF_MAX_MS: u64 = 120_000;

/// The smallest target this architecture can hold, in hops.
///
/// libDF's floor is one hop, because its audio callback blocks until the worker
/// delivers a block. Ours cannot: a PipeWire process callback must return
/// immediately or it stalls the graph for every client, so it emits silence
/// instead. One hop is therefore never enough — the consumer drains it faster
/// than the worker can refill at the moment a hop boundary is crossed — and
/// measurement agrees: one hop dropped thousands of buffers, one hop plus a
/// quantum dropped hundreds, two hops dropped none. Starting here rather than
/// at one hop also stops the startup from ratcheting the target upwards through
/// a series of avoidable underruns.
const MIN_HOPS: usize = 2;

/// The deepest the output queue may be asked to hold: one second of audio,
/// which is where libDF gives up entirely.
///
/// The rings are sized one hop beyond this, so a full queue can still take the
/// next hop the worker produces; the adaptive target never asks for more than
/// this, which is what keeps the "latency is bounded" rule true.
const MAX_QUEUE_SAMPLES: usize = RATE as usize;

#[derive(Default)]
pub struct RegistryData {
    sources: BTreeMap<String, Device>,
    sinks: BTreeMap<String, Device>,
    /// Raw registry contents, keyed by global id so hot-plug removal can find
    /// them again. `sources` and `sinks` are rebuilt from these.
    nodes: BTreeMap<u32, NodeRecord>,
    devices: BTreeMap<u32, DeviceRecord>,
    /// Global id of our exported node, so links to it can be recognised.
    own_node: Option<u32>,
    /// Every link in the graph, by global id, as (output node, input node).
    ///
    /// The stream state cannot answer "is anything listening": the exported node
    /// suspends and resumes with its consumers, and a consumer that is linked but
    /// silent does not make it `Streaming`, so the state is neither necessary nor
    /// sufficient. The links are the authoritative answer, and PipeWire publishes
    /// them as globals.
    links: BTreeMap<u32, (u32, u32)>,
    pub default_source: Option<String>,
    pub default_sink: Option<String>,
}

/// A node global as the registry reported it, before the device it belongs to
/// has necessarily arrived.
struct NodeRecord {
    sink: bool,
    name: String,
    description: String,
    bus: Option<String>,
    form_factor: Option<String>,
    virtual_node: bool,
    device_id: Option<u32>,
}

#[derive(Default)]
struct DeviceRecord {
    bus: Option<String>,
    form_factor: Option<String>,
}

impl RegistryData {
    /// How many links carry our node's output into a consumer.
    fn consumer_count(&self) -> usize {
        let Some(own) = self.own_node else {
            return 0;
        };
        self.links
            .values()
            .filter(|(output, _)| *output == own)
            .count()
    }

    pub fn devices(&self) -> Vec<Device> {
        self.sources.values().cloned().collect()
    }

    pub fn source(&self, name: &str) -> Option<Device> {
        self.sources.get(name).cloned()
    }

    pub fn default_source_name(&self) -> Option<String> {
        self.default_source.clone().or_else(|| {
            self.sources
                .values()
                .find(|device| device.is_system_default)
                .map(|device| device.name.clone())
        })
    }

    pub fn monitor_target(&self) -> MonitorTarget {
        let Some(name) = self.default_sink.clone() else {
            return MonitorTarget {
                name: String::new(),
                description: String::new(),
                kind: "unknown".into(),
            };
        };
        let Some(sink) = self.sinks.get(&name) else {
            return MonitorTarget {
                name,
                description: String::new(),
                kind: "unknown".into(),
            };
        };
        let haystack = format!(
            "{} {} {} {}",
            name, sink.description, sink.bus, sink.form_factor
        )
        .to_lowercase();
        let kind = if haystack.contains("headphone")
            || haystack.contains("headset")
            || haystack.contains("earbud")
            || (haystack.contains("bluetooth") && !haystack.contains("speaker"))
        {
            "headphones"
        } else if haystack.contains("speaker") || sink.form_factor.eq_ignore_ascii_case("speaker") {
            "speakers"
        } else {
            "unknown"
        };
        MonitorTarget {
            name,
            description: sink.description.clone(),
            kind: kind.into(),
        }
    }

    /// Rebuild the published device list from the raw registry contents.
    ///
    /// A node's bus and form factor come from its own properties when PipeWire
    /// sets them, and from its parent device otherwise; the two globals arrive
    /// in either order, which is why the records are kept apart.
    fn publish(&mut self) {
        let defaults = [
            (false, self.default_source.clone()),
            (true, self.default_sink.clone()),
        ];
        let mut published = Vec::with_capacity(self.nodes.len());
        for node in self.nodes.values() {
            let device = node.device_id.and_then(|id| self.devices.get(&id));
            let bus = node
                .bus
                .clone()
                .or_else(|| device.and_then(|device| device.bus.clone()))
                .unwrap_or_default();
            let form_factor = node
                .form_factor
                .clone()
                .or_else(|| device.and_then(|device| device.form_factor.clone()))
                .unwrap_or_default();
            let is_system_default = defaults
                .iter()
                .any(|(sink, name)| *sink == node.sink && name.as_deref() == Some(&node.name));
            published.push((
                node.sink,
                node.name.clone(),
                Device {
                    name: node.name.clone(),
                    description: node.description.clone(),
                    bus,
                    form_factor,
                    virtual_node: node.virtual_node,
                    is_system_default,
                },
            ));
        }
        self.sources.clear();
        self.sinks.clear();
        for (sink, name, device) in published {
            if sink {
                self.sinks.insert(name, device);
            } else {
                self.sources.insert(name, device);
            }
        }
    }
}

pub struct Shared {
    pub input: ring::Ring,
    pub output: ring::Ring,
    pub monitor_output: ring::Ring,
    pub failed: AtomicBool,
    pub error: Mutex<Option<String>>,
    pub registry: Mutex<RegistryData>,
    pub captured: AtomicU64,
    pub played: AtomicU64,
    pub overruns: AtomicU64,
    pub underruns: AtomicU64,
    /// Nudges the daemon's event loop when something changed out here.
    ///
    /// The loop blocks on its request channel instead of polling, so the
    /// PipeWire callbacks — which run on the loop thread and are the main source
    /// of "something changed" — have to ring this bell themselves.
    wake: Mutex<Option<std::sync::mpsc::Sender<crate::control::Request>>>,
    pub consumers: AtomicBool,
    pub monitor_active: AtomicBool,
    pub filtering: AtomicBool,
    pub min_buffer: AtomicUsize,
    /// The output queue's target depth in samples, adapted at runtime.
    ///
    /// libDF calls this `proc_delay`. It starts at [`MIN_HOPS`] hops, not one:
    /// one hop is never enough for a callback that must not block (see the
    /// constant's note), and it moves from there as the machine demands.
    proc_delay: AtomicUsize,
    /// Earliest time (ms since start) at which another shrink may be tried.
    shrink_after_ms: AtomicU64,
    /// Current wait between shrink attempts, doubled on every failure.
    shrink_backoff_ms: AtomicU64,
    priming: AtomicBool,
    monitor_priming: AtomicBool,
    /// Samples of silence the worker should push into the output ring, because
    /// the adaptive target just got deeper.
    ///
    /// A counter rather than a direct write: only the worker may produce into
    /// that ring, and only the playback callback may consume it. A third thread
    /// adding audio would race the producer's `write` index.
    pad_output: AtomicUsize,
    /// Bumped by anyone who wants a playback ring dropped and re-primed.
    ///
    /// A counter rather than a direct `clear`, because only the playback callback
    /// may move a ring's read index: see [`ring`].
    output_reprime: AtomicU64,
    monitor_reprime: AtomicU64,
    /// What the playback callback has already acted on. Written only by it.
    output_reprime_seen: AtomicU64,
    monitor_reprime_seen: AtomicU64,
    reset_generation: AtomicU64,
    output_levels: MetricRing,
    timing: TimingRing,
    pub inferences: AtomicU64,
    /// Bumped whenever the input ring reaches a whole hop. The worker sleeps on
    /// this and the capture callback wakes it, so a hop is processed the moment
    /// it is complete instead of on the next poll tick.
    hop_ready: AtomicU32,
    /// Why the capture stream last failed, so the daemon can retry it rather
    /// than exit.
    capture_failure: Mutex<Option<String>>,
    /// Why the self-monitor stream last failed, for the daemon to report.
    monitor_failure: Mutex<Option<String>>,
    voice: AtomicBool,
    agc_gain: AtomicU32,
    started: Instant,
}

impl Shared {
    /// Rings sized for the deepest queue the adaptive rule may ask for, plus one
    /// hop of slack so a full queue can still take the next hop.
    pub fn new() -> Arc<Self> {
        let started = Instant::now();
        let capacity = MAX_QUEUE_SAMPLES + HOP_SIZE;
        Arc::new(Self {
            input: ring::Ring::new(capacity),
            output: ring::Ring::new(capacity),
            monitor_output: ring::Ring::new(capacity),
            failed: AtomicBool::new(false),
            error: Mutex::new(None),
            registry: Mutex::new(RegistryData::default()),
            captured: AtomicU64::new(0),
            played: AtomicU64::new(0),
            overruns: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
            wake: Mutex::new(None),
            consumers: AtomicBool::new(false),
            monitor_active: AtomicBool::new(false),
            filtering: AtomicBool::new(false),
            min_buffer: AtomicUsize::new(0),
            proc_delay: AtomicUsize::new(HOP_SIZE * MIN_HOPS),
            shrink_after_ms: AtomicU64::new(0),
            shrink_backoff_ms: AtomicU64::new(0),
            priming: AtomicBool::new(true),
            monitor_priming: AtomicBool::new(true),
            output_reprime: AtomicU64::new(0),
            monitor_reprime: AtomicU64::new(0),
            output_reprime_seen: AtomicU64::new(0),
            monitor_reprime_seen: AtomicU64::new(0),
            reset_generation: AtomicU64::new(0),
            pad_output: AtomicUsize::new(0),
            output_levels: MetricRing::new(),
            timing: TimingRing::new(started),
            inferences: AtomicU64::new(0),
            hop_ready: AtomicU32::new(0),
            monitor_failure: Mutex::new(None),
            capture_failure: Mutex::new(None),
            voice: AtomicBool::new(false),
            agc_gain: AtomicU32::new(0.0f32.to_bits()),
            started,
        })
    }

    fn set_error(&self, message: String) {
        if let Ok(mut slot) = self.error.lock() {
            if slot.is_none() {
                *slot = Some(message);
            }
        }
        self.failed.store(true, Ordering::SeqCst);
        // Every asynchronous failure has to ring the loop: it blocks on its
        // request channel, so without this the error is not seen until an
        // unrelated wake, up to the 30 s safety timeout later.
        self.wake_loop();
    }

    pub fn error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|mut error| error.take())
    }

    /// Record that the self-monitor stream failed, for the daemon to report.
    ///
    /// This is not fatal and it is not retried: a sink that will not open stays
    /// that way until the user picks a different output, and recreating the
    /// stream every tick would churn it. The daemon turns the flag into a notice
    /// and switches monitoring off.
    fn note_monitor_failure(&self, message: String) {
        if let Ok(mut slot) = self.monitor_failure.lock() {
            *slot = Some(message);
        }
        self.monitor_active.store(false, Ordering::SeqCst);
        // The loop is asleep on its request channel; a failure nobody is told
        // about is not reported until the next unrelated wake, which can be the
        // 30 s safety timeout.
        self.wake_loop();
    }

    /// Note that the capture stream lost its device, without taking the daemon
    /// down.
    ///
    /// A microphone being unplugged is an `Error` state in PipeWire, not a
    /// transition to `Paused`, so the registry's disappearance bookkeeping never
    /// gets a chance to run on its own. Recording it here lets the daemon show
    /// `NoDevice` and reconnect when the device returns, which is the difference
    /// between a Bluetooth headset dropping out and the daemon exiting.
    fn note_capture_loss(&self, message: String) {
        if let Ok(mut slot) = self.capture_failure.lock() {
            // Only the first reason is kept until it is taken: a device that is
            // being replugged produces a burst of them and they all say the same
            // thing.
            if slot.is_none() {
                *slot = Some(message);
            }
        }
        self.wake_loop();
    }

    /// Take the pending capture loss, if any.
    pub fn take_capture_loss(&self) -> Option<String> {
        self.capture_failure
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
    }

    /// Take the pending self-monitor failure, if any.
    pub fn take_monitor_failure(&self) -> Option<String> {
        self.monitor_failure
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
    }

    pub fn set_monitor_active(&self, active: bool) {
        let was = self.monitor_active.swap(active, Ordering::SeqCst);
        if was != active {
            self.request_monitor_reprime();
            // Same reason as `set_consumers`: monitoring is a consumer too, and
            // the worker may be in a long sleep.
            self.wake_worker();
        }
    }

    /// Ask the playback callback to drop what is queued and start from silence,
    /// on the monitor ring only.
    pub fn request_monitor_reprime(&self) {
        self.monitor_reprime.fetch_add(1, Ordering::AcqRel);
    }

    /// Ask the worker to reset both ends. The worker is the input consumer and
    /// the output producer, so this preserves every ring's SPSC ownership.
    pub fn reset_audio(&self) {
        self.reset_generation.fetch_add(1, Ordering::AcqRel);
        // A new stream means a new depth to learn: keeping the previous session's
        // target and its non-decaying backoff made every reset start deeper and
        // slower to react than the last.
        self.reset_delay();
        self.request_reprime();
    }

    /// Take the sample count the worker should pad the output ring with, if any.
    pub fn take_pad(&self) -> usize {
        self.pad_output.swap(0, Ordering::AcqRel)
    }

    /// Ask the playback callback to drop what is queued and start from silence.
    ///
    /// A request, not a `clear`: the callback owns the read side of the output
    /// rings, and moving it from another thread would race with the pop it is in
    /// the middle of.
    pub fn request_reprime(&self) {
        self.output_reprime.fetch_add(1, Ordering::AcqRel);
        self.monitor_reprime.fetch_add(1, Ordering::AcqRel);
    }

    pub fn reset_generation(&self) -> u64 {
        self.reset_generation.load(Ordering::Acquire)
    }

    pub fn set_agc(&self, gain_db: f32, voice: bool) {
        self.agc_gain.store(gain_db.to_bits(), Ordering::Relaxed);
        self.voice.store(voice, Ordering::Release);
    }

    pub fn record_inference(&self, seconds: f64) {
        self.inferences.fetch_add(1, Ordering::Relaxed);
        self.timing
            .record(self.started.elapsed().as_secs_f64(), seconds * 1000.0);
    }

    pub fn inference_ms(&self) -> f64 {
        self.timing
            .mean_last_second(self.started.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    pub fn levels(&self) -> Levels {
        let now = self.started.elapsed().as_secs_f64();
        let output = self.output_levels.aggregate(now);
        Levels {
            // The daemon no longer meters its input: the window captures the
            // microphone itself when it wants a level, so nothing here is paid
            // for by a daemon that nobody is watching.
            input_active: false,
            input_peak: 0.0,
            input_rms: 0.0,
            input_clip: false,
            output_active: output.active,
            output_peak: output.peak,
            output_rms: output.rms,
            output_clip: output.clip,
            agc_gain_db: f32::from_bits(self.agc_gain.load(Ordering::Relaxed)) as f64,
            voice: self.voice.load(Ordering::Acquire),
        }
    }

    pub fn is_default(&self) -> bool {
        self.registry
            .lock()
            .is_ok_and(|registry| registry.default_source.as_deref() == Some(NODE_NAME))
    }

    pub fn listening(&self) -> bool {
        self.consumers.load(Ordering::Acquire) || self.monitor_active.load(Ordering::Acquire)
    }

    /// The smallest target the adaptive rule may shrink to: one hop per
    /// [`MIN_HOPS`], plus whatever the user asked to keep in reserve.
    fn floor_delay(&self) -> usize {
        HOP_SIZE * (MIN_HOPS + self.min_buffer.load(Ordering::Relaxed))
    }

    /// How deep the output queue should be, in samples.
    pub fn proc_delay(&self) -> usize {
        self.proc_delay.load(Ordering::Relaxed)
    }

    /// Grow the target by one hop after an underrun.
    ///
    /// libDF adds one frame of latency whenever its realtime factor exceeds 1.
    /// The difference here is the wait before trying to shrink again: because a
    /// PipeWire callback cannot block, a failed shrink shows up as an actual
    /// dropout rather than as a stalled callback, so repeating the attempt at a
    /// fixed interval would cost a click every time. Instead each failure
    /// doubles the delay before the next attempt, which converges on the depth
    /// this machine can hold instead of oscillating around it.
    ///
    /// The target is *filled* as well as raised. Raising it alone changed the
    /// trim length without putting any audio behind it, so the next callback
    /// trimmed to a depth the ring did not hold, underran, and grew again — an
    /// underrun every callback rather than one per failed shrink.
    fn grow_delay(&self) {
        let current = self.proc_delay.load(Ordering::Relaxed);
        // Up to one second of audio, which is where libDF gives up entirely, and
        // never past `MAX_QUEUE_SAMPLES`: the rings are sized one hop beyond that
        // so the producer always has room to fill the target.
        debug_assert!(self.output.capacity() >= MAX_QUEUE_SAMPLES + HOP_SIZE);
        if current + HOP_SIZE <= MAX_QUEUE_SAMPLES {
            let next = current + HOP_SIZE;
            self.proc_delay.store(next, Ordering::Relaxed);
            // Only the worker may write this ring, so it is asked rather than
            // filled here, and `priming` is left alone: asking for the pad to be
            // silence is not the same as asking for the queue to be dropped.
            self.pad_output
                .fetch_add(next.saturating_sub(self.output.len()), Ordering::AcqRel);
        } else {
            // At the ceiling already: there is no deeper to go.
            return;
        }
        let backoff = self.shrink_backoff_ms.load(Ordering::Relaxed);
        let next = if backoff == 0 {
            SHRINK_BACKOFF_MIN_MS
        } else {
            (backoff * 2).min(SHRINK_BACKOFF_MAX_MS)
        };
        self.shrink_backoff_ms.store(next, Ordering::Relaxed);
        self.shrink_after_ms
            .store(millis_timestamp(self.started) + next, Ordering::Relaxed);
    }

    /// Forget the adaptive depth and its backoff.
    ///
    /// Called when filtering stops and when the stream is reset: the depth that
    /// this machine held is a property of a session that has ended, and the
    /// backoff only ever doubled, so without this every enable/disable cycle
    /// would leave the controller slower to react than the last one.
    pub fn reset_delay(&self) {
        self.proc_delay
            .store(HOP_SIZE * MIN_HOPS, Ordering::Relaxed);
        self.shrink_backoff_ms.store(0, Ordering::Relaxed);
        self.shrink_after_ms.store(0, Ordering::Relaxed);
    }

    /// Shrink the target by one hop once the current depth has held for long
    /// enough, floored at [`Self::floor_delay`].
    fn maybe_shrink_delay(&self) {
        let current = self.proc_delay.load(Ordering::Relaxed);
        if current <= self.floor_delay() {
            return;
        }
        let now = millis_timestamp(self.started);
        if now < self.shrink_after_ms.load(Ordering::Relaxed) {
            return;
        }
        self.proc_delay.store(current - HOP_SIZE, Ordering::Relaxed);
        // Give the new depth the base interval to prove itself; a failure here
        // will double the backoff again.
        self.shrink_after_ms
            .store(now + SHRINK_HYSTERESIS_MS, Ordering::Relaxed);
    }

    /// Block until a whole hop is available in the input ring, or the timeout
    /// expires.
    ///
    /// A futex rather than a condvar: the capture callback runs on PipeWire's
    /// realtime thread and must not take a lock, and `FUTEX_WAKE` needs none.
    /// The timeout bounds how long shutdown has to wait for the worker.
    /// Block until a whole hop is available in the input ring, or `timeout`
    /// expires.
    ///
    /// A futex rather than a condvar: the capture callback runs on PipeWire's
    /// realtime thread and must not take a lock, and `FUTEX_WAKE` needs none.
    /// The timeout bounds how long shutdown has to wait for the worker, and while
    /// nothing is listening it is the only thing that wakes it — a long timeout
    /// there is what makes an idle daemon cost nothing.
    pub fn wait_for_hop(&self, timeout: Duration) {
        if self.input.len() >= HOP_SIZE {
            return;
        }
        let expected = self.hop_ready.load(Ordering::Acquire);
        // Re-check after reading the counter: a push between the two would
        // otherwise leave us asleep with a hop already waiting.
        if self.input.len() >= HOP_SIZE {
            return;
        }
        let timeout = libc::timespec {
            tv_sec: timeout.as_secs() as libc::time_t,
            tv_nsec: timeout.subsec_nanos() as libc::c_long,
        };
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                self.hop_ready.as_ptr(),
                libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                expected,
                &timeout,
            );
        }
    }

    /// Wake the worker. Safe to call from a realtime callback: one atomic and
    /// one syscall, no allocation and no lock.
    pub fn wake_worker(&self) {
        self.hop_ready.fetch_add(1, Ordering::Release);
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                self.hop_ready.as_ptr(),
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                1,
            );
        }
    }

    /// Apply a registry edit and republish the device list.
    fn with_registry(&self, edit: impl FnOnce(&mut RegistryData)) {
        if let Ok(mut registry) = self.registry.lock() {
            edit(&mut registry);
            registry.publish();
            let consumers = registry.consumer_count();
            drop(registry);
            self.set_consumers(consumers);
        }
        // After the lock is released: the loop takes the same lock, and waking it
        // while holding it would invert the order.
        self.wake_loop();
    }

    /// Let the daemon's event loop know that something changed.
    pub fn wake_loop(&self) {
        if let Ok(slot) = self.wake.lock() {
            if let Some(sender) = slot.as_ref() {
                crate::control::wake(sender);
            }
        }
    }

    /// Give the loop its wake sender. Called once, right after the channel exists.
    pub fn set_wake_sender(&self, sender: std::sync::mpsc::Sender<crate::control::Request>) {
        if let Ok(mut slot) = self.wake.lock() {
            *slot = Some(sender);
        }
    }

    /// Record how many consumers our node has, resetting the queue on arrival.
    fn set_consumers(&self, count: usize) {
        let was = self.consumers.swap(count > 0, Ordering::SeqCst);
        if count > 0 && !was {
            // A consumer just attached: whatever is queued is from before it, so
            // start from silence rather than from a stale tail. Only a request is
            // made here — dropping the queue is the playback callback's job,
            // because it owns the read side of the ring (see `ring`).
            self.request_reprime();
        }
        if was != (count > 0) {
            // The worker sleeps for a long time while nothing is listening, so a
            // consumer arriving has to wake it: the capture callback only starts
            // feeding the ring once it is listening.
            self.wake_worker();
        }
    }
}

fn millis_timestamp(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u64::MAX as u128) as u64
}

/// The buffer handed to a process callback, viewed as `f32` samples.
///
/// PipeWire mapped this memory for us (`MAP_BUFFERS`) and the negotiated format
/// is 32-bit float, so the cast is sound. The chunk's offset and size are in
/// bytes, and either may be nonsense if a client misbehaves: both are clamped to
/// the mapped region *before* the pointer is formed, because `ptr.add()` past the
/// allocation is undefined even when the slice built from it would be empty.
unsafe fn samples(data: &mut [u8], offset: u32, size: u32) -> &mut [f32] {
    let offset = offset as usize;
    if offset >= data.len() {
        return &mut [];
    }
    let frames = (size as usize / 4).min((data.len() - offset) / 4);
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().add(offset).cast::<f32>(), frames) }
}

fn capture_process(shared: &Shared, stream: &pw::stream::Stream) {
    let Some(mut buffer) = stream.dequeue_buffer() else {
        return;
    };
    let datas = buffer.datas_mut();
    let Some(data) = datas.first_mut() else {
        return;
    };
    let chunk = data.chunk();
    let (offset, size) = (chunk.offset(), chunk.size());
    let Some(bytes) = data.data() else {
        return;
    };
    let samples = unsafe { samples(bytes, offset, size) };
    if samples.is_empty() {
        return;
    }
    shared
        .captured
        .fetch_add(samples.len() as u64, Ordering::Relaxed);
    // Nothing is recording from us: drop the samples here rather than pushing
    // them for the worker to throw away. The daemon measures no levels — the
    // meters are the window's own capture streams — so with nobody listening
    // there is genuinely nothing to do with this audio.
    if !shared.listening() {
        return;
    }
    if !shared.input.push(samples) {
        shared
            .overruns
            .fetch_add(samples.len().div_ceil(HOP_SIZE) as u64, Ordering::Relaxed);
    } else if shared.input.len() >= HOP_SIZE {
        shared.wake_worker();
    }
}

fn playback_process(shared: &Shared, stream: &pw::stream::Stream, monitor: bool) {
    let Some(mut buffer) = stream.dequeue_buffer() else {
        return;
    };
    let requested = buffer.requested() as usize;
    let datas = buffer.datas_mut();
    let Some(data) = datas.first_mut() else {
        return;
    };
    let stride = std::mem::size_of::<f32>();
    let capacity = data.data().map_or(0, |bytes| bytes.len()) / stride;
    if capacity == 0 {
        return;
    }
    let frames = if requested == 0 {
        capacity
    } else {
        requested.min(capacity)
    };

    let (ring, priming, reprime, reprime_seen) = if monitor {
        (
            &shared.monitor_output,
            &shared.monitor_priming,
            &shared.monitor_reprime,
            &shared.monitor_reprime_seen,
        )
    } else {
        (
            &shared.output,
            &shared.priming,
            &shared.output_reprime,
            &shared.output_reprime_seen,
        )
    };
    // Somebody asked for this queue to be dropped and refilled. They could not do
    // it themselves — only this callback may move the read index — so it happens
    // here, before anything is popped, and it happens once per request.
    let requested = reprime.load(Ordering::Acquire);
    if reprime_seen.load(Ordering::Relaxed) != requested {
        ring.discard(ring.len());
        reprime_seen.store(requested, Ordering::Relaxed);
        priming.store(true, Ordering::Release);
    }
    let filtering = shared.filtering.load(Ordering::Acquire);
    if filtering && !priming.load(Ordering::Acquire) {
        // Hold the queue at its adaptive target. This is libDF's rule: it keeps
        // `proc_delay` worth of audio queued and drops anything above that, so
        // latency is bounded rather than drifting upwards.
        let target = shared.proc_delay();
        if ring.len() > target {
            ring.discard(ring.len() - target);
        }
    }

    let mut filled = 0;
    if let Some(bytes) = data.data() {
        let samples = unsafe { samples(bytes, 0, (frames * stride) as u32) };
        filled = samples.len();
        let mut waiting = false;
        if priming.load(Ordering::Acquire) {
            if ring.len() < HOP_SIZE {
                // Not enough audio buffered yet: emit silence rather than a gap.
                samples.fill(0.0);
                waiting = true;
            } else {
                priming.store(false, Ordering::Release);
            }
        }
        if !waiting {
            let got = ring.pop(samples);
            if !monitor {
                shared.played.fetch_add(got as u64, Ordering::Relaxed);
            }
            if filtering {
                if got < filled {
                    // libDF's response to an underrun is to add one frame of
                    // latency and try again, so the same depth never starves
                    // twice. The queue refills from the worker; growing the
                    // target is what stops the trim from undoing it.
                    shared.underruns.fetch_add(1, Ordering::Relaxed);
                    shared.grow_delay();
                } else {
                    shared.maybe_shrink_delay();
                }
            }
            if !monitor || !shared.consumers.load(Ordering::Acquire) {
                shared.output_levels.record(shared.started, samples);
            }
        }
    }

    // Tell PipeWire how much of the buffer we produced. Without this the
    // consumer is handed a zero-length chunk and the graph spins.
    let chunk = data.chunk_mut();
    *chunk.offset_mut() = 0;
    *chunk.stride_mut() = stride as i32;
    *chunk.size_mut() = (filled * stride) as u32;
}

/// The exported stream came up.
///
/// This fires every time a consumer attaches, because the node suspends itself
/// when the last one leaves. The queue is dropped and re-primed from silence so a
/// new listener does not start with the tail of audio buffered for the previous
/// one. Whether anything is *listening* is still answered by the link globals,
/// not by this: an attach and a link are not the same event.
fn on_source_started(shared: &Shared) {
    shared.request_reprime();
}

/// Extract the `name` field from PipeWire's `Spa:String:JSON` metadata value,
/// which looks like `{"name":"alsa_input.…"}`.
fn parse_default_name(value: &str) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_str(value).ok()?;
    parsed.get("name")?.as_str().map(str::to_owned)
}

#[derive(Clone, Copy)]
enum StreamRole {
    Source,
    Capture,
    Monitor,
}

/// `Shared` is created as an `Arc` because the NPU worker owns it too; the
/// bridge wraps it again in an `Rc` because pipewire-rs' thread-loop API is
/// `Rc`-based. Both point at the same instance.
type SharedRef = Rc<Arc<Shared>>;

/// One stream and the listener that keeps its callbacks registered.
///
/// The listener must outlive the stream, so it is declared first and therefore
/// dropped first.
struct StreamSlot {
    /// Owned for its `Drop`: dropping the listener removes the callback hooks.
    #[allow(dead_code)]
    listener: pw::stream::StreamListener<SharedRef>,
    stream: pw::stream::StreamRc,
}

#[derive(Default)]
struct Streams {
    playback: Option<StreamSlot>,
    capture: Option<StreamSlot>,
    monitor: Option<StreamSlot>,
}

/// The proxy and listener for the `default` metadata object.
///
/// Kept in a `RefCell` because the registry callback that binds it is an `Fn`.
/// It is only ever borrowed while the loop lock is held, so the loop thread
/// cannot be inside a callback at the same time.
type MetadataSlot = RefCell<Option<(pw::metadata::Metadata, pw::metadata::MetadataListener)>>;

pub struct Bridge {
    shared: SharedRef,
    streams: RefCell<Streams>,
    monitor_target: RefCell<Option<String>>,
    metadata: Rc<MetadataSlot>,
    /// Whether the capture stream is currently started.
    capture_active: AtomicBool,
    // The listeners are owned for their `Drop` (which removes the hook), and are
    // declared before the objects they watch so they are dropped first.
    #[allow(dead_code)]
    core_listener: pw::core::Listener,
    #[allow(dead_code)]
    registry_listener: pw::registry::Listener,
    #[allow(dead_code)]
    registry: pw::registry::RegistryRc,
    #[allow(dead_code)]
    core: pw::core::CoreRc,
    #[allow(dead_code)]
    context: pw::context::ContextRc,
    thread_loop: pw::thread_loop::ThreadLoopRc,
}

impl Bridge {
    pub fn start(
        node_name: &str,
        description: &str,
        target: Option<&str>,
        shared: Arc<Shared>,
    ) -> Result<Self, String> {
        // Callbacks run on the loop thread while the bridge lives on the main
        // thread; `Rc` is what pipewire-rs' thread-loop API is built around.
        let shared = Rc::new(shared);

        let thread_loop =
            unsafe { pw::thread_loop::ThreadLoopRc::new(Some("dfmic-pipewire"), None) }
                .map_err(|error| format!("could not create the PipeWire event loop: {error}"))?;
        // `libpipewire-module-rt` is refused for this context: it disables
        // `RLIMIT_RTTIME` for the whole process while it probes for realtime
        // privileges, and on an unprivileged process the attempt to set the
        // hard limit back to its default (`-1`, i.e. infinity) fails with
        // `EPERM`, leaving 0/0. A realtime thread with a zero budget is
        // SIGKILLed by the kernel on its first slice of CPU, which is what used
        // to take the whole daemon down the moment the worker was promoted.
        //
        // Scheduling is ours to arrange anyway: see `rtkit`, which asks for
        // exactly the one thread that needs it and leaves the limit alone.
        // `module.rt` is the module's own condition key, so this skips it without
        // touching the user's PipeWire configuration.
        let mut properties = pw::properties::PropertiesBox::new();
        properties.insert("module.rt", "false");
        let context = pw::context::ContextRc::new(&thread_loop, Some(properties))
            .map_err(|error| format!("could not create the PipeWire context: {error}"))?;
        thread_loop.start();

        let bridge = {
            let lock = thread_loop.lock();
            let core = context
                .connect_rc(None)
                .map_err(|error| format!("could not connect to PipeWire: {error}"))?;
            let core_listener = {
                let shared = Rc::clone(&shared);
                core.add_listener_local()
                    .error(move |_, _, result, message| {
                        // The core reports `ENOENT` when the *target of one of our
                        // streams* disappears, not only when the connection itself
                        // dies — unplugging a microphone produces exactly this. That
                        // is a device loss and the daemon carries on; anything else
                        // (a real protocol or connection error) is fatal, because
                        // nothing can work without the core.
                        if result == -2 {
                            shared.note_capture_loss(format!(
                                "the graph reported {message} ({result})"
                            ));
                        } else {
                            shared.set_error(format!("{message} ({result})"));
                        }
                    })
                    .register()
            };
            let registry = core
                .get_registry_rc()
                .map_err(|error| format!("could not read the PipeWire registry: {error}"))?;
            let metadata: Rc<MetadataSlot> = Rc::new(RefCell::new(None));
            let registry_listener = registry_listener(&registry, Rc::clone(&shared), &metadata);

            // Release the loop before moving it into the bridge.
            drop(lock);
            Self {
                shared,
                streams: RefCell::new(Streams::default()),
                monitor_target: RefCell::new(None),
                metadata,
                capture_active: AtomicBool::new(true),
                core_listener,
                registry_listener,
                registry,
                core,
                context,
                thread_loop,
            }
        };

        {
            let _lock = bridge.thread_loop.lock();
            bridge.start_source(node_name, description)?;
            bridge.start_capture(target)?;
        }
        Ok(bridge)
    }

    fn new_stream(
        &self,
        name: &str,
        direction: spa::utils::Direction,
        props: pw::properties::PropertiesBox,
        role: StreamRole,
    ) -> Result<StreamSlot, String> {
        let stream = pw::stream::StreamRc::new(self.core.clone(), name, props)
            .map_err(|error| format!("could not create the {name} stream: {error}"))?;

        let shared = Rc::clone(&self.shared);
        let listener = stream
            .add_local_listener_with_user_data(shared)
            .state_changed(move |_, shared, old, new| state_changed(shared, role, old, new))
            .process(move |stream, shared| match role {
                StreamRole::Capture => capture_process(shared, stream),
                StreamRole::Source => playback_process(shared, stream, false),
                StreamRole::Monitor => playback_process(shared, stream, true),
            })
            .register()
            .map_err(|error| format!("could not listen to the {name} stream: {error}"))?;

        let mut info = spa::param::audio::AudioInfoRaw::new();
        info.set_format(spa::param::audio::AudioFormat::F32LE);
        info.set_rate(RATE);
        info.set_channels(1);
        let values = spa::pod::serialize::PodSerializer::serialize(
            std::io::Cursor::new(Vec::new()),
            &spa::pod::Value::Object(spa::pod::Object {
                type_: spa::sys::SPA_TYPE_OBJECT_Format,
                id: spa::sys::SPA_PARAM_EnumFormat,
                properties: info.into(),
            }),
        )
        .map_err(|error| format!("could not build the {name} format: {error:?}"))?
        .0
        .into_inner();
        let mut params = [spa::pod::Pod::from_bytes(&values)
            .ok_or_else(|| format!("could not build the {name} format pod"))?];

        stream
            .connect(
                direction,
                None,
                pw::stream::StreamFlags::AUTOCONNECT
                    | pw::stream::StreamFlags::MAP_BUFFERS
                    | pw::stream::StreamFlags::RT_PROCESS,
                &mut params,
            )
            .map_err(|error| format!("could not connect the {name} stream: {error}"))?;

        Ok(StreamSlot { listener, stream })
    }

    fn start_source(&self, node_name: &str, description: &str) -> Result<(), String> {
        let mut props = properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CLASS => "Audio/Source",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Communication",
            *pw::keys::APP_NAME => "dfmic",
            *pw::keys::NODE_NAME => node_name,
            *pw::keys::NODE_DESCRIPTION => description,
            *pw::keys::NODE_NICK => description,
            *pw::keys::NODE_VIRTUAL => "true",
            // High enough to win default-source elections, so WirePlumber does
            // not quietly hand the default back to the physical microphone.
            *pw::keys::PRIORITY_SESSION => "1000",
        };
        // The node is exported, not linked: applications connect to it.
        props.insert("node.autoconnect", "0");
        // Deliberately *not* `node.passive`. A passive node keeps the graph
        // running with no consumer, which burns the NPU and the capture stream
        // for audio nobody receives. Whether anything is listening is answered by
        // the link globals the registry tracks (see `RegistryData::links`), not by
        // the stream state: a linked-but-silent consumer never makes the node
        // `Streaming`, and this node suspends and resumes as its consumers come
        // and go.
        set_latency(&mut props, NEUTRAL_QUANTUM);
        let slot = self.new_stream(
            node_name,
            spa::utils::Direction::Output,
            props,
            StreamRole::Source,
        )?;
        self.streams.borrow_mut().playback = Some(slot);
        Ok(())
    }

    fn start_capture(&self, target: Option<&str>) -> Result<(), String> {
        let mut props = properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Communication",
            *pw::keys::APP_NAME => "dfmic",
            *pw::keys::NODE_NAME => "dfmic-capture",
            *pw::keys::NODE_DESCRIPTION => "DeepFilter capture",
            // WirePlumber moves a capture stream to the new default whenever it
            // changes. dfmic is the default, so following it would record our
            // own output: silence, fed back forever.
            *pw::keys::NODE_DONT_RECONNECT => "true",
        };
        set_latency(&mut props, NEUTRAL_QUANTUM);
        if let Some(target) = target.filter(|target| !target.is_empty()) {
            props.insert(*pw::keys::TARGET_OBJECT, target);
        }
        let slot = self.new_stream(
            "dfmic-capture",
            spa::utils::Direction::Input,
            props,
            StreamRole::Capture,
        )?;
        self.streams.borrow_mut().capture = Some(slot);
        Ok(())
    }

    fn start_monitor(&self, target: Option<&str>) -> Result<(), String> {
        let mut props = properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Playback",
            *pw::keys::MEDIA_ROLE => "Music",
            *pw::keys::APP_NAME => "dfmic",
            *pw::keys::NODE_NAME => "dfmic-monitor",
            *pw::keys::NODE_DESCRIPTION => "DeepFilter Mic Monitor",
        };
        set_latency(&mut props, NEUTRAL_QUANTUM);
        if let Some(target) = target.filter(|target| !target.is_empty()) {
            props.insert(*pw::keys::TARGET_OBJECT, target);
        }
        let slot = self.new_stream(
            "dfmic-monitor",
            spa::utils::Direction::Output,
            props,
            StreamRole::Monitor,
        )?;
        self.streams.borrow_mut().monitor = Some(slot);
        Ok(())
    }

    pub fn ready(&self) -> bool {
        let streams = self.streams.borrow();
        let streaming = |slot: &Option<StreamSlot>| {
            slot.as_ref().is_some_and(|slot| {
                matches!(slot.stream.state(), pw::stream::StreamState::Streaming)
            })
        };
        streaming(&streams.capture) && streaming(&streams.playback)
    }

    pub fn set_source(&self, target: &str) -> Result<(), String> {
        let _lock = self.thread_loop.lock();
        if let Some(slot) = self.streams.borrow_mut().capture.take() {
            let _ = slot.stream.disconnect();
        }
        self.start_capture(Some(target))
    }

    pub fn set_monitor(&self, enabled: bool, target: Option<&str>) -> Result<(), String> {
        // Re-enabling an already running monitor for the same device is a no-op.
        if enabled
            && self.streams.borrow().monitor.is_some()
            && self.monitor_target.borrow().as_deref() == target
        {
            return Ok(());
        }
        let _lock = self.thread_loop.lock();
        if let Some(slot) = self.streams.borrow_mut().monitor.take() {
            let _ = slot.stream.disconnect();
        }
        *self.monitor_target.borrow_mut() = target.map(str::to_owned);
        if enabled {
            self.start_monitor(target)?;
        }
        Ok(())
    }

    /// Start or stop the physical-microphone capture stream.
    ///
    /// Off when nothing needs audio: the stream is the only reason the graph
    /// keeps running with no consumers, so pausing it is what makes an idle
    /// daemon actually idle rather than a low hum of callbacks. Resuming resets
    /// the audio path, because the model's history describes audio that this
    /// stream is no longer continuing.
    pub fn set_capture_active(&self, active: bool) -> Result<(), String> {
        if self.capture_active.swap(active, Ordering::SeqCst) == active {
            return Ok(());
        }
        if active {
            // Before the stream can produce anything, so the first hop after a
            // resume is not spliced onto the pre-pause tail.
            self.shared.reset_audio();
        }
        let _lock = self.thread_loop.lock();
        let streams = self.streams.borrow();
        let Some(slot) = streams.capture.as_ref() else {
            return Ok(());
        };
        slot.stream.set_active(active).map_err(|error| {
            format!(
                "could not {} the capture stream: {error}",
                if active { "start" } else { "pause" }
            )
        })
    }

    /// Ask the graph for a small quantum while filtering, and for PipeWire's
    /// own default when we stop, so a running-but-idle dfmic never changes the
    /// graph.
    pub fn set_low_latency(&self, on: bool) -> Result<(), String> {
        let quantum = if on { ACTIVE_QUANTUM } else { NEUTRAL_QUANTUM };
        let key = CString::new(*pw::keys::NODE_LATENCY).expect("no interior NUL");
        let value = CString::new(format!("{quantum}/{RATE}")).expect("no interior NUL");
        let items = [spa::sys::spa_dict_item {
            key: key.as_ptr(),
            value: value.as_ptr(),
        }];
        let dict = spa::sys::spa_dict {
            flags: 0,
            n_items: items.len() as u32,
            items: items.as_ptr().cast_mut(),
        };

        let _lock = self.thread_loop.lock();
        let streams = self.streams.borrow();
        let mut failed = false;
        for slot in [&streams.capture, &streams.playback, &streams.monitor]
            .into_iter()
            .flatten()
        {
            // `pw_stream_update_properties` is the one entry point pipewire-rs
            // does not wrap; `pw::sys` is its own bindgen output.
            let result =
                unsafe { pw::sys::pw_stream_update_properties(slot.stream.as_raw_ptr(), &dict) };
            failed |= result < 0;
        }
        if failed {
            return Err("PipeWire refused the latency request".into());
        }
        Ok(())
    }

    pub fn set_default_source(&self, name: &str) -> Result<(), String> {
        let value = format!("{{\"name\":\"{name}\"}}");
        let set = {
            let _lock = self.thread_loop.lock();
            match self.metadata.borrow().as_ref() {
                Some((metadata, _)) => {
                    metadata.set_property(
                        pw::sys::PW_ID_CORE,
                        "default.audio.source",
                        Some("Spa:String:JSON"),
                        Some(&value),
                    );
                    true
                }
                None => false,
            }
        };
        if set {
            return Ok(());
        }
        // The metadata object has not appeared yet (a very early start, or no
        // session manager); fall back to the same call `pw-metadata` makes.
        let output = std::process::Command::new("pw-metadata")
            .args([
                "-n",
                "default",
                "0",
                "default.audio.source",
                &value,
                "Spa:String:JSON",
            ])
            .output()
            .map_err(|error| format!("could not run pw-metadata: {error}"))?;
        if output.status.success() {
            Ok(())
        } else {
            Err(format!(
                "pw-metadata could not set the default source: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ))
        }
    }

    pub fn latency_ms(&self) -> Option<(f64, f64)> {
        let _lock = self.thread_loop.lock();
        let streams = self.streams.borrow();
        let capture = stream_latency_ms(streams.capture.as_ref().map(|slot| &*slot.stream))?;
        let mut output = stream_latency_ms(streams.playback.as_ref().map(|slot| &*slot.stream))?;
        if let Some(monitor) = stream_latency_ms(streams.monitor.as_ref().map(|slot| &*slot.stream))
        {
            output = output.max(monitor);
        }
        Some((capture, output))
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        {
            let _lock = self.thread_loop.lock();
            let mut streams = self.streams.borrow_mut();
            for slot in [&streams.capture, &streams.playback, &streams.monitor]
                .into_iter()
                .flatten()
            {
                let _ = slot.stream.disconnect();
            }
            // Listeners before streams: a hook lives inside the stream it
            // watches, so the hook must go first.
            *streams = Streams::default();
        }
        self.thread_loop.stop();
        // Drop the metadata proxy while the loop is stopped but the core is
        // still connected; the remaining proxies follow as fields drop.
        *self.metadata.borrow_mut() = None;
    }
}

fn state_changed(
    shared: &Shared,
    role: StreamRole,
    _old: pw::stream::StreamState,
    new: pw::stream::StreamState,
) {
    if let pw::stream::StreamState::Error(message) = &new {
        match role {
            // The exported source is the whole point of the daemon: if it really
            // dies there is nothing to fall back to. A missing target is not that
            // — the node is exported, not linked, and PipeWire reports `ENOENT`
            // for a link whose far end went away, which happens every time a
            // microphone is unplugged underneath a running graph.
            StreamRole::Source => {
                if is_missing_target(message) {
                    shared.note_capture_loss(format!("the graph lost a node: {message}"));
                } else {
                    shared.set_error(format!("PipeWire stream error: {message}"));
                }
            }
            // Losing the microphone is not fatal: devices are unplugged, and
            // Bluetooth headsets drop their links. PipeWire reports that as a
            // stream `Error`, so the daemon is told instead of being killed, and
            // it reconnects once the node is back in the registry.
            StreamRole::Capture => {
                shared.note_capture_loss(format!("the capture stream lost its device: {message}"));
            }
            // Headphones get unplugged. Taking the microphone down with them
            // would be absurd, so this only stops monitoring.
            StreamRole::Monitor => {
                shared.note_monitor_failure(format!("Self-monitor stopped: {message}"));
            }
        }
        return;
    }
    if !matches!(role, StreamRole::Source) {
        return;
    }
    if matches!(new, pw::stream::StreamState::Streaming) {
        on_source_started(shared);
    }
}

/// Whether PipeWire's error text means "the node I was linked to is gone".
///
/// `-ENOENT` is what the server reports when a link's far end disappears, and it
/// is the difference between "the microphone was unplugged" — recoverable, the
/// daemon keeps running and reconnects — and "this stream is broken", which for
/// the exported node is fatal.
fn is_missing_target(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    message.contains("target not found") || message.contains("no such file")
}

fn registry_listener(
    registry: &pw::registry::RegistryRc,
    shared: SharedRef,
    metadata: &Rc<MetadataSlot>,
) -> pw::registry::Listener {
    let metadata = Rc::clone(metadata);
    // The callbacks are `'static`, so they own their registry handle.
    let binding = registry.clone();
    let removed = Rc::clone(&shared);
    registry
        .add_listener_local()
        .global(move |global| {
            let Some(props) = global.props else {
                return;
            };
            match global.type_ {
                pw::types::ObjectType::Device => {
                    let record = DeviceRecord {
                        bus: dict_string(props, *pw::keys::DEVICE_BUS),
                        form_factor: dict_string(props, *pw::keys::DEVICE_FORM_FACTOR),
                    };
                    shared.with_registry(|registry| {
                        registry.devices.insert(global.id, record);
                    });
                }
                pw::types::ObjectType::Metadata => {
                    if props.get("metadata.name") != Some("default") {
                        return;
                    }
                    if metadata.borrow().is_some() {
                        return;
                    }
                    let Ok(proxy) = binding.bind::<pw::metadata::Metadata, _>(global) else {
                        return;
                    };
                    let listener = {
                        let shared = Rc::clone(&shared);
                        proxy
                            .add_listener_local()
                            .property(move |subject, key, _, value| {
                                metadata_property(&shared, subject, key, value);
                                0
                            })
                            .register()
                    };
                    // Stored only after the listener exists, and with no borrow
                    // held across the bind, because binding can dispatch the
                    // property callback above.
                    *metadata.borrow_mut() = Some((proxy, listener));
                }
                pw::types::ObjectType::Node => {
                    let Some(media_class) = props.get(*pw::keys::MEDIA_CLASS) else {
                        return;
                    };
                    let sink = match media_class {
                        "Audio/Source" => false,
                        "Audio/Sink" => true,
                        _ => return,
                    };
                    let Some(name) = props.get(*pw::keys::NODE_NAME) else {
                        return;
                    };
                    if name == NODE_NAME {
                        // Ours: remember the id so links to it can be counted,
                        // but keep it out of the device list.
                        shared.with_registry(|registry| registry.own_node = Some(global.id));
                        return;
                    }
                    let device_id = props
                        .get(*pw::keys::DEVICE_ID)
                        .and_then(|id| id.parse::<u32>().ok());
                    let record = NodeRecord {
                        sink,
                        name: name.to_owned(),
                        description: props
                            .get(*pw::keys::NODE_DESCRIPTION)
                            .or_else(|| props.get(*pw::keys::NODE_NICK))
                            .unwrap_or(name)
                            .to_owned(),
                        bus: dict_string(props, *pw::keys::DEVICE_BUS),
                        form_factor: dict_string(props, *pw::keys::DEVICE_FORM_FACTOR),
                        // A node with no device behind it is a filter chain or
                        // a loopback, which cannot be a physical microphone.
                        virtual_node: props.get("node.virtual") == Some("true")
                            || device_id.is_none(),
                        device_id,
                    };
                    shared.with_registry(|registry| {
                        registry.nodes.insert(global.id, record);
                    });
                }
                pw::types::ObjectType::Link => {
                    let node = |key: &str| props.get(key).and_then(|id| id.parse::<u32>().ok());
                    let (Some(output), Some(input)) = (
                        node(*pw::keys::LINK_OUTPUT_NODE),
                        node(*pw::keys::LINK_INPUT_NODE),
                    ) else {
                        return;
                    };
                    shared.with_registry(|registry| {
                        registry.links.insert(global.id, (output, input));
                    });
                }
                _ => {}
            }
        })
        .global_remove(move |id| {
            removed.with_registry(|registry| {
                registry.nodes.remove(&id);
                registry.devices.remove(&id);
                registry.links.remove(&id);
            });
        })
        .register()
}

fn dict_string(props: &spa::utils::dict::DictRef, key: &str) -> Option<String> {
    props.get(key).map(str::to_owned)
}

fn metadata_property(shared: &Shared, subject: u32, key: Option<&str>, value: Option<&str>) {
    if subject != pw::sys::PW_ID_CORE {
        return;
    }
    let Some(key) = key else {
        return;
    };
    let name = value
        .and_then(parse_default_name)
        .filter(|name| !name.is_empty());
    match key {
        "default.audio.source" => shared.with_registry(|registry| {
            registry.default_source = name;
        }),
        "default.audio.sink" => shared.with_registry(|registry| {
            registry.default_sink = name;
        }),
        _ => {}
    }
}

fn set_latency(props: &mut pw::properties::PropertiesBox, samples: u32) {
    props.insert(*pw::keys::NODE_LATENCY, format!("{samples}/{RATE}"));
}

/// Reported latency of one stream in milliseconds.
///
/// `pw_time.rate` is the duration of one tick (1/48000 s for audio), and `delay`
/// and `buffered` are counted in ticks, so the tick duration is the only
/// conversion factor needed.
///
/// `queued` is deliberately not added. It is the sum of `pw_buffer.size`, a field
/// the *application* has to set, and pipewire-rs exposes no setter for it, so it
/// is always zero here. Including it would suggest a figure this code cannot
/// produce; the daemon adds the ring's own depth to the total separately, where
/// it knows the number.
fn stream_latency_ms(stream: Option<&pw::stream::Stream>) -> Option<f64> {
    let time = stream?.time().ok()?;
    let time = time.as_raw();
    if time.rate.denom == 0 {
        return None;
    }
    let tick_ms = f64::from(time.rate.num) / f64::from(time.rate.denom) * 1000.0;
    // Output streams report the graph delay as negative.
    let graph = (time.delay as f64).abs() * tick_ms;
    let buffer = time.buffered as f64 * tick_ms;
    Some(graph + buffer)
}

#[derive(Clone, Copy, Default)]
struct LevelSummary {
    active: bool,
    peak: f64,
    rms: f64,
    clip: bool,
}

struct MetricSlot {
    sequence: AtomicU64,
    timestamp: AtomicU64,
    peak: AtomicU32,
    sum_squares: AtomicU64,
    frames: AtomicU32,
    clip: AtomicBool,
}

impl MetricSlot {
    fn new() -> Self {
        Self {
            sequence: AtomicU64::new(0),
            timestamp: AtomicU64::new(0),
            peak: AtomicU32::new(0),
            sum_squares: AtomicU64::new(0),
            frames: AtomicU32::new(0),
            clip: AtomicBool::new(false),
        }
    }
}

struct MetricRing {
    slots: [MetricSlot; 128],
    next: AtomicU64,
}

impl MetricRing {
    fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| MetricSlot::new()),
            next: AtomicU64::new(0),
        }
    }

    fn record(&self, started: Instant, samples: &[f32]) {
        let mut peak = 0.0f32;
        let mut sum_squares = 0.0f64;
        let mut clip = false;
        for sample in samples {
            let absolute = sample.abs();
            peak = peak.max(absolute);
            sum_squares += (*sample as f64) * (*sample as f64);
            clip |= absolute >= 1.0;
        }
        let generation = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        let slot = &self.slots[(generation as usize - 1) % self.slots.len()];
        slot.sequence.store(generation * 2 - 1, Ordering::Release);
        slot.timestamp
            .store(millis_timestamp(started), Ordering::Relaxed);
        slot.peak.store(peak.to_bits(), Ordering::Relaxed);
        slot.sum_squares
            .store(sum_squares.to_bits(), Ordering::Relaxed);
        slot.frames.store(samples.len() as u32, Ordering::Relaxed);
        slot.clip.store(clip, Ordering::Relaxed);
        slot.sequence.store(generation * 2, Ordering::Release);
    }

    fn aggregate(&self, now: f64) -> LevelSummary {
        let cutoff = now - 0.050;
        let cutoff_ms = (cutoff.max(0.0) * 1000.0) as u64;
        let mut peak = 0.0f32;
        let mut sum_squares = 0.0f64;
        let mut frames = 0u64;
        let mut clip = false;
        for slot in &self.slots {
            let before = slot.sequence.load(Ordering::Acquire);
            if before == 0 || before % 2 != 0 {
                continue;
            }
            let timestamp = slot.timestamp.load(Ordering::Relaxed);
            if timestamp < cutoff_ms {
                continue;
            }
            let slot_peak = f32::from_bits(slot.peak.load(Ordering::Relaxed));
            let slot_squares = f64::from_bits(slot.sum_squares.load(Ordering::Relaxed));
            let slot_frames = slot.frames.load(Ordering::Relaxed) as u64;
            let slot_clip = slot.clip.load(Ordering::Relaxed);
            let after = slot.sequence.load(Ordering::Acquire);
            if before != after || timestamp < cutoff_ms {
                continue;
            }
            peak = peak.max(slot_peak);
            sum_squares += slot_squares;
            frames += slot_frames;
            clip |= slot_clip;
        }
        let rms = if frames == 0 {
            0.0
        } else {
            (sum_squares / frames as f64).sqrt()
        };
        LevelSummary {
            active: frames > 0,
            peak: peak as f64,
            rms,
            clip: clip || peak >= 0.999,
        }
    }
}

struct TimingSlot {
    timestamp: AtomicU64,
    milliseconds: AtomicU64,
}

struct TimingRing {
    slots: [TimingSlot; 256],
    next: AtomicUsize,
}

impl TimingRing {
    fn new(_started: Instant) -> Self {
        Self {
            slots: std::array::from_fn(|_| TimingSlot {
                timestamp: AtomicU64::new(0),
                milliseconds: AtomicU64::new(0),
            }),
            next: AtomicUsize::new(0),
        }
    }

    fn record(&self, timestamp: f64, milliseconds: f64) {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.slots.len();
        let slot = &self.slots[index];
        slot.milliseconds
            .store(milliseconds.to_bits(), Ordering::Relaxed);
        slot.timestamp.store(timestamp.to_bits(), Ordering::Release);
    }

    fn mean_last_second(&self, now: f64) -> Option<f64> {
        let mut sum = 0.0;
        let mut count = 0u64;
        for slot in &self.slots {
            let timestamp = f64::from_bits(slot.timestamp.load(Ordering::Acquire));
            if timestamp > 0.0 && now - timestamp <= 1.0 {
                sum += f64::from_bits(slot.milliseconds.load(Ordering::Relaxed));
                count += 1;
            }
        }
        (count > 0).then_some(sum / count as f64)
    }
}

/// Single-producer single-consumer sample ring, lock-free.
///
/// The buffer is `UnsafeCell` rather than `Box<[f32]>` because both sides write
/// to and read from it through shared references: the producer owns the range
/// `[read, write)` and the consumer owns `[.., read)`, and the two never touch
/// the same sample. That is the invariant [`Sync`] rests on, and it is why every
/// access goes through [`UnsafeCell::get`] instead of a slice.
///
/// Which side a call belongs to is not a suggestion: `push` may only be called by
/// the producer, and `pop`, `discard` and `clear` only by the consumer. In
/// particular, asking a ring to drop its contents from a third thread would move
/// the read index under the consumer's feet, so `Shared` asks for it with a
/// counter (`request_reprime`) and the consumer does the discarding itself.
pub mod ring {
    use std::cell::UnsafeCell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct Ring {
        buffer: Box<[UnsafeCell<f32>]>,
        write: AtomicUsize,
        read: AtomicUsize,
    }

    // Sound because the ring is single-producer single-consumer: `push` is only
    // called by the thread that owns the write side, and `pop`/`discard`/`clear`
    // only by the thread that owns the read side. The two indices are published
    // with release/acquire ordering, so samples a producer writes are only read
    // after the release store of `write`, and samples a consumer frees are only
    // written after the release store of `read`.
    unsafe impl Sync for Ring {}

    impl Ring {
        pub fn new(capacity: usize) -> Self {
            Self {
                buffer: (0..capacity)
                    .map(|_| UnsafeCell::new(0.0f32))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                write: AtomicUsize::new(0),
                read: AtomicUsize::new(0),
            }
        }

        pub fn len(&self) -> usize {
            let write = self.write.load(Ordering::Acquire);
            let read = self.read.load(Ordering::Acquire);
            write.wrapping_sub(read)
        }

        /// Samples the ring can hold in total.
        pub fn capacity(&self) -> usize {
            self.buffer.len()
        }

        pub fn push(&self, samples: &[f32]) -> bool {
            if self.len() + samples.len() > self.buffer.len() {
                return false;
            }
            let mut write = self.write.load(Ordering::Relaxed);
            let mut rest = samples;
            while !rest.is_empty() {
                let index = write % self.buffer.len();
                let take = rest.len().min(self.buffer.len() - index);
                unsafe {
                    std::ptr::copy_nonoverlapping(rest.as_ptr(), self.buffer[index].get(), take);
                }
                write = write.wrapping_add(take);
                rest = &rest[take..];
            }
            self.write.store(write, Ordering::Release);
            true
        }

        pub fn clear(&self) {
            let write = self.write.load(Ordering::Acquire);
            self.read.store(write, Ordering::Release);
        }

        /// Discard the oldest samples without writing or allocating. Only the
        /// existing consumer calls this, preserving the SPSC ring invariant.
        pub fn discard(&self, frames: usize) {
            let available = self.len().min(frames);
            let read = self.read.load(Ordering::Relaxed);
            self.read
                .store(read.wrapping_add(available), Ordering::Release);
        }

        pub fn pop(&self, samples: &mut [f32]) -> usize {
            let available = self.len().min(samples.len());
            let mut read = self.read.load(Ordering::Relaxed);
            let mut rest = &mut samples[..available];
            while !rest.is_empty() {
                let index = read % self.buffer.len();
                let take = rest.len().min(self.buffer.len() - index);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.buffer[index].get(),
                        rest.as_mut_ptr(),
                        take,
                    );
                }
                read = read.wrapping_add(take);
                rest = &mut rest[take..];
            }
            samples[available..].fill(0.0);
            self.read.store(read, Ordering::Release);
            available
        }
    }
}
