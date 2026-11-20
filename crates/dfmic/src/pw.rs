//! The PipeWire bridge, in Rust.
//!
//! Three streams on one thread loop: a capture stream reading a physical
//! microphone, an output stream exported as the `dfmic-source` node (the
//! virtual microphone), and an optional monitor stream that plays the same
//! audio back so the user can hear themselves.
//!
//! Process callbacks run on PipeWire's realtime thread and only touch the
//! lock-free rings and atomics below. Everything else happens on the main
//! thread while holding the loop lock, which is what makes the proxies and
//! listeners safe to touch from both: PipeWire stops dispatching callbacks
//! while the lock is held.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

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
    /// The stream state cannot answer "is anything listening": our capture
    /// stream keeps the graph running, so the exported node never suspends and
    /// never leaves `Streaming` when its last consumer goes away. The links are
    /// the authoritative answer, and PipeWire publishes them as globals.
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
    pub consumers: AtomicBool,
    pub monitor_active: AtomicBool,
    pub filtering: AtomicBool,
    pub min_buffer: AtomicUsize,
    /// The output queue's target depth in samples, adapted at runtime.
    ///
    /// libDF calls this `proc_delay`: it starts at one hop and moves as the
    /// machine demands, rather than being fixed by configuration.
    proc_delay: AtomicUsize,
    /// Earliest time (ms since start) at which another shrink may be tried.
    shrink_after_ms: AtomicU64,
    /// Current wait between shrink attempts, doubled on every failure.
    shrink_backoff_ms: AtomicU64,
    priming: AtomicBool,
    monitor_priming: AtomicBool,
    reset_generation: AtomicU64,
    input_levels: MetricRing,
    output_levels: MetricRing,
    timing: TimingRing,
    pub inferences: AtomicU64,
    /// Bumped whenever the input ring reaches a whole hop. The worker sleeps on
    /// this and the capture callback wakes it, so a hop is processed the moment
    /// it is complete instead of on the next poll tick.
    hop_ready: AtomicU32,
    voice: AtomicBool,
    agc_gain: AtomicU32,
    started: Instant,
}

impl Shared {
    pub fn new(capacity: usize) -> Arc<Self> {
        let started = Instant::now();
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
            consumers: AtomicBool::new(false),
            monitor_active: AtomicBool::new(false),
            filtering: AtomicBool::new(false),
            min_buffer: AtomicUsize::new(0),
            proc_delay: AtomicUsize::new(HOP_SIZE * MIN_HOPS),
            shrink_after_ms: AtomicU64::new(0),
            shrink_backoff_ms: AtomicU64::new(0),
            priming: AtomicBool::new(true),
            monitor_priming: AtomicBool::new(true),
            reset_generation: AtomicU64::new(0),
            input_levels: MetricRing::new(),
            output_levels: MetricRing::new(),
            timing: TimingRing::new(started),
            inferences: AtomicU64::new(0),
            hop_ready: AtomicU32::new(0),
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
    }

    pub fn error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|mut error| error.take())
    }

    pub fn set_monitor_active(&self, active: bool) {
        self.monitor_active.store(active, Ordering::SeqCst);
        self.monitor_priming.store(true, Ordering::Release);
    }

    /// Ask the worker to reset both ends. The worker is the input consumer and
    /// the output producer, so this preserves every ring's SPSC ownership.
    pub fn reset_audio(&self) {
        self.reset_generation.fetch_add(1, Ordering::AcqRel);
        self.priming.store(true, Ordering::Release);
        self.monitor_priming.store(true, Ordering::Release);
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
        let input = self.input_levels.aggregate(now);
        let output = self.output_levels.aggregate(now);
        Levels {
            input_active: input.active,
            input_peak: input.peak,
            input_rms: input.rms,
            input_clip: input.clip,
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
    fn grow_delay(&self) {
        let current = self.proc_delay.load(Ordering::Relaxed);
        // One second of audio is where libDF gives up entirely.
        if current < RATE as usize {
            self.proc_delay.store(current + HOP_SIZE, Ordering::Relaxed);
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
    pub fn wait_for_hop(&self) {
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
            tv_sec: 0,
            tv_nsec: 2_000_000,
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
    fn wake_worker(&self) {
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
    }

    /// Record how many consumers our node has, resetting the queue on arrival.
    fn set_consumers(&self, count: usize) {
        let was = self.consumers.swap(count > 0, Ordering::SeqCst);
        if count > 0 && !was {
            // A consumer just attached: whatever is queued is from before it,
            // so start from silence rather than from a stale tail.
            self.output.clear();
            self.priming.store(true, Ordering::Release);
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
/// bytes and are clamped to the mapped region before use.
unsafe fn samples(data: &mut [u8], offset: u32, size: u32) -> &mut [f32] {
    let offset = offset as usize;
    let frames = (size as usize / 4).min(data.len().saturating_sub(offset) / 4);
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
    shared.input_levels.record(shared.started, samples);
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

    let (ring, priming) = if monitor {
        (&shared.monitor_output, &shared.monitor_priming)
    } else {
        (&shared.output, &shared.priming)
    };
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
/// This deliberately says nothing about consumers: with our capture stream
/// keeping the graph running, this fires once at startup and then not again when
/// applications come and go. Link bookkeeping in the registry is what tracks
/// them; this only makes sure the queue starts from silence.
fn on_source_started(shared: &Shared) {
    shared.output.clear();
    shared.priming.store(true, Ordering::Release);
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
        let context = pw::context::ContextRc::new(&thread_loop, None)
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
                        shared.set_error(format!("{message} ({result})"));
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
        // Deliberately *not* `node.passive`. A passive node stays running with
        // no consumers, so the stream never leaves `Streaming` and the daemon
        // never learns that the last application disconnected: it would filter
        // for ever and the output meter would keep showing signal. The stream
        // state is how "is anything listening" is detected (`on_source_started`
        // / `on_source_stopped`), so the node has to go idle when it has no
        // consumers.
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
        shared.set_error(format!("PipeWire stream error: {message}"));
        return;
    }
    if !matches!(role, StreamRole::Source) {
        return;
    }
    if matches!(new, pw::stream::StreamState::Streaming) {
        on_source_started(shared);
    }
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
/// `pw_time.rate` is the duration of one tick (1/48000 s for audio) and
/// `delay`, `queued` and `buffered` are all counted in ticks, so the tick
/// duration is the only conversion factor needed.
fn stream_latency_ms(stream: Option<&pw::stream::Stream>) -> Option<f64> {
    let time = stream?.time().ok()?;
    let time = time.as_raw();
    if time.rate.denom == 0 {
        return None;
    }
    let tick_ms = f64::from(time.rate.num) / f64::from(time.rate.denom) * 1000.0;
    // Output streams report the graph delay as negative.
    let graph = (time.delay as f64).abs() * tick_ms;
    let queue = (time.queued + time.buffered) as f64 * tick_ms;
    Some(graph + queue)
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
pub mod ring {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct Ring {
        buffer: Box<[f32]>,
        write: AtomicUsize,
        read: AtomicUsize,
    }

    impl Ring {
        pub fn new(capacity: usize) -> Self {
            Self {
                buffer: vec![0.0; capacity].into_boxed_slice(),
                write: AtomicUsize::new(0),
                read: AtomicUsize::new(0),
            }
        }

        pub fn len(&self) -> usize {
            let write = self.write.load(Ordering::Acquire);
            let read = self.read.load(Ordering::Acquire);
            write.wrapping_sub(read)
        }

        pub fn push(&self, samples: &[f32]) -> bool {
            if self.len() + samples.len() > self.buffer.len() {
                return false;
            }
            let mut write = self.write.load(Ordering::Relaxed);
            for &sample in samples {
                let index = write % self.buffer.len();
                unsafe {
                    let pointer = self.buffer.as_ptr().add(index) as *mut f32;
                    pointer.write(sample);
                }
                write = write.wrapping_add(1);
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
            for slot in samples.iter_mut().take(available) {
                *slot = self.buffer[read % self.buffer.len()];
                read = read.wrapping_add(1);
            }
            for slot in samples.iter_mut().skip(available) {
                *slot = 0.0;
            }
            self.read.store(read, Ordering::Release);
            available
        }
    }
}
