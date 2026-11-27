//! The virtual microphone daemon and its live audio worker.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command as ProcessCommand;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use dfn_dsp::{DfnPipeline, Params, HOP_SIZE};
use dfn_npu::NpuRuntime;

use crate::agc::{self, Agc};
use crate::control::{self, Command, ControlServer, Published, Reply, Request};
use crate::models;
use crate::modelwatch;
use crate::notifications::{self, Notifier};
use crate::proto::{
    Failure, MonitorTarget, Param, State, NODE_DESCRIPTION, NODE_NAME, SOFT_PRESET, SOURCE_DEFAULT,
};
use crate::pw::{Bridge, Shared};
use crate::service;
use crate::settings::{self, DeviceSettings, Settings};

const RATE: f64 = 48_000.0;
const MODEL_DELAY_MS: f64 = HOP_SIZE as f64 / RATE * 1000.0;

/// How long the worker blocks waiting for a hop while it is filtering.
///
/// Short on purpose: it only bounds how long a shutdown or a state change waits,
/// because the capture callback wakes the futex as soon as a hop completes.
const HOP_POLL: Duration = Duration::from_millis(2);
/// How long the event loop waits for a request when everything is settled.
///
/// This is a poll: every iteration re-derives source availability, model state
/// and the state machine, then publishes diagnostics if they are due. It is
/// long on purpose, because a settled daemon should cost nothing — commands
/// arrive through this very channel and wake it immediately, and every async
/// source (the registry, the worker, a download, a signal) rings the same bell.
///
/// It matters that filtering is *not* settled: while the filter runs, the
/// diagnostic counters change continuously and nothing else wakes the loop, so
/// [`ACTIVE_POLL`] is used instead. The old single timeout meant the window's
/// inference and overrun figures froze for half a minute at a time.
const STEADY_POLL: Duration = Duration::from_secs(30);
/// How long it waits while filtering, so the published counters track the audio.
///
/// The meters do not depend on this — `Levels()` is stateless and read directly —
/// but `InferenceMs`, `Inferences`, `Overruns` and `Underruns` are properties, and
/// they are only refreshed when the loop comes round. This is a *fifth* of a
/// second rather than a tenth because the publish below is rate-limited to 100 ms
/// anyway and the audio thread is what the CPU is for.
const ACTIVE_POLL: Duration = Duration::from_millis(200);
/// How long it waits while a start-up or a download is in flight.
const BUSY_POLL: Duration = Duration::from_millis(50);
/// How long it waits when the model is missing and cannot be watched.
///
/// Only used if `inotify` is unavailable: the watch normally wakes the loop the
/// moment the files appear, so this is a fallback rather than a cadence.
const MODEL_POLL: Duration = Duration::from_millis(2000);

/// How long it blocks while nothing is listening.
///
/// Nothing feeds the ring then, so this is a sleep rather than a poll: the only
/// thing that ends it is a consumer or the monitor arriving, and both wake the
/// futex directly.
const IDLE_POLL: Duration = Duration::from_millis(100);

pub struct Options {
    pub models: PathBuf,
    pub device: String,
    pub source: Option<String>,
    pub take_default: Option<bool>,
    pub param_overrides: Vec<(Param, f64)>,
}

#[derive(Clone)]
struct WorkerConfig {
    enabled: bool,
    available: bool,
    agc: bool,
    params: Params,
}

enum WorkerEvent {
    Failed(String),
}

/// Ask for realtime scheduling for the NPU worker, if the system allows it.
///
/// This is an optimisation, never a requirement: without it the worker is an
/// ordinary thread and still works, it just has a longer tail. Measured over
/// 4000 hops of real audio, promoting it moved p99.9 from 8.3 ms to 6.8 ms and
/// the worst case from 50 ms to 20 ms. A long stall here starves the output
/// ring and clicks, so a shorter tail buys dropout margin, not lower latency:
/// the queue floor is set by bursty production, not by jitter.
///
/// It needs `CAP_SYS_NICE`, which a NixOS user does not normally have; the
/// module grants it when `services.dfmic.realtime` is enabled. Priority 40 is
/// deliberately below PipeWire's own 88, so audio callbacks always preempt us.
fn promote_to_realtime() {
    const PRIORITY: libc::c_int = 40;
    let param = libc::sched_param {
        sched_priority: PRIORITY,
    };
    let result =
        unsafe { libc::pthread_setschedparam(libc::pthread_self(), libc::SCHED_FIFO, &param) };
    if result != 0 {
        // Expected on a default install: say so once, and carry on normally.
        eprintln!(
            "dfmic: note: the NPU worker is running at normal priority ({}); \
             enable services.dfmic.realtime for a shorter worst case",
            std::io::Error::from_raw_os_error(result)
        );
    }
}

fn worker_loop(
    mut pipeline: Option<DfnPipeline<NpuRuntime>>,
    models: PathBuf,
    device: String,
    shared: Arc<Shared>,
    config_rx: Receiver<WorkerConfig>,
    stop: Arc<AtomicBool>,
) {
    let mut config = WorkerConfig {
        enabled: true,
        available: true,
        agc: true,
        params: Params::default(),
    };
    // Kept separately from `config.params`: without a pipeline there is nothing
    // to write them into, and they have to survive until there is.
    let mut params = Params::default();
    // How long to wait before looking for the model again, and the last error
    // reported, so a broken model does not fill the log.
    let mut next_load_check = Instant::now();
    let mut last_load_error: Option<String> = None;
    let mut input = vec![0.0f32; HOP_SIZE];
    let mut output = vec![0.0f32; HOP_SIZE];
    let mut agc = Agc::new();
    let mut reset_generation = 0;
    while !stop.load(Ordering::Acquire) {
        while let Ok(next) = config_rx.try_recv() {
            shared
                .min_buffer
                .store(next.params.min_buffer as usize, Ordering::Release);
            params = next.params.clone();
            if let Some(pipeline) = pipeline.as_mut() {
                *pipeline.params_mut() = params.clone();
            }
            config = next;
        }
        // The model may have been downloaded while we were running. Picking it
        // up here, rather than at startup, is what lets the daemon run without
        // one: it starts in pass-through and begins filtering once the files
        // appear.
        if pipeline.is_none() && Instant::now() >= next_load_check {
            match NpuRuntime::load(&models, &device, None) {
                Ok(runtime) => {
                    let mut loaded = DfnPipeline::new(runtime, params.clone());
                    // The rings may hold audio from before the model existed;
                    // the model's history must not be spliced onto it.
                    loaded.reset();
                    pipeline = Some(loaded);
                    shared.request_reprime();
                    last_load_error = None;
                    eprintln!("dfmic: model loaded; filtering is available");
                }
                Err(error) => {
                    // Expected until the model is downloaded, so it is not a
                    // failure: report it once, then look again in a while.
                    if last_load_error.as_deref() != Some(error.as_str()) {
                        eprintln!("dfmic: no model yet ({error})");
                        last_load_error = Some(error);
                    }
                    next_load_check = Instant::now() + Duration::from_secs(5);
                }
            }
        }
        let next_reset = shared.reset_generation();
        if next_reset != reset_generation {
            // Both ends of the pipeline, not just the rings: the STFT history,
            // the normalisation states and the model's GRU and convolution
            // history all describe audio that this stream is no longer
            // continuing (a different device, or a restart).
            shared.input.clear();
            shared.request_reprime();
            if let Some(pipeline) = pipeline.as_mut() {
                pipeline.reset();
            }
            agc = Agc::new();
            reset_generation = next_reset;
        }
        let listening = shared.listening();
        let filtering = config.enabled && config.available && pipeline.is_some();
        let should_process = listening && filtering;
        if !config.available || !listening {
            while shared.input.pop(&mut input) > 0 {}
            shared.request_reprime();
            shared.set_agc(0.0, false);
            // The AGC's gain, floor and hold describe a session that has ended.
            // Left alone, the first hop after a resume is boosted by a gain from
            // hours ago and gated against a floor from another room.
            agc = Agc::new();
            // Nothing is recording from us and the capture callback is not
            // feeding the ring, so there is nothing to poll for: sleep until the
            // state changes. Whoever makes us listen again wakes this futex, so
            // the timeout is only a safety net — and a long one is what makes an
            // idle daemon cost nothing instead of 500 wakeups a second.
            shared.wait_for_hop(IDLE_POLL);
            continue;
        }
        if shared.input.len() < HOP_SIZE {
            // The capture callback wakes us the moment a hop is complete, so
            // this returns within microseconds of the data arriving rather than
            // on the next poll tick.
            shared.wait_for_hop(HOP_POLL);
            continue;
        }
        // A failed shrink deepened the target; the pad was asked for here, by the
        // only thread allowed to write this ring. Silence rather than a repeat of
        // the last hop: the depth exists to absorb jitter, not to echo audio.
        let pad = shared.take_pad();
        if pad > 0 {
            // A stack buffer, not an allocation: this is the audio worker, and
            // the pad is bounded by one second of 48 kHz samples.
            let silence = [0.0f32; 1024];
            let mut left = pad;
            while left > 0 {
                let take = silence.len().min(left);
                if !shared.output.push(&silence[..take]) {
                    break;
                }
                left -= take;
            }
        }
        shared.input.pop(&mut input);
        if should_process {
            let input_rms = (input.iter().map(|sample| sample * sample).sum::<f32>()
                / input.len() as f32)
                .sqrt();
            let started = Instant::now();
            let Some(pipeline) = pipeline.as_mut() else {
                unreachable!("should_process implies a pipeline")
            };
            let lsnr = pipeline.process(&input, &mut output);
            shared.record_inference(started.elapsed().as_secs_f64());
            let (gain, voice) = agc.process(
                &mut output,
                agc::Hop {
                    lsnr,
                    input_rms,
                    min_db_thresh: config.params.min_db_thresh,
                },
                config.agc,
            );
            shared.set_agc(gain, voice);
        } else {
            // Listening, but not filtering — either switched off or with no
            // model yet: hand the microphone through untouched. Every other
            // combination returned above.
            output.copy_from_slice(&input);
            shared.set_agc(0.0, false);
        }
        if shared.consumers.load(Ordering::Acquire) {
            let _ = shared.output.push(&output);
        }
        if shared.monitor_active.load(Ordering::Acquire) {
            let _ = shared.monitor_output.push(&output);
        }
    }
}

type WorkerHandle = (Sender<WorkerConfig>, Receiver<WorkerEvent>, JoinHandle<()>);

fn spawn_worker(
    models: PathBuf,
    device: String,
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
) -> Result<WorkerHandle, String> {
    // Loading happens inside the thread, not here: a missing model is a normal
    // state (the daemon runs and passes audio through), not a startup failure.
    let (config_tx, config_rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let shared_for_wake = Arc::clone(&shared);
    let thread = thread::Builder::new()
        .name("dfmic-npu".into())
        .spawn(move || {
            promote_to_realtime();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                worker_loop(None, models, device, shared, config_rx, stop);
            }));
            if let Err(panic) = result {
                let message = panic
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown NPU worker panic".into());
                let _ = event_tx.send(WorkerEvent::Failed(format!(
                    "The NPU stopped while processing audio: {message}"
                )));
                // Nothing polls `worker_events` any more: the loop is asleep on
                // its request channel, so it has to be told.
                shared_for_wake.wake_loop();
            }
        })
        .map_err(|error| format!("could not start the NPU worker: {error}"))?;
    Ok((config_tx, event_rx, thread))
}

fn default_backup_path() -> PathBuf {
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/state")))
        .unwrap_or_else(|| PathBuf::from(".local/state"))
        .join("dfmic/default-source")
}

fn metadata_set_default(name: &str) -> Result<(), String> {
    let value = format!("{{\"name\":\"{name}\"}}");
    let output = ProcessCommand::new("pw-metadata")
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
            "could not restore PipeWire default source {name:?}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn remove_backup() {
    if let Err(error) = fs::remove_file(default_backup_path()) {
        if error.kind() != std::io::ErrorKind::NotFound {
            eprintln!("dfmic: could not remove default-source backup: {error}");
        }
    }
}

fn recover_stale_default() -> Result<Option<String>, String> {
    let path = default_backup_path();
    let previous = match fs::read_to_string(&path) {
        Ok(previous) => previous,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("could not read {}: {error}", path.display())),
    };
    let previous = previous.trim().to_string();
    if previous.is_empty() || previous == NODE_NAME {
        remove_backup();
        return Ok(None);
    }
    metadata_set_default(&previous)?;
    remove_backup();
    Ok(Some(previous))
}

struct Takeover {
    previous: Option<String>,
    active: bool,
    /// The user overrode the default once, so stop taking it over.
    ///
    /// Without this the takeover re-runs on the next tick — `active` is false and
    /// the current default is the user's choice — and overwrites both the default
    /// and the record of the device we displaced. Enabling filtering again is
    /// what clears it.
    suppressed: bool,
}

impl Takeover {
    fn sync(&mut self, bridge: &Bridge, shared: &Shared, should_take: bool) -> Result<(), String> {
        let current = shared
            .registry
            .lock()
            .ok()
            .and_then(|registry| registry.default_source.clone());
        if self.active && current.as_deref() != Some(NODE_NAME) {
            // The user picked a different default. Honour it: stop taking over
            // and stop retrying, or this would fight the choice every tick. The
            // device we displaced is still remembered — it is the backup file and
            // `previous` — but it is no longer ours to restore, so the record is
            // dropped deliberately.
            self.active = false;
            self.previous = None;
            self.suppressed = true;
            remove_backup();
            return Ok(());
        }
        if !self.suppressed && !should_take {
            if self.active {
                if current.as_deref() == Some(NODE_NAME) {
                    if let Some(previous) = self.previous.as_deref() {
                        bridge.set_default_source(previous)?;
                    }
                }
                self.active = false;
                self.previous = None;
                remove_backup();
            }
            return Ok(());
        }
        if self.suppressed || self.active || current.as_deref() == Some(NODE_NAME) {
            return Ok(());
        }
        let Some(previous) = current.filter(|name| !name.is_empty()) else {
            return Ok(());
        };
        settings::save_atomic(&default_backup_path(), &previous)?;
        bridge.set_default_source(NODE_NAME)?;
        self.previous = Some(previous);
        self.active = true;
        Ok(())
    }
}

struct Daemon {
    options: Options,
    settings: Settings,
    notices: Vec<String>,
    shared: Arc<Shared>,
    bridge: Bridge,
    control: ControlServer,
    notifier: Notifier,
    worker: Sender<WorkerConfig>,
    worker_events: Receiver<WorkerEvent>,
    worker_thread: Option<JoinHandle<()>>,
    worker_stop: Arc<AtomicBool>,
    terminate: Arc<AtomicBool>,
    takeover: Takeover,
    last_failure: Failure,
    autostart: bool,
    state: State,
    source_available: bool,
    /// The device actually captured from: `settings.source`, resolved.
    ///
    /// They differ when the setting is [`SOURCE_DEFAULT`], where this follows
    /// the system default and moves when it does.
    resolved: String,
    monitor_stream: Option<String>,
    last_sink: String,
    /// The graph quantum currently requested, so we only talk to PipeWire when
    /// the answer changes. `None` means "leave the graph alone".
    low_latency: Option<bool>,
    /// Whether the filter is on. Runtime state, never persisted: every start
    /// begins with it off, so a login cannot silently take the microphone over.
    enabled: bool,
    /// Where the model is read from and downloaded into.
    ///
    /// Resolved once from `--models`/`DFMIC_MODELS`/the data directory and kept,
    /// so `ModelPresent`, `ModelDownloadPossible` and the download itself all
    /// answer about the same directory. Recomputed from the ambient environment
    /// they would disagree with the worker, which loads from `options.models`.
    models_root: PathBuf,
    /// Whether the model files are on disk, refreshed every event-loop tick.
    model_present: bool,
    /// A download started by `DownloadModel()`, while it is still running.
    model_download: Option<ModelDownload>,
    /// Watches for the model appearing on disk, so a missing one does not have
    /// to be polled for.
    model_watch: Option<modelwatch::ModelWatch>,
    /// The directory that watch is on.
    ///
    /// Kept so the watch can follow the tree downwards: the model directory is
    /// usually created in several steps, and a watch on a distant ancestor only
    /// reports the first one.
    model_watch_root: Option<PathBuf>,
}

/// A model download running on its own thread.
///
/// The thread only reports through these two shared cells; the event loop polls
/// them, so nothing touches the daemon's state from another thread.
struct ModelDownload {
    result: Arc<Mutex<Option<Result<PathBuf, String>>>>,
}

impl ModelDownload {
    /// Take the outcome once the thread has finished, without blocking.
    fn take(&self) -> Option<Result<PathBuf, String>> {
        self.result.lock().ok().and_then(|mut slot| slot.take())
    }
}

impl Daemon {
    fn new(
        options: Options,
        restored: Option<String>,
        control: ControlServer,
        shared: Arc<Shared>,
        sender: Sender<Request>,
        requests: Receiver<Request>,
    ) -> Result<(Self, Receiver<Request>), String> {
        let (mut settings, mut notices) = settings::load()?;
        // The one directory every model question is asked about. `options.models`
        // has already been resolved from `--models`/`DFMIC_MODELS`/the data
        // directory by the caller, so it is the same path the worker loads from.
        let models_root = options.models.clone();
        if let Some(source) = &options.source {
            settings.source = Some(source.clone());
        }
        if let Some(take_default) = options.take_default {
            settings.take_default = take_default;
        }
        let initial_source = settings.source.clone().unwrap_or_default();
        let following = initial_source == SOURCE_DEFAULT;

        let bridge = Bridge::start(
            NODE_NAME,
            NODE_DESCRIPTION,
            // Following the default means the target is unknown until the
            // registry has been read, which happens after the stream is up.
            // Starting unpinned is safe here: the takeover has not happened
            // yet, so the system default is still a physical microphone.
            (!initial_source.is_empty() && !following).then_some(initial_source.as_str()),
            Arc::clone(&shared),
        )?;
        // Registry and metadata delivery is asynchronous. Wait only for the
        // initial snapshot, never for audio or for the NPU.
        thread::sleep(Duration::from_millis(200));
        let default_source = shared.registry.lock().ok().and_then(|registry| {
            registry
                .default_source_name()
                .filter(|name| name != NODE_NAME)
        });
        if settings.source.is_none() {
            settings.source = default_source.clone().or_else(|| restored.clone());
        }
        let mut source = if following {
            default_source.clone().unwrap_or_default()
        } else {
            settings.source.clone().unwrap_or_default()
        };
        if source == NODE_NAME {
            notices.push(
                "The saved source was DeepFilter Mic itself; the real system default was selected."
                    .into(),
            );
            source = default_source.clone().unwrap_or_default();
            settings.source = (!source.is_empty()).then_some(source.clone());
        }
        let source_available = !source.is_empty()
            && shared
                .registry
                .lock()
                .is_ok_and(|registry| registry.source(&source).is_some());
        if !source.is_empty() && !source_available {
            notices.push(format!(
                "Saved microphone {source:?} is not connected. DeepFilter Mic will stay silent until it returns."
            ));
        }
        if settings.source != options.source && options.source.is_none() {
            settings::save(&settings)?;
        }
        if source_available && source != initial_source {
            bridge.set_source(&source)?;
        }
        let mut device_settings = settings.devices.get(&source).cloned().unwrap_or_default();
        for (param, value) in &options.param_overrides {
            validate_param(*param, *value)?;
            if device_settings.soft && SOFT_PRESET.iter().any(|(key, _)| key == param) {
                return Err(format!("--{} cannot override soft mode", param.spec().key));
            }
            device_settings.set(*param, *value);
        }
        settings
            .devices
            .insert(source.clone(), device_settings.clone());
        if !options.param_overrides.is_empty() {
            settings::save(&settings)?;
        }
        if settings.monitor {
            let target = shared
                .registry
                .lock()
                .ok()
                .map(|registry| registry.monitor_target())
                .unwrap_or(MonitorTarget {
                    name: String::new(),
                    description: String::new(),
                    kind: "unknown".into(),
                });
            if target.kind != "headphones" {
                settings.monitor = false;
                notices.push(format!(
                    "Self-monitor stayed off because the default output is {} ({}), not headphones.",
                    if target.description.is_empty() { "an unknown output" } else { &target.description },
                    target.kind
                ));
                settings::save(&settings)?;
            }
        }
        let worker_stop = Arc::new(AtomicBool::new(false));
        let (worker, worker_events, worker_thread) = spawn_worker(
            options.models.clone(),
            options.device.clone(),
            Arc::clone(&shared),
            Arc::clone(&worker_stop),
        )?;
        let autostart = if !service::unit_installed() {
            false
        } else {
            match service::autostart_enabled() {
                Ok(enabled) => enabled,
                Err(error) => {
                    notices.push(error);
                    false
                }
            }
        };
        let notifier = Notifier::new(control.connection(), sender);
        if settings.notify {
            if let Some(previous) = &restored {
                notifier.report(
                    "Microphone default restored",
                    &format!(
                        "DeepFilter Mic recovered after an unclean stop and restored {previous}."
                    ),
                    false,
                );
            }
        }
        let terminate = Arc::new(AtomicBool::new(false));
        // A flag alone is not enough now that the loop sleeps on its channel: a
        // signal has to wake it as well, or SIGTERM would sit unread for as long
        // as the safety timeout. The iterator owns a self-pipe, so this thread
        // blocks in `read` rather than polling.
        {
            let mut signals = signal_hook::iterator::Signals::new([
                signal_hook::consts::SIGINT,
                signal_hook::consts::SIGTERM,
            ])
            .map_err(|error| format!("could not install termination handler: {error}"))?;
            let terminate = Arc::clone(&terminate);
            let wake = Arc::clone(&shared);
            thread::Builder::new()
                .name("dfmic-signals".into())
                .spawn(move || {
                    for _ in signals.forever() {
                        terminate.store(true, Ordering::SeqCst);
                        wake.wake_loop();
                    }
                })
                .map_err(|error| format!("could not start the signal thread: {error}"))?;
        }
        let mut daemon = Self {
            options,
            settings,
            notices,
            shared,
            bridge,
            control,
            notifier,
            worker,
            worker_events,
            worker_thread: Some(worker_thread),
            worker_stop,
            terminate,
            takeover: Takeover {
                previous: None,
                active: false,
                suppressed: false,
            },
            last_failure: service::last_failure().unwrap_or(Failure {
                message: String::new(),
                when: 0,
                phase: String::new(),
            }),
            autostart,
            state: State::Starting,
            source_available,
            resolved: source.clone(),
            monitor_stream: None,
            last_sink: String::new(),
            low_latency: None,
            enabled: false,
            models_root: models_root.clone(),
            model_present: models::is_ready(Some(&models_root)),
            model_download: None,
            model_watch: None,
            model_watch_root: None,
        };
        daemon.send_worker_config();
        daemon.sync_monitor()?;
        daemon.sync_state()?;
        // Replace the "starting" state published above with the real one.
        daemon.control.publish(daemon.snapshot(), true);
        Ok((daemon, requests))
    }

    fn current_device(&self) -> DeviceSettings {
        self.settings
            .devices
            .get(&self.resolved)
            .cloned()
            .unwrap_or_default()
    }

    /// Whether the source setting follows the system default.
    fn following(&self) -> bool {
        self.settings.source.as_deref() == Some(SOURCE_DEFAULT)
    }

    /// Move the capture stream to whatever the system default now is.
    ///
    /// Only meaningful while following. Our own node is skipped: once the
    /// takeover has run, dfmic *is* the default source, and following that
    /// would make the daemon record its own output.
    fn follow_default_source(&mut self) -> Result<(), String> {
        if !self.following() {
            return Ok(());
        }
        // While the takeover is active the system default *is* us, so the
        // device to follow is the one we displaced. Reading it from the
        // takeover rather than from the registry is what keeps "follow the
        // default" from meaning "record our own output".
        let by_registry = self
            .shared
            .registry
            .lock()
            .ok()
            .and_then(|registry| registry.default_source_name())
            .filter(|name| name != NODE_NAME);
        let next = match by_registry {
            Some(name) => name,
            None => match self.takeover.previous.clone() {
                Some(previous) => previous,
                // Nothing to follow yet, and what we already follow is still
                // the best guess: hold it rather than dropping to no device.
                None if !self.resolved.is_empty() => self.resolved.clone(),
                None => return Ok(()),
            },
        };
        // An empty name means the system has no default microphone at all right
        // now; stay on what we have rather than pinning silence to a device
        // that is gone.
        if next.is_empty() || next == self.resolved {
            return Ok(());
        }
        self.resolved = next;
        // Best effort: a device that vanished a moment ago is retried by the
        // availability check on the next tick, and one that is gone must not take
        // the daemon with it.
        if let Err(error) = self.bridge.set_source(&self.resolved) {
            self.notice(format!("Could not open {}: {error}", self.resolved));
        }
        self.shared.reset_audio();
        self.settings
            .devices
            .entry(self.resolved.clone())
            .or_default();
        self.send_worker_config();
        let description = self
            .shared
            .registry
            .lock()
            .ok()
            .and_then(|registry| registry.source(&self.resolved))
            .map(|device| device.description)
            .unwrap_or_else(|| self.resolved.clone());
        self.notify(
            "Following the system default",
            &format!("Now capturing from {description}."),
            false,
        );
        Ok(())
    }

    /// Start fetching the model, on a thread of its own.
    ///
    /// The daemon is the only thing that downloads: the window asks over the
    /// bus, so there is one place that knows where the files go and one place
    /// that decides whether a download is possible at all.
    fn start_model_download(&mut self) -> Result<(), String> {
        if self.model_download.is_some() {
            return Err("A model download is already running.".into());
        }
        if self.model_present {
            return Ok(());
        }
        if !models::downloadable() {
            return Err(
                "This build has no model download URL configured, so it cannot fetch one.\n\
                 Pass --models DIR or set DFMIC_MODELS to a directory holding the model."
                    .into(),
            );
        }
        if !models::can_download(Some(&self.models_root)) {
            return Err(format!(
                "Cannot write the model to {}.\n\
                 Pass --models DIR or set DFMIC_MODELS to somewhere writable.",
                self.models_root.display()
            ));
        }
        let result = Arc::new(Mutex::new(None));
        let slot = Arc::clone(&result);
        let wake = Arc::clone(&self.shared);
        let root = self.models_root.clone();
        thread::Builder::new()
            .name("dfmic-model".into())
            .spawn(move || {
                let outcome = models::download_default(Some(&root));
                if let Ok(mut slot) = slot.lock() {
                    *slot = Some(outcome);
                }
                // The loop is asleep; tell it there is an outcome to collect.
                wake.wake_loop();
            })
            .map_err(|error| format!("could not start the model download: {error}"))?;
        self.model_download = Some(ModelDownload { result });
        self.sync_state()?;
        Ok(())
    }

    /// Watch for the model appearing on disk, while it is missing.
    ///
    /// Started only in the missing state: once the files are there the daemon has
    /// nothing left to wait for, and a watch that never fires is just a thread.
    fn sync_model_watch(&mut self) {
        if self.model_present {
            self.model_watch = None;
            self.model_watch_root = None;
            return;
        }
        // Watch the deepest directory that exists. Each time one more level
        // appears this moves down, so a model directory created in several steps
        // is followed rather than only noticing its first step.
        let root = modelwatch::watch_root(&self.models_root);
        if self.model_watch_root.as_deref() == Some(root.as_path()) && self.model_watch.is_some() {
            return;
        }
        // Remembered even when the watch fails, so a machine without inotify does
        // not re-create one and re-log the same warning every ``MODEL_POLL`` for
        // as long as the model is missing.
        let attempted =
            self.model_watch_root.as_deref() == Some(root.as_path()) && self.model_watch.is_none();
        if attempted {
            return;
        }
        self.model_watch = None;
        let wake = Arc::clone(&self.shared);
        self.model_watch = modelwatch::ModelWatch::start(&root, move || wake.wake_loop());
        // The root is recorded whether or not the watch started: it is what tells
        // the next tick whether this has already been tried.
        self.model_watch_root = Some(root.clone());
        if self.model_watch.is_none() {
            // No inotify, or nothing watchable yet. The loop keeps its periodic
            // look as a fallback, so this is a lost optimisation, not a failure.
            eprintln!(
                "dfmic: cannot watch {} for a model; falling back to polling",
                root.display()
            );
        }
    }

    /// Notice a finished download, and the model appearing by any other route.
    fn sync_model(&mut self) {
        let finished = self.model_download.as_ref().and_then(ModelDownload::take);
        if let Some(outcome) = finished {
            self.model_download = None;
            match outcome {
                Ok(dir) => {
                    self.model_present = true;
                    self.notify(
                        "Model installed",
                        &format!("Filtering is available. Saved to {}.", dir.display()),
                        false,
                    );
                }
                Err(error) => {
                    self.model_present = models::is_ready(Some(&self.models_root));
                    self.notify("Could not download the model", &error, false);
                }
            }
        }
        // `dfmic models download` run by hand, or a file dropped in place: the
        // daemon picks it up on its own.
        if !self.model_present && models::is_ready(Some(&self.models_root)) {
            self.model_present = true;
        }
    }

    fn send_worker_config(&self) {
        let config = WorkerConfig {
            enabled: self.enabled,
            available: self.source_available,
            agc: self.settings.agc,
            params: self.current_device().effective_params(),
        };
        self.shared
            .min_buffer
            .store(config.params.min_buffer as usize, Ordering::Release);
        if self.worker.send(config).is_err() {
            eprintln!("dfmic: NPU worker control channel closed");
        }
    }

    fn sync_state(&mut self) -> Result<(), String> {
        let next = if !self.model_present {
            // Nothing can be filtered without the model, whatever else is true.
            State::NoModel
        } else if !self.source_available {
            State::NoDevice
        } else if !self.enabled {
            State::Off
        } else if self.shared.listening() && self.bridge.ready() {
            State::Filtering
        } else {
            State::Idle
        };
        self.state = next;
        self.shared
            .filtering
            .store(next == State::Filtering, Ordering::Release);
        // PipeWire sizes the whole graph from the smallest latency any client
        // asks for, so this is only requested while we are actually filtering
        // and is handed back the moment we stop.
        // Capture is needed whenever somebody could hear something: filtering,
        // or the pass-through an app gets while we are on but idle. With nothing
        // listening and the filter off there is no one to hear it, so the stream
        // is paused and the graph costs nothing; the window's own capture keeps
        // the input meter alive while the daemon is silent.
        let capture = match next {
            State::Filtering | State::Idle | State::NoDevice | State::Starting => true,
            State::Off | State::NoModel => self.shared.listening(),
        };
        self.bridge.set_capture_active(capture)?;
        let low_latency = Some(self.settings.graph_quantum && next == State::Filtering);
        if low_latency != self.low_latency {
            self.bridge.set_low_latency(low_latency == Some(true))?;
            self.low_latency = low_latency;
        }
        // The old depth describes a session that has ended. Handing it back when
        // the filter stops is what makes each start begin at the measured floor.
        if next != State::Filtering {
            self.shared.reset_delay();
        }
        self.takeover.sync(
            &self.bridge,
            &self.shared,
            // While filtering, which is what the contract promises: taking the
            // system default away from the real microphone to hand every app a
            // device that is not filtering would only add a hop of latency to
            // nothing. Without a model there is nothing to filter at all.
            self.settings.take_default
                && self.enabled
                && self.source_available
                && self.model_present
                && next == State::Filtering,
        )
    }

    fn sync_monitor(&mut self) -> Result<(), String> {
        let target = self
            .shared
            .registry
            .lock()
            .map_err(|_| "registry lock poisoned".to_string())?
            .monitor_target();
        let desired = if self.settings.monitor && !target.name.is_empty() {
            Some(target.name.clone())
        } else {
            None
        };
        if desired != self.monitor_stream {
            self.bridge
                .set_monitor(desired.is_some(), desired.as_deref())?;
            self.shared.set_monitor_active(desired.is_some());
            self.monitor_stream = desired;
        }
        // Monitoring plays the microphone into the speakers, so it is only ever
        // on for headphones. That was enforced once at startup, which left it
        // playing acoustically after the default output changed to the speakers
        // mid-session. Checked again whenever the sink changes.
        self.last_sink = target.name;
        if self.settings.monitor && !matches!(target.kind.as_str(), "headphones" | "unknown") {
            self.settings.monitor = false;
            self.persist()?;
            self.notices.push(format!(
                "Self-monitor turned off: the default output is now {} ({}).",
                if target.description.is_empty() {
                    "an unknown output"
                } else {
                    &target.description
                },
                target.kind
            ));
            self.bridge.set_monitor(false, None)?;
            self.shared.set_monitor_active(false);
            self.monitor_stream = None;
        }
        Ok(())
    }

    /// Add a note for the window, unless it is already the last one.
    ///
    /// These are produced from inside the event loop, which runs several times a
    /// second while a device is missing: without this the list would grow without
    /// bound and the `NoticeChanged` signal would fire on every tick.
    fn notice(&mut self, message: String) {
        if self.notices.last() != Some(&message) {
            self.notices.push(message);
        }
    }

    fn persist(&self) -> Result<(), String> {
        settings::save(&self.settings)
    }

    fn notify(&self, summary: &str, body: &str, toggle_action: bool) {
        if self.settings.notify {
            self.notifier.report(summary, body, toggle_action);
        }
    }

    fn set_enabled(&mut self, enabled: bool) -> Result<(), String> {
        let changed = self.enabled != enabled;
        self.enabled = enabled;
        self.shared.reset_audio();
        self.send_worker_config();
        self.sync_state()?;
        if changed {
            self.notify(
                if enabled {
                    "Noise suppression on"
                } else {
                    "Noise suppression off"
                },
                if enabled {
                    "DeepFilter Mic is processing speech on the Intel NPU."
                } else {
                    "DeepFilter Mic is passing the raw microphone through unchanged."
                },
                true,
            );
        }
        Ok(())
    }

    fn set_source(&mut self, name: String) -> Result<(), String> {
        if name == SOURCE_DEFAULT {
            self.settings.source = Some(SOURCE_DEFAULT.into());
            self.follow_default_source()?;
            // Asked of the registry, not inferred from a non-empty name: while
            // following the default the resolved name can outlive the device it
            // names, and reporting it as available made `State` say `idle` while
            // `SourceAvailable` said no.
            self.source_available = !self.resolved.is_empty()
                && self
                    .shared
                    .registry
                    .lock()
                    .is_ok_and(|registry| registry.source(&self.resolved).is_some());
            self.settings
                .devices
                .entry(self.resolved.clone())
                .or_default();
            self.shared.reset_audio();
            self.persist()?;
            self.send_worker_config();
            return self.sync_state();
        }
        if name == NODE_NAME {
            return Err("DeepFilter Mic cannot capture from itself.".into());
        }
        let available = self
            .shared
            .registry
            .lock()
            .map_err(|_| "registry lock poisoned".to_string())?
            .source(&name)
            .is_some();
        if !available {
            return Err(format!(
                "No PipeWire Audio/Source named {name:?} is currently available."
            ));
        }
        self.bridge.set_source(&name)?;
        self.settings.source = Some(name.clone());
        self.resolved = name.clone();
        self.settings.devices.entry(name).or_default();
        self.source_available = true;
        self.shared.reset_audio();
        self.persist()?;
        self.send_worker_config();
        self.sync_state()
    }

    fn snapshot(&self) -> Published {
        let is_default = self.shared.is_default();
        let source = self.resolved.clone();
        // Take what the snapshot needs out of the registry and let the guard go
        // before touching PipeWire. Holding it across `Bridge` calls inverts the
        // lock order: the loop thread takes the registry lock from its metadata
        // callback while holding the loop lock, so the two deadlock against each
        // other.
        let (selected, devices, monitor_target) = {
            // Not `expect`: every other registry acquisition in the daemon
            // degrades on a poisoned lock, and a panic here — the function runs
            // on every request and every tick — would take the whole daemon down
            // because a PipeWire callback panicked once.
            match self.shared.registry.lock() {
                Ok(registry) => (
                    registry.source(&source),
                    registry.devices(),
                    registry.monitor_target(),
                ),
                Err(_) => (
                    None,
                    Vec::new(),
                    MonitorTarget {
                        name: String::new(),
                        description: String::new(),
                        kind: "unknown".into(),
                    },
                ),
            }
        };
        let device = self.current_device();
        let listening = self.shared.listening();
        let inference_ms = if self.state == State::Filtering {
            self.shared.inference_ms()
        } else {
            0.0
        };
        let (latency_ms, latency_parts) =
            if listening && self.source_available && self.bridge.ready() {
                if let Some((capture, output)) = self.bridge.latency_ms() {
                    let buffer = (self.shared.input.len() as f64
                        + self
                            .shared
                            .output
                            .len()
                            .max(self.shared.monitor_output.len())
                            as f64)
                        / RATE
                        * 1000.0;
                    let model = if self.enabled { MODEL_DELAY_MS } else { 0.0 };
                    // Deliberately *not* part of the total: inference time is
                    // absorbed by the buffered audio that plays while the NPU
                    // works, so adding it would count the same milliseconds
                    // twice. It stays visible as its own `InferenceMs` property
                    // because it is what forces the buffer to be non-zero.
                    let parts = HashMap::from([
                        ("capture".to_string(), capture),
                        ("buffer".to_string(), buffer),
                        ("model".to_string(), model),
                        ("output".to_string(), output),
                    ]);
                    (capture + buffer + model + output, parts)
                } else {
                    (-1.0, HashMap::new())
                }
            } else {
                (-1.0, HashMap::new())
            };
        Published {
            state: self.state.as_str().into(),
            enabled: self.enabled,
            listening,
            source_description: selected
                .as_ref()
                .map(|device| device.description.clone())
                .unwrap_or_default(),
            source_available: selected.is_some(),
            source_following: self.following(),
            model_present: self.model_present,
            model_path: self.models_root.to_string_lossy().into_owned(),
            model_downloading: self.model_download.is_some(),
            model_download_possible: models::can_download(Some(&self.models_root)),
            source,
            devices: devices
                .into_iter()
                .map(|mut device| {
                    if is_default && self.takeover.previous.as_deref() == Some(device.name.as_str())
                    {
                        device.is_system_default = true;
                    }
                    device
                })
                .collect(),
            compute_device: self.options.device.clone(),
            inference_ms,
            inferences: self.shared.inferences.load(Ordering::Relaxed),
            latency_ms,
            latency_parts,
            overruns: self.shared.overruns.load(Ordering::Relaxed),
            underruns: self.shared.underruns.load(Ordering::Relaxed),
            params: device.params_table().into_iter().collect(),
            soft_mode: device.soft,
            agc: self.settings.agc,
            graph_quantum: self.settings.graph_quantum,
            monitor: self.settings.monitor,
            monitor_target,
            take_default: self.settings.take_default,
            is_default,
            autostart: self.autostart,
            notify: self.settings.notify,
            notices: self.notices.clone(),
            last_failure: self.last_failure.clone(),
        }
    }

    fn handle(&mut self, request: Request) {
        let command = request.command.clone();
        let result: Result<Reply, String> = match command.clone() {
            Command::Toggle => self
                .set_enabled(!self.enabled)
                .map(|()| Reply::Bool(self.enabled)),
            Command::SetEnabled(enabled) => self.set_enabled(enabled).map(|()| Reply::None),
            Command::SetParam(key, value) => (|| {
                let param =
                    Param::from_key(&key).ok_or_else(|| format!("Unknown parameter {key:?}."))?;
                validate_param(param, value)?;
                let source = self.resolved.clone();
                let device = self
                    .settings
                    .devices
                    .get_mut(&source)
                    .ok_or("No capture device is selected.")?;
                if device.soft
                    && SOFT_PRESET
                        .iter()
                        .any(|(overridden, _)| *overridden == param)
                {
                    return Err(format!(
                        "{} is controlled by soft mode. Turn soft mode off to change it.",
                        param.spec().key
                    ));
                }
                device.set(param, value);
                // Push to the worker before writing the file: if the save fails
                // the caller is told so, but the running filter must already be
                // using what the published properties now report.
                self.send_worker_config();
                self.persist()?;
                Ok(Reply::None)
            })(),
            Command::ResetParams => (|| {
                let source = self.resolved.clone();
                self.settings
                    .devices
                    .insert(source, DeviceSettings::default());
                self.send_worker_config();
                self.persist()?;
                Ok(Reply::None)
            })(),
            Command::SetSoftMode(on) => (|| {
                let source = self.resolved.clone();
                self.settings
                    .devices
                    .get_mut(&source)
                    .ok_or("No capture device is selected.")?
                    .soft = on;
                self.send_worker_config();
                self.persist()?;
                Ok(Reply::None)
            })(),
            Command::SetSource(name) => self.set_source(name).map(|()| Reply::None),
            Command::SetAgc(on) => (|| {
                self.settings.agc = on;
                self.send_worker_config();
                self.persist()?;
                Ok(Reply::None)
            })(),
            Command::SetGraphQuantum(on) => (|| {
                self.settings.graph_quantum = on;
                self.persist()?;
                // Takes effect on the next state sync, which also covers the
                // case where filtering is running right now.
                self.low_latency = None;
                self.sync_state()?;
                Ok(Reply::None)
            })(),
            Command::SetMonitor(on) => (|| {
                self.settings.monitor = on;
                self.persist()?;
                self.sync_monitor()?;
                Ok(Reply::None)
            })(),
            Command::SetTakeDefault(on) => (|| {
                self.settings.take_default = on;
                self.persist()?;
                self.sync_state()?;
                Ok(Reply::None)
            })(),
            Command::SetAutostart(on) => (|| {
                service::set_autostart(on)?;
                self.autostart = service::autostart_enabled()?;
                Ok(Reply::None)
            })(),
            Command::SetNotify(on) => (|| {
                self.settings.notify = on;
                self.persist()?;
                Ok(Reply::None)
            })(),
            Command::ResetAll => (|| {
                let current = self
                    .shared
                    .registry
                    .lock()
                    .ok()
                    .and_then(|registry| registry.default_source_name());
                let source = self
                    .takeover
                    .previous
                    .clone()
                    .or(current)
                    .filter(|name| name != NODE_NAME)
                    .unwrap_or_default();
                self.settings = Settings::default();
                self.settings.source = (!source.is_empty()).then_some(source.clone());
                self.notices.clear();
                // A reset is the user asking for the defaults; whatever stopped
                // the takeover earlier is part of what is being reset.
                self.takeover.suppressed = false;
                if !source.is_empty() {
                    self.bridge.set_source(&source)?;
                }
                self.settings.devices.entry(source.clone()).or_default();
                self.source_available = !source.is_empty();
                self.shared.reset_audio();
                // Everything the daemon acts on is applied before the file is
                // written: a reset that cannot be saved is still a reset in
                // effect, and the caller hears about the save separately.
                self.send_worker_config();
                self.persist()?;
                self.sync_monitor()?;
                self.sync_state()?;
                Ok(Reply::None)
            })(),
            Command::DismissNotices => {
                self.notices.clear();
                Ok(Reply::None)
            }
            Command::DownloadModel => self.start_model_download().map(|()| Reply::None),
            // A nudge, not a request: the tick that follows this does the work.
            Command::Wake => Ok(Reply::None),
            Command::ClearFailure => {
                service::clear_last_failure();
                self.last_failure = Failure {
                    message: String::new(),
                    when: 0,
                    phase: String::new(),
                };
                Ok(Reply::None)
            }
        };
        // Publish before replying. `dfmic toggle` and `set_enabled` read the
        // properties back as soon as the call returns, so a reply sent first races
        // the client against the state it just changed.
        self.control.publish(self.snapshot(), false);
        match result {
            Ok(reply) => request.succeed(reply),
            Err(message) => request.fail(message),
        }
    }

    fn event_loop(&mut self, requests: &Receiver<Request>) -> Result<(), String> {
        let mut last_publish = Instant::now() - Duration::from_secs(1);
        let mut last_volatile = Instant::now() - Duration::from_secs(1);
        let mut previous_available = self.source_available;
        while !self.terminate.load(Ordering::Acquire) {
            // Every source of change rings this channel: commands from the bus,
            // PipeWire's registry and metadata callbacks, the worker, a finished
            // download, and the signal handler. So the loop only has to poll
            // while something is in flight, and can otherwise sleep for a long
            // time — the timeout is a safety net, not a cadence.
            let poll = if self.state == State::Starting || self.model_download.is_some() {
                BUSY_POLL
            } else if self.state == State::Filtering {
                ACTIVE_POLL
            } else if !self.model_present && self.model_watch.is_none() {
                MODEL_POLL
            } else {
                STEADY_POLL
            };
            match requests.recv_timeout(poll) {
                Ok(request) => self.handle(request),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("The D-Bus control channel disconnected.".into());
                }
            }
            if let Ok(WorkerEvent::Failed(message)) = self.worker_events.try_recv() {
                return Err(message);
            }
            if let Some(error) = self.shared.error() {
                return Err(format!("PipeWire failed: {error}"));
            }
            if let Some(failure) = self.shared.take_monitor_failure() {
                // A sink that will not open is not a reason to take the
                // microphone down, and retrying it every tick would only churn
                // streams. Turn monitoring off and say why; the user can turn it
                // back on once the output is there.
                self.notices.push(failure);
                if self.settings.monitor {
                    self.settings.monitor = false;
                    self.persist()?;
                    self.sync_monitor()?;
                }
            }
            // The capture stream lost its device. Not fatal, and not something
            // waiting helps with: the stream is left alone here and the
            // availability check below decides, because the registry can still
            // name a node the server has just destroyed. Pointing a stream at one
            // is how "the microphone was unplugged" used to become "the daemon
            // exited".
            if let Some(loss) = self.shared.take_capture_loss() {
                self.notice(format!(
                    "Microphone capture dropped ({loss}); waiting for it to return."
                ));
                self.shared.reset_audio();
            }
            self.follow_default_source()?;
            let source = self.resolved.clone();
            self.source_available = !source.is_empty()
                && self
                    .shared
                    .registry
                    .lock()
                    .is_ok_and(|registry| registry.source(&source).is_some());
            if self.source_available != previous_available {
                if self.source_available {
                    if let Err(error) = self.bridge.set_source(&source) {
                        self.notice(format!("Could not open {source}: {error}"));
                    }
                    self.shared.reset_audio();
                    self.notify(
                        "Microphone reconnected",
                        &format!(
                            "{source} is available again; DeepFilter Mic resumed automatically."
                        ),
                        false,
                    );
                } else {
                    self.notify(
                        "Microphone disconnected",
                        &format!("{source} disappeared. DeepFilter Mic is silent and will resume when it returns."),
                        false,
                    );
                }
                previous_available = self.source_available;
                self.send_worker_config();
                self.sync_state()?;
            }
            self.sync_model();
            self.sync_model_watch();
            self.sync_monitor()?;
            self.sync_state()?;
            // Diagnostics are properties, so they only reach a client when this
            // runs. The loop's timeout is the cadence: `ACTIVE_POLL` while
            // filtering, because nothing else wakes it and a half-minute gap made
            // the window's inference and overrun counters look frozen.
            if last_publish.elapsed() >= Duration::from_millis(100) {
                self.control.publish(
                    self.snapshot(),
                    last_volatile.elapsed() >= Duration::from_millis(500),
                );
                if last_volatile.elapsed() >= Duration::from_millis(500) {
                    last_volatile = Instant::now();
                }
                last_publish = Instant::now();
            }
        }
        Ok(())
    }

    fn shutdown(&mut self) {
        let enabled = self.enabled;
        self.enabled = false;
        if let Err(error) = self.sync_state() {
            eprintln!("dfmic: could not restore the previous default source: {error}");
        }
        self.enabled = enabled;
        if let Err(error) = self.bridge.set_monitor(false, None) {
            eprintln!("dfmic: could not stop self-monitor: {error}");
        }
        self.shared.set_monitor_active(false);
        self.worker_stop.store(true, Ordering::Release);
        // It may be in its long idle sleep; without this the join below waits
        // for that timeout to expire.
        self.shared.wake_worker();
        if let Some(thread) = self.worker_thread.take() {
            if thread.join().is_err() {
                eprintln!("dfmic: NPU worker panicked during shutdown");
            }
        }
    }
}

fn validate_param(param: Param, value: f64) -> Result<(), String> {
    let spec = param.spec();
    if !value.is_finite() || value < spec.min || value > spec.max {
        return Err(format!(
            "{} must be between {} and {}; got {}.",
            spec.key, spec.min, spec.max, value
        ));
    }
    if matches!(param, Param::MinBuffer | Param::MaskReduce) && value.fract() != 0.0 {
        return Err(format!("{} must be a whole number.", spec.key));
    }
    if param == Param::MaskReduce && value > 2.0 {
        return Err("mask-reduce must be 0 (none), 1 (max) or 2 (mean).".into());
    }
    Ok(())
}

pub fn run(options: &Options) -> Result<(), String> {
    // Take the bus name before anything with a side effect on the running
    // system — restoring the previous default source, creating PipeWire streams,
    // loading the NPU — because a second instance doing any of them would
    // sabotage the first one before it discovered that it is not allowed to
    // exist. The interface is exported first so that the name only becomes
    // visible once it can be answered.
    let shared = Shared::new();
    let (sender, requests) = control::channel();
    // Every asynchronous source of change nudges this: PipeWire's callbacks (via
    // `Shared::wake_loop`), the worker, the download thread and the signals.
    shared.set_wake_sender(sender.clone());
    let control = ControlServer::connect()?
        .start(
            sender.clone(),
            Arc::clone(&shared),
            Published::initial(options.device.clone()),
        )?
        .claim()?;
    let restored = match recover_stale_default() {
        Ok(restored) => restored,
        Err(error) => return startup_failure(error),
    };
    let (mut daemon, requests) = match Daemon::new(
        options.clone_for_startup(),
        restored,
        control,
        shared,
        sender,
        requests,
    ) {
        Ok(daemon) => daemon,
        Err(error) => return startup_failure(error),
    };
    let result = daemon.event_loop(&requests);
    let notify = daemon.settings.notify;
    daemon.shutdown();
    match result {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = recover_stale_default();
            let failure = service::record_failure(error.clone(), "stream");
            if notify {
                notifications::standalone_failure(&failure.message);
            }
            Err(error)
        }
    }
}

impl Options {
    fn clone_for_startup(&self) -> Self {
        Self {
            models: self.models.clone(),
            device: self.device.clone(),
            source: self.source.clone(),
            take_default: self.take_default,
            param_overrides: self.param_overrides.clone(),
        }
    }
}

fn startup_failure(error: String) -> Result<(), String> {
    let _ = recover_stale_default();
    let failure = service::record_failure(error.clone(), "startup");
    if settings::load().is_ok_and(|(settings, _)| settings.notify) {
        notifications::standalone_failure(&failure.message);
    }
    Err(error)
}
