//! The virtual microphone daemon and its live audio worker.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command as ProcessCommand;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use dfn_dsp::{DfnPipeline, Params, HOP_SIZE};
use dfn_npu::NpuRuntime;

use crate::control::{self, Command, ControlServer, Published, Reply, Request};
use crate::notifications::{self, Notifier};
use crate::proto::{
    Failure, MonitorTarget, Param, State, NODE_DESCRIPTION, NODE_NAME, SOFT_PRESET, SOURCE_DEFAULT,
};
use crate::pw::{Bridge, Shared};
use crate::service;
use crate::settings::{self, DeviceSettings, Settings};

const RATE: f64 = 48_000.0;
const MODEL_DELAY_MS: f64 = HOP_SIZE as f64 / RATE * 1000.0;

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

struct Agc {
    gain_db: f32,
    limiter_gain: f32,
    previous: f32,
}

impl Agc {
    fn new() -> Self {
        Self {
            gain_db: 0.0,
            limiter_gain: 1.0,
            previous: 0.0,
        }
    }

    fn process(
        &mut self,
        samples: &mut [f32],
        lsnr: f32,
        enabled: bool,
        input_rms: f32,
    ) -> (f32, bool) {
        let voice = enabled && lsnr >= 0.0 && input_rms >= 10f32.powf(-50.0 / 20.0);
        if enabled && voice && input_rms > 1e-7 {
            let speech_target = 10f32.powf(-18.0 / 20.0);
            let desired = (20.0 * (speech_target / input_rms).log10()).clamp(-12.0, 18.0);
            // Roughly 0.5 s attack and 2 s release at one hop every 10 ms.
            let coefficient = if desired > self.gain_db { 0.02 } else { 0.005 };
            self.gain_db += (desired - self.gain_db) * coefficient;
        }
        let gain_db = if enabled { self.gain_db } else { 0.0 };
        if !enabled {
            self.gain_db = 0.0;
        }
        let gain = 10f32.powf(gain_db / 20.0);

        // Causal four-times-oversampled peak detection. The detector has 2 dB
        // of headroom for intersample overshoot and the final clamp enforces
        // the -1 dBFS sample ceiling even for an abrupt full-scale transition.
        let ceiling = 10f32.powf(-1.0 / 20.0);
        let detector_ceiling = 10f32.powf(-3.0 / 20.0);
        let mut previous = self.previous;
        let mut limiter_gain = self.limiter_gain;
        for sample in samples {
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
        (gain_db, voice)
    }
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
    mut pipeline: DfnPipeline<NpuRuntime>,
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
    let mut input = vec![0.0f32; HOP_SIZE];
    let mut output = vec![0.0f32; HOP_SIZE];
    let mut agc = Agc::new();
    let mut reset_generation = 0;
    while !stop.load(Ordering::Acquire) {
        while let Ok(next) = config_rx.try_recv() {
            shared
                .min_buffer
                .store(next.params.min_buffer as usize, Ordering::Release);
            *pipeline.params_mut() = next.params.clone();
            config = next;
        }
        let next_reset = shared.reset_generation();
        if next_reset != reset_generation {
            shared.input.clear();
            shared.output.clear();
            shared.monitor_output.clear();
            agc = Agc::new();
            reset_generation = next_reset;
        }
        let listening = shared.listening();
        let should_process = listening && config.enabled && config.available;
        let should_passthrough = listening && !config.enabled && config.available;
        if !config.available || !listening {
            while shared.input.len() > 0 {
                shared.input.pop(&mut input);
            }
            shared.output.clear();
            shared.monitor_output.clear();
            shared.set_agc(0.0, false);
            thread::sleep(Duration::from_millis(2));
            continue;
        }
        if shared.input.len() < HOP_SIZE {
            // The capture callback wakes us the moment a hop is complete, so
            // this returns within microseconds of the data arriving rather than
            // on the next poll tick.
            shared.wait_for_hop();
            continue;
        }
        shared.input.pop(&mut input);
        if should_passthrough {
            output.copy_from_slice(&input);
            shared.set_agc(0.0, false);
        } else if should_process {
            let input_rms = (input.iter().map(|sample| sample * sample).sum::<f32>()
                / input.len() as f32)
                .sqrt();
            let started = Instant::now();
            let lsnr = pipeline.process(&input, &mut output);
            shared.record_inference(started.elapsed().as_secs_f64());
            let (gain, voice) = agc.process(&mut output, lsnr, config.agc, input_rms);
            shared.set_agc(gain, voice);
        } else {
            continue;
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
    params: Params,
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
) -> Result<WorkerHandle, String> {
    let runtime = NpuRuntime::load(&models, &device, None)?;
    let pipeline = DfnPipeline::new(runtime, params);
    let (config_tx, config_rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let thread = thread::Builder::new()
        .name("dfmic-npu".into())
        .spawn(move || {
            promote_to_realtime();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                worker_loop(pipeline, shared, config_rx, stop);
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
}

impl Takeover {
    fn sync(&mut self, bridge: &Bridge, shared: &Shared, should_take: bool) -> Result<(), String> {
        let current = shared
            .registry
            .lock()
            .ok()
            .and_then(|registry| registry.default_source.clone());
        if self.active && current.as_deref() != Some(NODE_NAME) {
            // A user-selected default wins over our later clean-up and is not
            // immediately overwritten by the next 50 ms state tick.
            self.active = false;
            self.previous = None;
            remove_backup();
            return Ok(());
        }
        if !should_take {
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
        if self.active || current.as_deref() == Some(NODE_NAME) {
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
}

impl Daemon {
    fn new(
        options: Options,
        restored: Option<String>,
    ) -> Result<(Self, Receiver<Request>), String> {
        let (mut settings, mut notices) = settings::load()?;
        if let Some(source) = &options.source {
            settings.source = Some(source.clone());
        }
        if let Some(take_default) = options.take_default {
            settings.take_default = take_default;
        }
        let initial_source = settings.source.clone().unwrap_or_default();
        let following = initial_source == SOURCE_DEFAULT;

        let shared = Shared::new(48_000 / 2);
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
            device_settings.effective_params(),
            Arc::clone(&shared),
            Arc::clone(&worker_stop),
        )?;
        let (sender, receiver) = control::channel();
        let mut published = Published::initial(options.device.clone());
        published.last_failure = service::last_failure().unwrap_or(Failure {
            message: String::new(),
            when: 0,
            phase: String::new(),
        });
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
        published.autostart = autostart;
        let control = ControlServer::start(sender.clone(), Arc::clone(&shared), published)?;
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
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            signal_hook::flag::register(signal, Arc::clone(&terminate))
                .map_err(|error| format!("could not install termination handler: {error}"))?;
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
        };
        daemon.send_worker_config();
        daemon.sync_monitor()?;
        daemon.sync_state()?;
        Ok((daemon, receiver))
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
        self.bridge.set_source(&self.resolved)?;
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

    fn send_worker_config(&self) {
        let config = WorkerConfig {
            enabled: self.settings.enabled,
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
        let next = if !self.source_available {
            State::NoDevice
        } else if !self.settings.enabled {
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
        let low_latency = Some(self.settings.graph_quantum && next == State::Filtering);
        if low_latency != self.low_latency {
            self.bridge.set_low_latency(low_latency == Some(true))?;
            self.low_latency = low_latency;
        }
        self.takeover.sync(
            &self.bridge,
            &self.shared,
            self.settings.take_default && self.settings.enabled && self.source_available,
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
        if target.name != self.last_sink {
            self.last_sink = target.name;
        }
        Ok(())
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
        let changed = self.settings.enabled != enabled;
        self.settings.enabled = enabled;
        self.persist()?;
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
            self.source_available = !self.resolved.is_empty();
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
            let registry = self.shared.registry.lock().expect("registry lock poisoned");
            (
                registry.source(&source),
                registry.devices(),
                registry.monitor_target(),
            )
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
                    let model = if self.settings.enabled {
                        MODEL_DELAY_MS
                    } else {
                        0.0
                    };
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
            enabled: self.settings.enabled,
            listening,
            source_description: selected
                .as_ref()
                .map(|device| device.description.clone())
                .unwrap_or_default(),
            source_available: selected.is_some(),
            source_following: self.following(),
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
                .set_enabled(!self.settings.enabled)
                .map(|()| Reply::Bool(self.settings.enabled)),
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
                self.persist()?;
                self.send_worker_config();
                Ok(Reply::None)
            })(),
            Command::ResetParams => (|| {
                let source = self.resolved.clone();
                self.settings
                    .devices
                    .insert(source, DeviceSettings::default());
                self.persist()?;
                self.send_worker_config();
                Ok(Reply::None)
            })(),
            Command::SetSoftMode(on) => (|| {
                let source = self.resolved.clone();
                self.settings
                    .devices
                    .get_mut(&source)
                    .ok_or("No capture device is selected.")?
                    .soft = on;
                self.persist()?;
                self.send_worker_config();
                Ok(Reply::None)
            })(),
            Command::SetSource(name) => self.set_source(name).map(|()| Reply::None),
            Command::SetAgc(on) => (|| {
                self.settings.agc = on;
                self.persist()?;
                self.send_worker_config();
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
                if !source.is_empty() {
                    self.bridge.set_source(&source)?;
                }
                self.settings.devices.entry(source.clone()).or_default();
                self.source_available = !source.is_empty();
                self.shared.reset_audio();
                self.persist()?;
                self.send_worker_config();
                self.sync_monitor()?;
                self.sync_state()?;
                Ok(Reply::None)
            })(),
            Command::DismissNotices => {
                self.notices.clear();
                Ok(Reply::None)
            }
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
        match result {
            Ok(reply) => request.succeed(reply),
            Err(message) => request.fail(message),
        }
        self.control.publish(self.snapshot(), false);
    }

    fn event_loop(&mut self, requests: &Receiver<Request>) -> Result<(), String> {
        let mut last_publish = Instant::now() - Duration::from_secs(1);
        let mut last_volatile = Instant::now() - Duration::from_secs(1);
        let mut previous_available = self.source_available;
        while !self.terminate.load(Ordering::Acquire) {
            match requests.recv_timeout(Duration::from_millis(50)) {
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
                    self.bridge.set_source(&source)?;
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
            self.sync_monitor()?;
            self.sync_state()?;
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
        let enabled = self.settings.enabled;
        self.settings.enabled = false;
        if let Err(error) = self.sync_state() {
            eprintln!("dfmic: could not restore the previous default source: {error}");
        }
        self.settings.enabled = enabled;
        if let Err(error) = self.bridge.set_monitor(false, None) {
            eprintln!("dfmic: could not stop self-monitor: {error}");
        }
        self.shared.set_monitor_active(false);
        self.worker_stop.store(true, Ordering::Release);
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
    let restored = match recover_stale_default() {
        Ok(restored) => restored,
        Err(error) => return startup_failure(error),
    };
    let (mut daemon, requests) = match Daemon::new(options.clone_for_startup(), restored) {
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
