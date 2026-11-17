//! The virtual microphone daemon.
//!
//! Audio path: physical microphone -> PipeWire capture stream -> lock-free ring
//! -> worker thread (NPU) -> lock-free ring -> exported `Audio/Source` node.
//!
//! The realtime callbacks never block on the NPU; if the worker falls behind,
//! samples are dropped and counted rather than stalling the audio graph.
//!
//! Failure is loud on purpose: if the NPU or the model is unusable the daemon
//! exits non-zero without touching the system default source, and if it dies
//! mid-stream it restores the previous default first.

use std::path::PathBuf;
use std::process::Command as Process;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dfn_dsp::{DfnPipeline, Params, HOP_SIZE};
use dfn_npu::NpuRuntime;

use crate::control::{Command as BusCommand, Control, Status};
use crate::pw::{Bridge, Shared};

pub const NODE_NAME: &str = "dfmic-source";
pub const NODE_DESCRIPTION: &str = "DeepFilter Mic";

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_signal: i32) {
    STOP.store(true, Ordering::SeqCst);
}

extern "C" {
    fn signal(signum: i32, handler: usize) -> usize;
}

const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;

pub struct Options {
    pub models: PathBuf,
    pub device: String,
    pub source: Option<String>,
    pub take_default: bool,
    pub params: Params,
}

/// Directory for the status file and the saved default source.
fn state_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("DFMIC_STATE_DIR") {
        return PathBuf::from(dir);
    }
    let base = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(base).join("dfmic")
}

fn write_default_backup(name: &str) {
    let dir = state_dir();
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join("previous-default"), name);
}

fn read_default_backup() -> Option<String> {
    std::fs::read_to_string(state_dir().join("previous-default"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn clear_default_backup() {
    let _ = std::fs::remove_file(state_dir().join("previous-default"));
}

/// Ask WirePlumber which source is the default, by name.
fn current_default_source() -> Option<String> {
    let out = Process::new("pw-metadata").arg("-n").arg("default").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if let Some(rest) = line.split("key:'default.audio.source'").nth(1) {
            let value = rest.split("value:'").nth(1)?;
            let json = value.split('\'').next()?;
            let name = json.split("\"name\":\"").nth(1)?.split('"').next()?;
            return Some(name.to_string());
        }
    }
    None
}

fn set_default_source(name: &str) -> Result<(), String> {
    let payload = format!("{{\"name\":\"{name}\"}}");
    let status = Process::new("pw-metadata")
        .args([
            "-n",
            "default",
            "0",
            "default.audio.source",
            &payload,
            "Spa:String:JSON",
        ])
        .status()
        .map_err(|e| format!("running pw-metadata: {e}"))?;
    if !status.success() {
        return Err("pw-metadata failed to set the default source".into());
    }
    Ok(())
}

/// Tell the world what we are doing, so `dfmic status` and the GUI can read it.
fn write_status(fields: &[(&str, String)]) {
    let dir = state_dir();
    let _ = std::fs::create_dir_all(&dir);
    let mut text = String::new();
    for (key, value) in fields {
        text.push_str(&format!("{key}={value}\n"));
    }
    let _ = std::fs::write(dir.join("status"), text);
}

fn clear_status() {
    let _ = std::fs::remove_file(state_dir().join("status"));
}

pub fn run(opts: &Options) -> Result<(), String> {
    unsafe {
        let handler = on_signal as extern "C" fn(i32) as usize;
        signal(SIGINT, handler);
        signal(SIGTERM, handler);
    }

    // A previous run that died without cleaning up leaves a stale backup; put it
    // back before we start so we never lose the user's real microphone.
    if let Some(previous) = read_default_backup() {
        eprintln!("dfmic: restoring default source left behind by a previous run: {previous}");
        let _ = set_default_source(&previous);
        clear_default_backup();
    }

    let started = Instant::now();
    // OpenVINO refuses to use a cache directory that does not exist yet, so make
    // it first: a fresh install must work on its very first run.
    let cache = state_dir().join("npu-cache");
    std::fs::create_dir_all(&cache)
        .map_err(|e| format!("cannot create {}: {e}", cache.display()))?;
    let runtime = NpuRuntime::load(&opts.models, &opts.device, Some(&cache))
        .map_err(|e| format!("cannot use the NPU: {e}"))?;
    let mut pipeline = DfnPipeline::new(runtime, opts.params.clone());
    eprintln!(
        "dfmic: models compiled on {} in {:.2}s",
        opts.device,
        started.elapsed().as_secs_f64()
    );

    // Pin the capture stream to a physical device *before* we take over the
    // default source. Following the default would make us capture our own
    // output once the takeover lands, which is silence plus a feedback loop.
    let capture_target = match opts.source.clone().or_else(current_default_source) {
        Some(target) if target != NODE_NAME => target,
        Some(_) => {
            return Err(
                "the default source is already the dfmic node; pass --source with a real device"
                    .into(),
            )
        }
        None => {
            return Err("cannot determine which microphone to capture; pass --source <node name>"
                .into())
        }
    };
    eprintln!("dfmic: capturing from {capture_target}");

    let shared = Shared::new(HOP_SIZE * 64);
    let bridge = Bridge::start(
        NODE_NAME,
        NODE_DESCRIPTION,
        Some(&capture_target),
        Arc::clone(&shared),
    )
    .map_err(|e| format!("cannot create the PipeWire streams: {e}"))?;

    // Wait for both streams to reach STREAMING before touching the default.
    let deadline = Instant::now() + Duration::from_secs(5);
    while !bridge.ready() {
        if shared.failed.load(Ordering::SeqCst) {
            let message = shared
                .error
                .lock()
                .ok()
                .and_then(|m| m.clone())
                .unwrap_or_else(|| "unknown PipeWire error".into());
            return Err(message);
        }
        if Instant::now() > deadline {
            return Err("PipeWire streams did not start within 5s".into());
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    if opts.take_default {
        if let Some(previous) = current_default_source() {
            write_default_backup(&previous);
        }
        set_default_source(NODE_NAME)?;
        eprintln!("dfmic: '{NODE_DESCRIPTION}' is now the default source");
    }

    // One writer only: the worker owns the status file so the static fields and
    // the live counters cannot clobber each other.
    let base: Vec<(&'static str, String)> = vec![
        ("node", NODE_NAME.to_string()),
        ("source", capture_target.clone()),
        ("device", opts.device.clone()),
        ("default", opts.take_default.to_string()),
        ("pid", std::process::id().to_string()),
    ];
    write_status(&base);

    // The control interface: the GUI and `dfmic toggle` drive the daemon here.
    let (tx, rx) = std::sync::mpsc::channel::<BusCommand>();
    let control = match Control::start(&opts.device, &capture_target, tx) {
        Ok(c) => Some(c),
        Err(e) => {
            // Without D-Bus the microphone still works; say so and carry on.
            eprintln!("dfmic: warning: control interface unavailable: {e}");
            None
        }
    };
    let enabled = Arc::new(AtomicBool::new(true));
    // Live parameters travel as raw f32 bits so the audio thread never takes a
    // lock: the GUI writes, the worker applies at the top of each block.
    let live = Arc::new(LiveParams::new(&opts.params));
    control.as_ref().map(|c| {
        c.update(&Status {
            enabled: true,
            ..Default::default()
        })
    });

    let worker = {
        let shared = Arc::clone(&shared);
        let enabled = Arc::clone(&enabled);
        let live = Arc::clone(&live);
        let params_base = opts.params.clone();
        std::thread::spawn(move || {
            worker_loop(&mut pipeline, shared, base, enabled, live, params_base)
        })
    };

    // Run until told to stop, until PipeWire reports a failure, or until the bus
    // asks us to quit.
    let mut last_in = 0.0f64;
    let mut last_out = 0.0f64;
    while !STOP.load(Ordering::SeqCst) && !shared.failed.load(Ordering::SeqCst) {
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(BusCommand::Toggle) => {
                let now = !enabled.load(Ordering::SeqCst);
                enabled.store(now, Ordering::SeqCst);
                if std::env::var("DFMIC_DEBUG").is_ok() {
                    eprintln!("dfmic: toggle -> {now}");
                }
            }
            Ok(BusCommand::Enable) => enabled.store(true, Ordering::SeqCst),
            Ok(BusCommand::Disable) => enabled.store(false, Ordering::SeqCst),
            Ok(BusCommand::SetParam(name, value)) => match name.as_str() {
                "atten-lim" => live.atten_lim.store((value as f32).to_bits(), Ordering::Relaxed),
                "post-filter" => {
                    live.post_filter.store((value as f32).to_bits(), Ordering::Relaxed)
                }
                _ => eprintln!("dfmic: ignoring unknown parameter {name:?}"),
            },
            Ok(BusCommand::Shutdown) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {}
        }

        if let Some(control) = control.as_ref() {
            // Peak decay: jump up instantly, fall smoothly, like a real meter.
            let in_peak = shared.peak.swap(0, Ordering::Relaxed) as f64 / 1e6;
            last_in = if in_peak > last_in { in_peak } else { last_in * 0.85 };
            let out_peak = shared.output_peak.swap(0, Ordering::Relaxed) as f64 / 1e6;
            last_out = if out_peak > last_out { out_peak } else { last_out * 0.85 };
            control.update(&Status {
                enabled: enabled.load(Ordering::SeqCst),
                atten_lim: live.atten_lim() as f64,
                post_filter: live.post_filter() as f64,
                level: last_in,
                out_level: last_out,
                latency_ms: 10.0,
                inferences: shared.played.load(Ordering::Relaxed) as u32 / HOP_SIZE as u32,
                dropped: shared.dropped.load(Ordering::Relaxed) as u32 / HOP_SIZE as u32,
                error: String::new(),
            });
            control.set_params(live.atten_lim() as f64, live.post_filter() as f64);
            control.notify_changed();
        }
    }

    let failure = shared.failed.load(Ordering::SeqCst).then(|| {
        shared
            .error
            .lock()
            .ok()
            .and_then(|m| m.clone())
            .unwrap_or_else(|| "unknown error".into())
    });

    // Restore before tearing anything down, so the real microphone comes back.
    if opts.take_default {
        if let Some(previous) = read_default_backup() {
            if let Err(e) = set_default_source(&previous) {
                eprintln!("dfmic: could not restore the default source: {e}");
            } else {
                eprintln!("dfmic: restored default source to {previous}");
            }
            clear_default_backup();
        }
    }
    clear_status();
    drop(bridge);
    let _ = worker.join();

    match failure {
        Some(message) => Err(format!("stream failed: {message}")),
        None => Ok(()),
    }
}

/// Parameters the GUI can change while the audio thread runs.
struct LiveParams {
    atten_lim: AtomicU32,
    post_filter: AtomicU32,
}

impl LiveParams {
    fn new(params: &Params) -> Self {
        Self {
            atten_lim: AtomicU32::new(params.atten_lim_db.to_bits()),
            post_filter: AtomicU32::new(params.post_filter_beta.to_bits()),
        }
    }

    fn atten_lim(&self) -> f32 {
        f32::from_bits(self.atten_lim.load(Ordering::Relaxed))
    }

    fn post_filter(&self) -> f32 {
        f32::from_bits(self.post_filter.load(Ordering::Relaxed))
    }
}

fn worker_loop(
    pipeline: &mut DfnPipeline<NpuRuntime>,
    shared: Arc<Shared>,
    base: Vec<(&'static str, String)>,
    enabled: Arc<AtomicBool>,
    live: Arc<LiveParams>,
    params_base: Params,
) {
    let mut block = vec![0.0f32; HOP_SIZE];
    let mut enhanced = vec![0.0f32; HOP_SIZE];
    let mut last_status = Instant::now();

    while !STOP.load(Ordering::SeqCst) && !shared.failed.load(Ordering::SeqCst) {
        // Disabled, or nothing is recording from us: drop the audio and leave the
        // NPU idle rather than filtering into a void.
        if !enabled.load(Ordering::SeqCst) || !shared.consumers.load(Ordering::SeqCst) {
            let mut sink = vec![0.0f32; 4096];
            while shared.input.pop(&mut sink) > 0 {}
            std::thread::sleep(Duration::from_millis(20));
            continue;
        }
        if shared.input.len() < HOP_SIZE {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        if shared.input.pop(&mut block) < HOP_SIZE {
            continue;
        }
        // Apply any live parameter changes before filtering this block.
        let mut params = params_base.clone();
        params.atten_lim_db = live.atten_lim();
        params.post_filter_beta = live.post_filter();
        *pipeline.params_mut() = params;

        pipeline.process(&block, &mut enhanced);
        if !shared.output.push(&enhanced) {
            shared.dropped.fetch_add(HOP_SIZE as u64, Ordering::Relaxed);
        }

        if last_status.elapsed() > Duration::from_secs(5) {
            last_status = Instant::now();
            let mut fields = base.clone();
            fields.extend([
                ("captured", shared.captured.load(Ordering::Relaxed).to_string()),
                ("played", shared.played.load(Ordering::Relaxed).to_string()),
                ("dropped", shared.dropped.load(Ordering::Relaxed).to_string()),
                ("queued", shared.input.len().to_string()),
                ("inferences", pipeline.runtime_mut().infer_count().to_string()),
                ("peak", format!("{:.4}", shared.peak.load(Ordering::Relaxed) as f64 / 1e6)),
            ]);
            write_status(&fields);
        }
    }
}

/// Toggle the filter in a running daemon, from the command line (keybinds,
/// notification buttons, scripts).
pub fn toggle() -> Result<(), String> {
    let out = Process::new("gdbus")
        .args([
            "call",
            "--session",
            "--dest",
            "org.dfmic.Control",
            "--object-path",
            "/org/dfmic/Control",
            "--method",
            "org.dfmic.Control.Toggle",
        ])
        .output()
        .map_err(|e| format!("running gdbus: {e}"))?;
    if !out.status.success() {
        return Err("no daemon is listening on org.dfmic.Control".into());
    }
    let state = String::from_utf8_lossy(&out.stdout);
    let on = state.contains("true");
    println!("{}", if on { "on" } else { "off" });
    Ok(())
}

pub fn status() -> Result<(), String> {
    let dir = state_dir();
    let status = std::fs::read_to_string(dir.join("status"))
        .map_err(|_| "no daemon running (no status file)".to_string())?;
    let mut pid = None;
    for line in status.lines() {
        let (key, value) = line.split_once('=').unwrap_or((line, ""));
        if key == "pid" {
            pid = value.parse::<u32>().ok();
        }
        println!("{key}: {value}");
    }
    match pid {
        Some(pid) if std::path::Path::new(&format!("/proc/{pid}")).exists() => {
            println!("running: yes");
            Ok(())
        }
        _ => Err("status file is stale (daemon not running)".into()),
    }
}
