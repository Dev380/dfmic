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
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dfn_dsp::{DfnPipeline, Params, HOP_SIZE};
use dfn_npu::NpuRuntime;

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
    pub self_monitor: bool,
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
    let out = Command::new("pw-metadata").arg("-n").arg("default").output().ok()?;
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
    let status = Command::new("pw-metadata")
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
        signal(SIGINT, on_signal as usize);
        signal(SIGTERM, on_signal as usize);
    }

    // A previous run that died without cleaning up leaves a stale backup; put it
    // back before we start so we never lose the user's real microphone.
    if let Some(previous) = read_default_backup() {
        eprintln!("dfmic: restoring default source left behind by a previous run: {previous}");
        let _ = set_default_source(&previous);
        clear_default_backup();
    }

    let started = Instant::now();
    let runtime = NpuRuntime::load(&opts.models, &opts.device, Some(&state_dir().join("npu-cache")))
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

    let worker = {
        let shared = Arc::clone(&shared);
        std::thread::spawn(move || worker_loop(&mut pipeline, shared, base))
    };

    // Run until told to stop, or until PipeWire reports a failure.
    while !STOP.load(Ordering::SeqCst) && !shared.failed.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(50));
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

fn worker_loop(
    pipeline: &mut DfnPipeline<NpuRuntime>,
    shared: Arc<Shared>,
    base: Vec<(&'static str, String)>,
) {
    let mut block = vec![0.0f32; HOP_SIZE];
    let mut enhanced = vec![0.0f32; HOP_SIZE];
    let mut last_status = Instant::now();

    while !STOP.load(Ordering::SeqCst) && !shared.failed.load(Ordering::SeqCst) {
        // Nothing is recording from us: drop the audio and leave the NPU idle
        // rather than filtering into a void.
        if !shared.consumers.load(Ordering::SeqCst) {
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
