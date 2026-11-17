//! The GUI: a Slint window that drives the daemon over D-Bus.
//!
//! The window is deliberately small in scope — it is a remote control and a
//! meter, not an audio engine. Everything it shows comes from the daemon's
//! `org.dfmic.Control` object, and every control it offers is a method call.

use std::process::Command;
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

use slint::{ComponentHandle, Timer, TimerMode};

slint::include_modules!();

/// One poll of the daemon's properties.
#[derive(Debug, Default, Clone)]
struct Snapshot {
    connected: bool,
    enabled: bool,
    level: f64,
    device: String,
    source: String,
    latency_ms: f64,
    inferences: u32,
    dropped: u32,
    error: String,
    atten_lim: f64,
    post_filter: f64,
    out_level: f64,
}

/// Read every property in one `GetAll` call.
fn query() -> Snapshot {
    let out = Command::new("gdbus")
        .args([
            "call",
            "--session",
            "--dest",
            "org.dfmic.Control",
            "--object-path",
            "/org/dfmic/Control",
            "--method",
            "org.freedesktop.DBus.Properties.GetAll",
            "org.dfmic.Control",
        ])
        .output();

    let Ok(out) = out else {
        return Snapshot::default();
    };
    if !out.status.success() {
        return Snapshot::default();
    }
    let text = String::from_utf8_lossy(&out.stdout);
    parse_snapshot(&text)
}

/// `gdbus` prints GVariant text like `({'Enabled': <true>, 'Level': <0.01>, ...})`.
fn parse_snapshot(text: &str) -> Snapshot {
    let mut snap = Snapshot {
        connected: true,
        ..Default::default()
    };
    let field = |name: &str| -> Option<String> {
        let key = format!("'{name}': <");
        let start = text.find(&key)? + key.len();
        let rest = &text[start..];
        let end = rest.find('>')?;
        Some(rest[..end].trim().to_string())
    };

    if let Some(v) = field("Enabled") {
        snap.enabled = v.eq_ignore_ascii_case("true");
    }
    if let Some(v) = field("Level") {
        snap.level = v.parse().unwrap_or(0.0);
    }
    if let Some(v) = field("OutLevel") {
        snap.out_level = v.parse().unwrap_or(0.0);
    }
    if let Some(v) = field("Device") {
        snap.device = v.trim_matches('\'').to_string();
    }
    if let Some(v) = field("Source") {
        snap.source = v.trim_matches('\'').to_string();
    }
    if let Some(v) = field("LatencyMs") {
        snap.latency_ms = v.parse().unwrap_or(0.0);
    }
    if let Some(v) = field("Inferences") {
        snap.inferences = v.split_whitespace().last().unwrap_or("0").parse().unwrap_or(0);
    }
    if let Some(v) = field("Dropped") {
        snap.dropped = v.split_whitespace().last().unwrap_or("0").parse().unwrap_or(0);
    }
    if let Some(v) = field("Error") {
        snap.error = v.trim_matches('\'').to_string();
    }
    if let Some(v) = field("AttenLim") {
        snap.atten_lim = v.parse().unwrap_or(100.0);
    }
    if let Some(v) = field("PostFilter") {
        snap.post_filter = v.parse().unwrap_or(0.0);
    }
    snap
}

/// Turn a PipeWire node name into something a human wants to read.
///
/// `alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Mic1__source`
/// becomes `Mic1`, and `...HiFi__Speaker__sink` would become `Speaker`.
fn friendly(name: &str) -> (String, String) {
    let bus = if name.starts_with("alsa_input") {
        "internal"
    } else if name.starts_with("bluez_input") {
        "bluetooth"
    } else if name.starts_with("alsa_output") {
        "loopback"
    } else {
        "pipewire"
    };

    // PipeWire encodes the UCM verb and device between double underscores.
    if let Some(rest) = name.rsplit("__").nth(1).or_else(|| name.rsplit('.').next()) {
        let cleaned = rest
            .trim_end_matches("_source")
            .trim_end_matches("_sink")
            .replace('_', " ");
        if !cleaned.is_empty() {
            return (cleaned, bus.to_string());
        }
    }
    (name.to_string(), bus.to_string())
}

fn call_method(method: &str, extra: &[&str]) -> bool {
    let out = Command::new("gdbus")
        .args([
            "call",
            "--session",
            "--dest",
            "org.dfmic.Control",
            "--object-path",
            "/org/dfmic/Control",
            "--method",
        ])
        .arg(format!("org.dfmic.Control.{method}"))
        .args(extra)
        .output();
    matches!(out, Ok(o) if o.status.success())
}

/// Ask the daemon for a parameter change it can apply live.
fn set_param(name: &str, value: f64) {
    let text = format!("'{name}'");
    let v = format!("{value}");
    let _ = call_method("SetParam", &[text.as_str(), v.as_str()]);
}

pub fn run() -> Result<(), String> {
    let window = AppWindow::new().map_err(|e| format!("cannot create the window: {e}"))?;

    // Controls -> daemon.
    {
        let handle = window.as_weak();
        window.on_toggle(move || {
            if let Some(ui) = handle.upgrade() {
                // Optimistic flip so the UI feels instant; the poll corrects it.
                let now = !ui.get_enabled();
                ui.set_enabled(now);
                call_method(if now { "Enable" } else { "Disable" }, &[]);
            }
        });
    }
    window.on_set_atten(|v| set_param("atten-lim", v as f64));
    window.on_set_post_filter(|v| set_param("post-filter", v as f64));
    window.on_quit(|| {
        let _ = slint::quit_event_loop();
    });

    // Daemon -> UI.
    let (tx, rx): (_, Receiver<Snapshot>) = channel();
    std::thread::spawn(move || loop {
        let snap = query();
        if tx.send(snap).is_err() {
            break;
        }
        std::thread::sleep(Duration::from_millis(120));
    });

    let timer = Timer::default();
    {
        let handle = window.as_weak();
        timer.start(TimerMode::Repeated, Duration::from_millis(120), move || {
            let Some(ui) = handle.upgrade() else { return };
            while let Ok(snap) = rx.try_recv() {
                ui.set_connected(snap.connected);
                ui.set_enabled(snap.enabled);
                ui.set_input_level(snap.level as f32);
                ui.set_output_level(snap.out_level as f32);
                // "In use" means something is actually recording from us.
                ui.set_listening(snap.inferences > 0 && snap.enabled);
                ui.set_device(if snap.device.is_empty() {
                    "NPU".into()
                } else {
                    snap.device.as_str().into()
                });
                let (friendly, bus) = friendly(&snap.source);
                ui.set_source(snap.source.as_str().into());
                ui.set_friendly_source(friendly.as_str().into());
                ui.set_bus_name(bus.as_str().into());
                ui.set_latency_ms(snap.latency_ms as f32);
                ui.set_inferences(snap.inferences as i32);
                ui.set_dropped(snap.dropped as i32);
                ui.set_error(snap.error.as_str().into());
                ui.set_atten_lim(snap.atten_lim as f32);
                ui.set_post_filter(snap.post_filter as f32);
            }
        });
    }

    window
        .run()
        .map_err(|e| format!("the window could not start: {e}"))
}
