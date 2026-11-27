//! Command-line client. Every runtime operation uses the generated zbus proxy.

use serde::Serialize;
use std::collections::HashMap;

use crate::proto::{ControlProxyBlocking, Device, Failure, Levels, MonitorTarget, Param, State};
use crate::service;

#[derive(Serialize)]
struct Status {
    state: String,
    enabled: bool,
    listening: bool,
    source: String,
    source_description: String,
    source_available: bool,
    source_following: bool,
    model_present: bool,
    model_downloading: bool,
    model_download_possible: bool,
    devices: Vec<Device>,
    compute_device: String,
    levels: Levels,
    inference_ms: f64,
    inferences: u64,
    latency_ms: f64,
    latency_parts: HashMap<String, f64>,
    overruns: u64,
    underruns: u64,
    params: HashMap<String, f64>,
    soft_mode: bool,
    agc: bool,
    graph_quantum: bool,
    monitor: bool,
    monitor_target: MonitorTarget,
    take_default: bool,
    is_default: bool,
    autostart: bool,
    notify: bool,
    notices: Vec<String>,
    last_failure: Failure,
}

fn daemon_error(context: &str, error: impl std::fmt::Display) -> String {
    let mut message = format!(
        "{context}: {error}\nThe dfmic daemon is not running. Start it with `dfmic service start` or `dfmic run`."
    );
    if let Some(failure) = service::last_failure() {
        if !failure.message.is_empty() {
            message.push_str(&format!(
                "\nLast failure ({}): {}",
                failure.phase, failure.message
            ));
        }
    }
    message
}

/// A failed method call, as the user should see it.
///
/// The daemon's errors carry a plain-language sentence (`proto.rs` promises a
/// client may show it verbatim), but `zbus::Error`'s `Display` prefixes it with
/// `org.freedesktop.DBus.Error.Failed: `. That prefix is noise in a terminal, so
/// it is stripped here; everything else is passed through unchanged.
fn method_error(error: zbus::Error) -> String {
    match error {
        zbus::Error::MethodError(_, Some(message), _) if !message.is_empty() => message,
        other => other.to_string(),
    }
}

fn with_proxy<T>(
    operation: impl FnOnce(&ControlProxyBlocking<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let connection = zbus::blocking::Connection::session()
        .map_err(|error| daemon_error("could not connect to the session D-Bus", error))?;
    let proxy = ControlProxyBlocking::new(&connection)
        .map_err(|error| daemon_error("could not reach the dfmic daemon", error))?;
    operation(&proxy)
}

fn read_status(proxy: &ControlProxyBlocking<'_>) -> Result<Status, String> {
    macro_rules! get {
        ($method:ident) => {
            proxy.$method().map_err(|error| {
                daemon_error(concat!("could not read ", stringify!($method)), error)
            })?
        };
    }
    Ok(Status {
        state: get!(state),
        enabled: get!(enabled),
        listening: get!(listening),
        source: get!(source),
        source_description: get!(source_description),
        source_available: get!(source_available),
        source_following: get!(source_following),
        model_present: get!(model_present),
        model_downloading: get!(model_downloading),
        model_download_possible: get!(model_download_possible),
        devices: get!(devices),
        compute_device: get!(compute_device),
        levels: get!(levels),
        inference_ms: get!(inference_ms),
        inferences: get!(inferences),
        latency_ms: get!(latency_ms),
        latency_parts: get!(latency_parts),
        overruns: get!(overruns),
        underruns: get!(underruns),
        params: get!(params),
        soft_mode: get!(soft_mode),
        agc: get!(agc),
        graph_quantum: get!(graph_quantum),
        monitor: get!(monitor),
        monitor_target: get!(monitor_target),
        take_default: get!(take_default),
        is_default: get!(is_default),
        autostart: get!(autostart),
        notify: get!(notify),
        notices: get!(notices),
        last_failure: get!(last_failure),
    })
}

fn db(value: f64) -> f64 {
    if value > 0.0 {
        20.0 * value.log10()
    } else {
        -160.0
    }
}

pub fn status(json: bool) -> Result<(), String> {
    with_proxy(|proxy| status_inner(proxy, json))
}

fn status_inner(proxy: &ControlProxyBlocking<'_>, json: bool) -> Result<(), String> {
    let status = read_status(proxy)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&status).map_err(|error| error.to_string())?
        );
        return Ok(());
    }
    println!(
        "state: {}\nenabled: {}\nlistening: {}",
        status.state, status.enabled, status.listening
    );
    println!(
        "source: {}",
        if status.source_following {
            format!(
                "default → {}{}",
                status.source,
                if status.source.is_empty() {
                    ""
                } else {
                    " (following the system default)"
                }
            )
        } else if status.source.is_empty() {
            "(none)".to_string()
        } else {
            status.source.clone()
        }
    );
    println!(
        "source description: {}\nsource available: {}",
        status.source_description, status.source_available
    );
    println!("compute device: {}", status.compute_device);
    println!(
        "input: active={} peak={:.5} ({:.1} dBFS) rms={:.5} ({:.1} dBFS) clip={}",
        status.levels.input_active,
        status.levels.input_peak,
        db(status.levels.input_peak),
        status.levels.input_rms,
        db(status.levels.input_rms),
        status.levels.input_clip
    );
    println!(
        "output: active={} peak={:.5} ({:.1} dBFS) rms={:.5} ({:.1} dBFS) clip={}",
        status.levels.output_active,
        status.levels.output_peak,
        db(status.levels.output_peak),
        status.levels.output_rms,
        db(status.levels.output_rms),
        status.levels.output_clip
    );
    println!(
        "agc gain: {:.2} dB; voice: {}",
        status.levels.agc_gain_db, status.levels.voice
    );
    println!("latency: {:.2} ms", status.latency_ms);
    for key in ["capture", "buffer", "model", "output"] {
        if let Some(value) = status.latency_parts.get(key) {
            println!("  {key}: {value:.3} ms");
        }
    }
    println!(
        "inference: {:.3} ms/hop; total: {}",
        status.inference_ms, status.inferences
    );
    println!(
        "overruns: {}; underruns: {}",
        status.overruns, status.underruns
    );
    println!("parameters (current device's own values):");
    for param in Param::ALL {
        if let Some(value) = status.params.get(param.spec().key) {
            println!(
                "  {} = {value} {} (step {})",
                param.spec().key,
                param.spec().unit,
                param.spec().step
            );
        }
    }
    println!(
        "soft mode: {}\nagc: {}\nlow-latency graph: {}\nmonitor: {}",
        status.soft_mode, status.agc, status.graph_quantum, status.monitor
    );
    println!(
        "monitor target: {} ({}) [{}]",
        status.monitor_target.description, status.monitor_target.name, status.monitor_target.kind
    );
    println!(
        "take default: {}; is default: {}",
        status.take_default, status.is_default
    );
    println!("autostart: {}; notify: {}", status.autostart, status.notify);
    println!("devices:");
    for device in &status.devices {
        println!(
            "  {}{} {} [bus={} form={} virtual={}]",
            device.name,
            if device.is_system_default {
                " (system default)"
            } else {
                ""
            },
            device.description,
            device.bus,
            device.form_factor,
            device.virtual_node
        );
    }
    for notice in &status.notices {
        println!("notice: {notice}");
    }
    if !status.last_failure.message.is_empty() {
        println!(
            "last failure: {} ({})",
            status.last_failure.message, status.last_failure.phase
        );
    }
    Ok(())
}

fn print_state(proxy: &ControlProxyBlocking<'_>) -> Result<(), String> {
    let enabled = proxy.enabled().map_err(method_error)?;
    let state = proxy.state().map_err(method_error)?;
    State::parse(&state).ok_or_else(|| format!("daemon returned an unknown state {state:?}"))?;
    println!("enabled: {enabled}; state: {state}");
    Ok(())
}

pub fn toggle() -> Result<(), String> {
    with_proxy(|proxy| {
        proxy.toggle().map_err(method_error)?;
        print_state(proxy)
    })
}

pub fn set_enabled(enabled: bool) -> Result<(), String> {
    with_proxy(|proxy| {
        proxy.set_enabled(enabled).map_err(method_error)?;
        print_state(proxy)
    })
}

pub fn set_param(key: &str, value: &str) -> Result<(), String> {
    let value = if key == Param::MaskReduce.spec().key {
        match value {
            "none" => 0.0,
            "max" => 1.0,
            "mean" => 2.0,
            _ => value.parse::<f64>().map_err(|_| {
                format!("mask-reduce expects none, max or mean (also 0, 1 or 2); got {value:?}")
            })?,
        }
    } else {
        value
            .parse::<f64>()
            .map_err(|_| format!("{key} expects a number; got {value:?}"))?
    };
    with_proxy(|proxy| {
        proxy.set_param(key, value).map_err(method_error)?;
        println!("{key} = {value}");
        Ok(())
    })
}

pub fn reset(all: bool) -> Result<(), String> {
    with_proxy(|proxy| reset_inner(proxy, all))
}

fn reset_inner(proxy: &ControlProxyBlocking<'_>, all: bool) -> Result<(), String> {
    if all {
        proxy.reset_all().map_err(method_error)?;
        println!("All settings and per-device parameters were reset (autostart unchanged).");
    } else {
        proxy.reset_params().map_err(method_error)?;
        println!("Current device parameters were reset and soft mode was turned off.");
    }
    Ok(())
}

pub fn source(argument: Option<&str>) -> Result<(), String> {
    with_proxy(|proxy| source_inner(proxy, argument))
}

fn source_inner(proxy: &ControlProxyBlocking<'_>, argument: Option<&str>) -> Result<(), String> {
    match argument {
        None => {
            let following = proxy.source_following().map_err(method_error)?;
            println!(
                "{}{}\n{}\navailable: {}",
                proxy.source().map_err(method_error)?,
                if following {
                    " (following the system default)"
                } else {
                    ""
                },
                proxy.source_description().map_err(method_error)?,
                proxy.source_available().map_err(method_error)?
            );
        }
        Some("list") => {
            let following = proxy.source_following().map_err(method_error)?;
            println!(
                "{}\tfollow the system default{}",
                crate::proto::SOURCE_DEFAULT,
                if following { " (selected)" } else { "" }
            );
            for device in proxy.devices().map_err(method_error)? {
                println!(
                    "{}{}\t{}\tbus={} form={} virtual={}",
                    device.name,
                    if device.is_system_default {
                        " (system default)"
                    } else {
                        ""
                    },
                    device.description,
                    device.bus,
                    device.form_factor,
                    device.virtual_node
                );
            }
        }
        Some(name) => {
            proxy.set_source(name).map_err(method_error)?;
            println!("source: {name}");
        }
    }
    Ok(())
}

fn on_off(value: &str) -> Result<bool, String> {
    match value {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => Err(format!("expected on or off, got {value:?}")),
    }
}

pub fn setting(command: &str, value: &str) -> Result<(), String> {
    let on = on_off(value)?;
    with_proxy(|proxy| setting_inner(proxy, command, on))
}

fn setting_inner(proxy: &ControlProxyBlocking<'_>, command: &str, on: bool) -> Result<(), String> {
    match command {
        "soft" => proxy.set_soft_mode(on),
        "agc" => proxy.set_agc(on),
        "graph-quantum" => proxy.set_graph_quantum(on),
        "monitor" => proxy.set_monitor(on),
        "default" => proxy.set_take_default(on),
        "autostart" => proxy.set_autostart(on),
        "notify" => proxy.set_notify(on),
        _ => return Err(format!("unknown setting {command:?}")),
    }
    .map_err(method_error)?;
    println!("{command}: {}", if on { "on" } else { "off" });
    Ok(())
}

pub fn dismiss_notices() -> Result<(), String> {
    with_proxy(|proxy| {
        proxy.dismiss_notices().map_err(method_error)?;
        println!("notices cleared");
        Ok(())
    })
}

pub fn clear_failure() -> Result<(), String> {
    with_proxy(|proxy| {
        proxy.clear_failure().map_err(method_error)?;
        println!("last failure cleared");
        Ok(())
    })
}

pub fn service_start() -> Result<(), String> {
    service::start()?;
    println!(
        "{}: start requested; active: {}",
        crate::proto::UNIT,
        service::is_active()
    );
    Ok(())
}

/// Print where the model is, and whether it is there.
///
/// `models` is the caller's `--models DIR`; without it the search is
/// `DFMIC_MODELS` then the data directory, the same rule the daemon uses.
pub fn models(explicit: Option<&std::path::Path>) -> Result<(), String> {
    match explicit {
        Some(dir) => println!("models directory: {}", dir.display()),
        None => println!("data directory: {}", crate::models::data_dir().display()),
    }
    match crate::models::resolve(explicit) {
        Ok(dir) => {
            println!("model: present");
            println!("using: {}", dir.display());
            Ok(())
        }
        Err(_) => {
            println!("model: not installed");
            if crate::models::downloadable() {
                println!("run `dfmic models download` to fetch it (about 20 MB)");
            } else {
                println!("this build has no download URL; pass --models DIR or set DFMIC_MODELS");
            }
            Err("the model is not installed".into())
        }
    }
}

/// Fetch the model into the data directory.
///
/// Through the daemon when one is running, so the download happens in the
/// process that owns the data directory and the window can watch it; directly
/// otherwise, which is the case before the service has ever been started.
///
/// `models` is the caller's `--models DIR`. It decides where a direct download
/// lands; when a daemon is running it is the daemon's own path that matters, and
/// the daemon reports where the files went, so nothing here has to guess.
pub fn models_download(models: Option<&std::path::Path>) -> Result<(), String> {
    // Whether a daemon answers decides which path to take, and that has to be
    // established before the download is requested: falling back to a direct
    // download after the daemon has *refused* would quietly write the model
    // somewhere the daemon may not read, and hide the reason it gave.
    // Only when the daemon would write where this invocation asked for. A daemon
    // is started from a session manager with the login environment, so a caller
    // that overrode `XDG_DATA_HOME` (or passed `--models`) is describing a
    // directory the daemon does not read; asking it anyway would put the files
    // somewhere else and then report success for a path that stayed empty.
    let daemon_path = with_proxy(|proxy| proxy.model_path().map_err(method_error)).ok();
    let live = with_proxy(|proxy| proxy.model_present().map(|_| ()).map_err(method_error)).is_ok()
        && daemon_path.as_deref().is_some_and(|path| {
            std::path::Path::new(path) == crate::models::download_target(models)
        });
    if live {
        with_proxy(|proxy| proxy.download_model().map_err(method_error))?;
        // Two bounds, because the two ways this can stall are different: the
        // daemon never starts, or the transfer itself hangs. Once it has started,
        // `curl`'s own `--max-time` is the limit, with slack over it so a slow
        // link is not reported as a failure by this loop first.
        let start_deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        let transfer_limit = std::time::Duration::from_secs(20 * 60);
        let mut started = None;
        loop {
            std::thread::sleep(std::time::Duration::from_millis(200));
            let (downloading, present) = with_proxy(|proxy| {
                Ok((
                    proxy.model_downloading().map_err(method_error)?,
                    proxy.model_present().map_err(method_error)?,
                ))
            })?;
            if present {
                // The daemon knows where it wrote the files; it says so through
                // the Source of truth every other client reads.
                let dir = with_proxy(|proxy| proxy.model_path().map_err(method_error))?;
                println!("model ready in {dir}");
                return Ok(());
            }
            if downloading {
                let start = *started.get_or_insert_with(std::time::Instant::now);
                if start.elapsed() >= transfer_limit {
                    return Err(format!(
                        "the download did not finish within {} minutes",
                        transfer_limit.as_secs() / 60
                    ));
                }
            } else if started.is_some() {
                // It ran and finished without producing a model: the daemon
                // has already told the user why, through a notification.
                return Err("the download finished without installing the model".into());
            } else if std::time::Instant::now() >= start_deadline {
                return Err("the daemon did not start the download".into());
            }
        }
    }
    let dir = crate::models::download_default(models)?;
    if let Some(path) = daemon_path {
        if !path.is_empty() && std::path::Path::new(&path) != dir {
            eprintln!(
                "dfmic: note: the running daemon reads its model from {path}, which this \
                 download did not touch; restart it against {} to use what was fetched.",
                dir.display()
            );
        }
    }
    println!("model ready in {}", dir.display());
    Ok(())
}
