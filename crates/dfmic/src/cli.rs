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
    let enabled = proxy.enabled().map_err(|error| error.to_string())?;
    let state = proxy.state().map_err(|error| error.to_string())?;
    State::parse(&state).ok_or_else(|| format!("daemon returned an unknown state {state:?}"))?;
    println!("enabled: {enabled}; state: {state}");
    Ok(())
}

pub fn toggle() -> Result<(), String> {
    with_proxy(|proxy| {
        proxy.toggle().map_err(|error| error.to_string())?;
        print_state(proxy)
    })
}

pub fn set_enabled(enabled: bool) -> Result<(), String> {
    with_proxy(|proxy| {
        proxy
            .set_enabled(enabled)
            .map_err(|error| error.to_string())?;
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
        proxy
            .set_param(key, value)
            .map_err(|error| error.to_string())?;
        println!("{key} = {value}");
        Ok(())
    })
}

pub fn reset(all: bool) -> Result<(), String> {
    with_proxy(|proxy| reset_inner(proxy, all))
}

fn reset_inner(proxy: &ControlProxyBlocking<'_>, all: bool) -> Result<(), String> {
    if all {
        proxy.reset_all().map_err(|error| error.to_string())?;
        println!("All settings and per-device parameters were reset (autostart unchanged).");
    } else {
        proxy.reset_params().map_err(|error| error.to_string())?;
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
            let following = proxy.source_following().map_err(|e| e.to_string())?;
            println!(
                "{}{}\n{}\navailable: {}",
                proxy.source().map_err(|e| e.to_string())?,
                if following {
                    " (following the system default)"
                } else {
                    ""
                },
                proxy.source_description().map_err(|e| e.to_string())?,
                proxy.source_available().map_err(|e| e.to_string())?
            );
        }
        Some("list") => {
            let following = proxy.source_following().map_err(|e| e.to_string())?;
            println!(
                "{}\tfollow the system default{}",
                crate::proto::SOURCE_DEFAULT,
                if following { " (selected)" } else { "" }
            );
            for device in proxy.devices().map_err(|error| error.to_string())? {
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
            proxy.set_source(name).map_err(|error| error.to_string())?;
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
    .map_err(|error| error.to_string())?;
    println!("{command}: {}", if on { "on" } else { "off" });
    Ok(())
}

pub fn dismiss_notices() -> Result<(), String> {
    with_proxy(|proxy| {
        proxy.dismiss_notices().map_err(|error| error.to_string())?;
        println!("notices cleared");
        Ok(())
    })
}

pub fn clear_failure() -> Result<(), String> {
    with_proxy(|proxy| {
        proxy.clear_failure().map_err(|error| error.to_string())?;
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
