//! Command-line client. Every runtime operation uses the generated zbus proxy.

use std::collections::HashMap;

use serde::Serialize;

use crate::proto::{ControlProxyBlocking, Device, Failure, Levels, MonitorTarget, Param, MASK_REDUCE_NAMES};
use crate::service;

#[derive(Serialize)]
struct Status {
    state: String,
    enabled: bool,
    listening: bool,
    source: String,
    source_description: String,
    source_available: bool,
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

fn proxy() -> Result<ControlProxyBlocking<'static>, String> {
    let connection = zbus::blocking::Connection::session()
        .map_err(|error| daemon_error("could not connect to the session D-Bus", error))?;
    ControlProxyBlocking::new_owned(connection)
        .map_err(|error| daemon_error("could not reach the dfmic daemon", error))
}

fn read_status(proxy: &ControlProxyBlocking<'_>) -> Result<Status, String> {
    macro_rules! get {
        ($method:ident) => {
            proxy.$method().map_err(|error| format!("could not read {}: {error}", stringify!($method)))?
        };
    }
    Ok(Status {
        state: get!(state),
        enabled: get!(enabled),
        listening: get!(listening),
        source: get!(source),
        source_description: get!(source_description),
        source_available: get!(source_available),
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
    if value > 0.0 { 20.0 * value.log10() } else { -160.0 }
}

pub fn status(json: bool) -> Result<(), String> {
    let proxy = proxy()?;
    let status = read_status(&proxy)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&status).map_err(|error| error.to_string())?);
        return Ok(());
    }
    println!("state: {}", status.state);
    println!("enabled: {}", status.enabled);
    println!("listening: {}", status.listening);
    println!("source: {}", if status.source.is_empty() { "(none)" } else { &status.source });
    println!("source description: {}", status.source_description);
    println!("source available: {}", status.source_available);
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
    println!("agc gain: {:.2} dB; voice: {}", status.levels.agc_gain_db, status.levels.voice);
    println!("latency: {:.2} ms", status.latency_ms);
    for key in ["capture", "buffer", "model", "inference", "output"] {
        if let Some(value) = status.latency_parts.get(key) {
            println!("  {key}: {value:.3} ms");
        }
    }
    println!("inference: {:.3} ms/hop; total: {}", status.inference_ms, status.inferences);
    println!("overruns: {}; underruns: {}", status.overruns, status.underruns);
    println!("parameters (current device's own values):");
    for param in Param::ALL {
        if let Some(value) = status.params.get(param.spec().key) {
            println!("  {} = {value}", param.spec().key);
        }
    }
    println!("soft mode: {}", status.soft_mode);
    println!("agc: {}", status.agc);
    println!("monitor: {}", status.monitor);
    println!(
        "monitor target: {} ({}) [{}]",
        status.monitor_target.description, status.monitor_target.name, status.monitor_target.kind
    );
    println!("take default: {}; is default: {}", status.take_default, status.is_default);
    println!("autostart: {}; notify: {}", status.autostart, status.notify);
    println!("devices:");
    for device in &status.devices {
        println!(
            "  {}{} {} [bus={} form={} virtual={}]",
            device.name,
            if device.is_system_default { " (system default)" } else { "" },
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
    println!("enabled: {enabled}; state: {state}");
    Ok(())
}

pub fn toggle() -> Result<(), String> {
    let proxy = proxy()?;
    proxy.toggle().map_err(|error| error.to_string())?;
    print_state(&proxy)
}

pub fn set_enabled(enabled: bool) -> Result<(), String> {
    let proxy = proxy()?;
    proxy.set_enabled(enabled).map_err(|error| error.to_string())?;
    print_state(&proxy)
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
        value.parse::<f64>().map_err(|_| format!("{key} expects a number; got {value:?}"))?
    };
    let proxy = proxy()?;
    proxy.set_param(key, value).map_err(|error| error.to_string())?;
    println!("{key} = {value}");
    Ok(())
}

pub fn reset(all: bool) -> Result<(), String> {
    let proxy = proxy()?;
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
    let proxy = proxy()?;
    match argument {
        None => {
            let source = proxy.source().map_err(|error| error.to_string())?;
            let description = proxy.source_description().map_err(|error| error.to_string())?;
            let available = proxy.source_available().map_err(|error| error.to_string())?;
            println!("{source}\n{description}\navailable: {available}");
        }
        Some("list") => {
            for device in proxy.devices().map_err(|error| error.to_string())? {
                println!(
                    "{}{}\t{}\tbus={} form={} virtual={}",
                    device.name,
                    if device.is_system_default { " (system default)" } else { "" },
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
    let proxy = proxy()?;
    match command {
        "soft" => proxy.set_soft_mode(on),
        "agc" => proxy.set_agc(on),
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

pub fn service_start() -> Result<(), String> {
    service::start()?;
    println!("{}\": start requested", crate::proto::UNIT);
    Ok(())
}

pub fn mask_names() -> &'static [&'static str; 3] {
    &MASK_REDUCE_NAMES
}
