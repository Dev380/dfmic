//! systemd user-service operations shared by the daemon, CLI and GUI.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::proto::{Failure, UNIT};
use crate::settings::save_atomic;

fn systemctl(arguments: &[&str]) -> std::io::Result<std::process::Output> {
    Command::new("systemctl")
        .arg("--user")
        .args(arguments)
        .output()
}

fn command_error(action: &str, output: std::process::Output) -> String {
    let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let detail = if detail.is_empty() {
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    } else {
        detail
    };
    format!("could not {action} {UNIT}: {detail}")
}

pub fn unit_installed() -> bool {
    systemctl(&["cat", UNIT]).is_ok_and(|output| output.status.success())
}

pub fn is_active() -> bool {
    systemctl(&["is-active", "--quiet", UNIT]).is_ok_and(|output| output.status.success())
}

pub fn start() -> Result<(), String> {
    if !unit_installed() {
        return Err(format!(
            "{UNIT} is not installed. Install the NixOS module or user unit first."
        ));
    }
    let output = systemctl(&["start", UNIT])
        .map_err(|error| format!("could not run systemctl --user start {UNIT}: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(command_error("start", output))
    }
}

pub fn autostart_enabled() -> Result<bool, String> {
    let output = systemctl(&["is-enabled", UNIT])
        .map_err(|error| format!("could not run systemctl --user is-enabled {UNIT}: {error}"))?;
    if output.status.success() {
        return Ok(true);
    }
    let state = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if matches!(
        state.as_str(),
        "disabled" | "indirect" | "masked" | "static" | "generated" | "transient"
    ) {
        Ok(false)
    } else {
        Err(command_error("query autostart for", output))
    }
}

pub fn set_autostart(enabled: bool) -> Result<(), String> {
    if !unit_installed() {
        return Err(format!(
            "{UNIT} is not installed; install the user service first."
        ));
    }
    let operation = if enabled { "enable" } else { "disable" };
    let output = systemctl(&[operation, UNIT])
        .map_err(|error| format!("could not run systemctl --user {operation} {UNIT}: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(command_error(operation, output))
    }
}

pub fn failure_path() -> PathBuf {
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/state")))
        .unwrap_or_else(|| PathBuf::from(".local/state"))
        .join("dfmic/last-failure")
}

pub fn last_failure() -> Option<Failure> {
    let path = failure_path();
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            eprintln!("dfmic: could not read {}: {error}", path.display());
            return None;
        }
    };
    match serde_json::from_str(&text) {
        Ok(failure) => Some(failure),
        Err(error) => {
            eprintln!("dfmic: could not parse {}: {error}", path.display());
            None
        }
    }
}

pub fn clear_last_failure() {
    match fs::remove_file(failure_path()) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => eprintln!("dfmic: could not clear last failure: {error}"),
    }
}

pub fn record_failure(message: String, phase: &str) -> Failure {
    let failure = Failure {
        message,
        phase: phase.into(),
        when: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
    };
    if let Ok(text) = serde_json::to_string_pretty(&failure) {
        if let Err(error) = save_atomic(&failure_path(), &text) {
            eprintln!("dfmic: {error}");
        }
    }
    failure
}
