//! Persistent daemon settings, validated at the trust boundary.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use dfn_dsp::{MaskReduce, Params};
use serde::{Deserialize, Serialize};

use crate::proto::{Param, MASK_REDUCE_NAMES, SOFT_PRESET};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    // Deliberately no `enabled` here. Whether the filter is on is runtime state,
    // not a setting: the daemon always starts with it off, so nothing can grab
    // the microphone unattended at login. An old file's `enabled = true` is
    // simply ignored.
    pub source: Option<String>,
    pub agc: bool,
    pub monitor: bool,
    pub take_default: bool,
    pub notify: bool,
    /// Ask PipeWire for a small graph quantum while filtering, and give it back
    /// when we stop. Off leaves the graph exactly as it would be without dfmic.
    pub graph_quantum: bool,
    pub devices: BTreeMap<String, DeviceSettings>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            source: None,
            agc: true,
            monitor: false,
            take_default: true,
            notify: true,
            graph_quantum: true,
            devices: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct DeviceSettings {
    pub atten_lim: f64,
    pub post_filter: f64,
    pub min_thresh: f64,
    pub max_erb_thresh: f64,
    pub max_df_thresh: f64,
    pub min_buffer: f64,
    pub mask_reduce: String,
    pub soft: bool,
}

impl Default for DeviceSettings {
    fn default() -> Self {
        Self {
            atten_lim: Param::AttenLim.spec().default,
            post_filter: Param::PostFilter.spec().default,
            min_thresh: Param::MinThresh.spec().default,
            max_erb_thresh: Param::MaxErbThresh.spec().default,
            max_df_thresh: Param::MaxDfThresh.spec().default,
            // Zero is libDF's default and the adaptive rule's floor: the queue
            // keeps two hops regardless (see `MIN_HOPS` in `pw.rs`), so this is
            // reserve the user asks for *on top* of the measured floor rather
            // than the floor itself.
            min_buffer: Param::MinBuffer.spec().default,
            mask_reduce: "none".into(),
            soft: false,
        }
    }
}

impl DeviceSettings {
    pub fn params(&self) -> Params {
        Params {
            atten_lim_db: self.atten_lim as f32,
            post_filter_beta: self.post_filter as f32,
            min_db_thresh: self.min_thresh as f32,
            max_db_erb_thresh: self.max_erb_thresh as f32,
            max_db_df_thresh: self.max_df_thresh as f32,
            min_buffer: self.min_buffer as u32,
            mask_reduce: match self.mask_reduce.as_str() {
                "max" => MaskReduce::Max,
                "mean" => MaskReduce::Mean,
                _ => MaskReduce::None,
            },
        }
    }

    pub fn effective_params(&self) -> Params {
        let mut params = self.params();
        if self.soft {
            for (param, value) in SOFT_PRESET {
                match param {
                    Param::AttenLim => params.atten_lim_db = value as f32,
                    Param::PostFilter => params.post_filter_beta = value as f32,
                    _ => unreachable!(),
                }
            }
        }
        params
    }

    pub fn value(&self, param: Param) -> f64 {
        match param {
            Param::AttenLim => self.atten_lim,
            Param::PostFilter => self.post_filter,
            Param::MinThresh => self.min_thresh,
            Param::MaxErbThresh => self.max_erb_thresh,
            Param::MaxDfThresh => self.max_df_thresh,
            Param::MinBuffer => self.min_buffer,
            Param::MaskReduce => match self.mask_reduce.as_str() {
                "max" => 1.0,
                "mean" => 2.0,
                _ => 0.0,
            },
        }
    }

    pub fn set(&mut self, param: Param, value: f64) {
        match param {
            Param::AttenLim => self.atten_lim = value,
            Param::PostFilter => self.post_filter = value,
            Param::MinThresh => self.min_thresh = value,
            Param::MaxErbThresh => self.max_erb_thresh = value,
            Param::MaxDfThresh => self.max_df_thresh = value,
            Param::MinBuffer => self.min_buffer = value,
            Param::MaskReduce => {
                self.mask_reduce = MASK_REDUCE_NAMES[value as usize].to_string();
            }
        }
    }

    pub fn params_table(&self) -> Vec<(String, f64)> {
        Param::ALL
            .iter()
            .map(|param| (param.spec().key.to_string(), self.value(*param)))
            .collect()
    }
}

fn validate_number(
    device: &str,
    settings: &mut DeviceSettings,
    param: Param,
    notices: &mut Vec<String>,
) {
    let value = settings.value(param);
    let spec = param.spec();
    let valid = value.is_finite() && value >= spec.min && value <= spec.max;
    let integral = !matches!(param, Param::MinBuffer | Param::MaskReduce)
        || (value.fract() == 0.0 && (param != Param::MaskReduce || value <= 2.0));
    if valid && integral {
        return;
    }
    settings.set(param, spec.default);
    let label = spec.key.replace('-', " ");
    let subject = if device.is_empty() {
        format!("Saved {label}")
    } else {
        format!("Saved {label} for {device}")
    };
    notices.push(format!(
        "{subject} {value} was out of range ({}–{}); reset to {}.",
        spec.min, spec.max, spec.default
    ));
}

fn validate_device(device: &str, settings: &mut DeviceSettings, notices: &mut Vec<String>) {
    for param in Param::ALL {
        validate_number(device, settings, param, notices);
    }
    if !MASK_REDUCE_NAMES.contains(&settings.mask_reduce.as_str()) {
        notices.push(format!(
            "Saved mask reduction {:?} for {device} was not recognised; reset to none.",
            settings.mask_reduce
        ));
        settings.mask_reduce = "none".into();
    }
}

pub fn settings_path() -> PathBuf {
    config_root().join("dfmic/settings.toml")
}

fn config_root() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .unwrap_or_else(|| PathBuf::from(".config"))
}

pub fn load() -> Result<(Settings, Vec<String>), String> {
    let path = settings_path();
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((Settings::default(), Vec::new()));
        }
        Err(error) => return Err(format!("could not read {}: {error}", path.display())),
    };
    let mut settings: Settings = match toml::from_str(&text) {
        Ok(settings) => settings,
        Err(error) => {
            return Ok((
                Settings::default(),
                vec![format!(
                    "Saved settings could not be read ({error}); defaults were loaded."
                )],
            ));
        }
    };
    let mut notices = Vec::new();
    if settings
        .source
        .as_ref()
        .is_some_and(|source| source.trim().is_empty())
    {
        settings.source = None;
        notices.push("Saved source was empty; the current system default will be used.".into());
    }
    for (device, device_settings) in &mut settings.devices {
        validate_device(device, device_settings, &mut notices);
    }
    Ok((settings, notices))
}

/// Write settings without ever exposing a partially written file to another
/// daemon or to a crash-recovery run.
pub fn save(settings: &Settings) -> Result<(), String> {
    let path = settings_path();
    save_atomic(
        &path,
        &toml::to_string_pretty(settings).map_err(|error| error.to_string())?,
    )
}

pub fn save_atomic(path: &Path, contents: &str) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    // The name carries a random suffix and is created exclusively, so two
    // processes saving at once cannot interleave in one file, and an existing
    // symlink at the temporary's name is a failure rather than a file written
    // through a link the caller never asked for.
    let temporary = path.with_extension(format!("tmp.{}.{}", std::process::id(), random_suffix()));
    let result = (|| -> Result<(), String> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| format!("could not create {}: {error}", temporary.display()))?;
        file.write_all(contents.as_bytes())
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("could not write {}: {error}", temporary.display()))?;
        drop(file);
        fs::rename(&temporary, path)
            .map_err(|error| format!("could not replace {}: {error}", path.display()))?;
        // The rename is atomic but not durable until the *directory* is synced:
        // without this a power cut can leave the old contents, or no file at all,
        // even though `rename` returned success.
        fs::File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("could not sync {}: {error}", parent.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

/// A short unpredictable suffix, so two writers never pick the same temporary.
fn random_suffix() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u32(std::process::id());
    format!("{:x}", hasher.finish() as u32)
}
