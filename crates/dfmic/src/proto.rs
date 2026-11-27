//! The control contract between the daemon and its clients (CLI and GUI).
//!
//! The daemon is the only owner of state. Everything a client can do goes
//! through the `org.dfmic.Control1` interface described here; the GUI is a view
//! onto it, the CLI is another view onto it, and neither keeps state of its own.
//!
//! Bus name `org.dfmic.Control`, object `/org/dfmic/Control`, session bus.
//!
//! Property semantics:
//!   * Every property below emits `org.freedesktop.DBus.Properties.PropertiesChanged`
//!     when it changes. Fast-moving diagnostics (`LatencyMs`, `LatencyParts`,
//!     `InferenceMs`, `Inferences`, `Overruns`, `Underruns`) are coalesced to at
//!     most 2 Hz.
//!   * Levels are *not* properties: they are pulled with `Levels()`, which is
//!     stateless (any number of clients may poll it) and costs nothing when no
//!     client is polling.
//!
//! Method errors are D-Bus errors whose message is a plain-language sentence a
//! client may show to the user verbatim (for example "That is DeepFilter Mic
//! itself; capturing it would feed its own output back in.").

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use zbus::zvariant::{OwnedValue, Type, Value};

pub const BUS_NAME: &str = "org.dfmic.Control";
pub const OBJECT_PATH: &str = "/org/dfmic/Control";
pub const INTERFACE: &str = "org.dfmic.Control1";

/// PipeWire node name and description of the exported virtual microphone.
pub const NODE_NAME: &str = "dfmic-source";

/// The `source` setting that means "follow whatever the system default is".
///
/// Stored as a plain string like any device name, so the settings file, the
/// command line and the bus all carry it without a separate flag. The daemon
/// resolves it to a real device on every registry change, which is what makes
/// plugging in a different microphone just work.
pub const SOURCE_DEFAULT: &str = "default";
pub const NODE_DESCRIPTION: &str = "DeepFilter Mic";

/// systemd user unit that runs the daemon.
pub const UNIT: &str = "dfmic.service";

/// What the daemon is doing right now. Exposed as the `State` string property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Bringing the streams up and loading the NPU. The interface is already
    /// exported and the bus name already owned (both happen first, so a client
    /// never sees a name with nothing behind it), but no device list has been
    /// read yet.
    Starting,
    /// Enabled, and at least one application (or the self-monitor) is
    /// recording from the virtual microphone: audio is going through the NPU.
    Filtering,
    /// Enabled, but nothing is recording from the virtual microphone. The NPU is
    /// idle; the input meter still works, the output meter has nothing to show.
    Idle,
    /// Disabled by the user. The virtual microphone stays in the device list and
    /// passes the raw microphone through untouched (no suppression, no AGC). The
    /// default-source takeover is released while off.
    Off,
    /// The selected capture device is gone. The virtual microphone outputs
    /// silence and the takeover is released; the daemon resumes by itself when
    /// the device comes back, or when the user picks another. It never falls
    /// back to a different device on its own.
    NoDevice,
    /// The model has not been downloaded yet. The daemon runs normally — it owns
    /// the bus name, lists devices and keeps the node up — but it cannot filter,
    /// so audio passes through untouched until `DownloadModel()` is called.
    NoModel,
}

impl State {
    pub fn as_str(self) -> &'static str {
        match self {
            State::Starting => "starting",
            State::Filtering => "filtering",
            State::Idle => "idle",
            State::Off => "off",
            State::NoDevice => "no-device",
            State::NoModel => "no-model",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "starting" => State::Starting,
            "filtering" => State::Filtering,
            "idle" => State::Idle,
            "off" => State::Off,
            "no-device" => State::NoDevice,
            "no-model" => State::NoModel,
            _ => return None,
        })
    }
}

/// A physical capture device, as listed in the `Devices` property.
///
/// The daemon never lists its own node. Virtual sources from other software
/// are listed with `virtual_node = true` so a client can de-emphasise them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type, Value, OwnedValue)]
pub struct Device {
    /// PipeWire `node.name`; the stable identifier used by `SetSource`.
    pub name: String,
    /// PipeWire `node.description` (falls back to `node.nick`, then `name`).
    pub description: String,
    /// `device.bus`: "bluetooth", "usb", "pci", or "" when unknown.
    pub bus: String,
    /// `device.form-factor`: "internal", "microphone", "headset", "handset",
    /// "webcam", "headphone", or "" when unknown.
    pub form_factor: String,
    /// True when the node has no backing device (filter chains, loopbacks, ...).
    pub virtual_node: bool,
    /// True if this is the system default source *other than* DeepFilter Mic,
    /// i.e. the device a fresh install would pick.
    pub is_system_default: bool,
}

/// Where the self-monitor plays to: the current default sink.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type, Value, OwnedValue)]
pub struct MonitorTarget {
    pub name: String,
    pub description: String,
    /// "headphones" (wired/bluetooth headphones or a headset), "speakers",
    /// or "unknown". Clients must treat anything but "headphones" as speakers.
    pub kind: String,
}

/// The last fatal failure, persisted across runs so a client that comes up
/// after the daemon died can still say what happened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type, Value, OwnedValue)]
pub struct Failure {
    /// Plain-language description; empty means "no failure on record".
    pub message: String,
    /// Unix time (seconds) when it happened.
    pub when: i64,
    /// "startup" (the daemon refused to start: NPU, models, PipeWire) or
    /// "stream" (it died while running and restored the real microphone).
    pub phase: String,
}

/// One meter reading, from `Levels()`. Amplitudes are linear full-scale
/// (1.0 = 0 dBFS) and cover the most recent ~50 ms of audio.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, Type)]
pub struct Levels {
    /// The capture stream delivered samples in the window. False means "no
    /// audio at all" (device stalled/absent), as opposed to a quiet signal.
    pub input_active: bool,
    pub input_peak: f64,
    pub input_rms: f64,
    /// Any sample in the window reached full scale (|x| >= 0.999).
    pub input_clip: bool,
    /// Samples were handed to a consumer in the window (an app or the monitor).
    pub output_active: bool,
    pub output_peak: f64,
    pub output_rms: f64,
    pub output_clip: bool,
    /// Gain the AGC is currently applying, in dB (0 when AGC is off or bypassed).
    pub agc_gain_db: f64,
    /// The AGC's gate was open in the window: the input is above the tracked
    /// room floor and the model has not zero-masked the frame. False whenever
    /// the filter is not running.
    pub voice: bool,
}

/// Tunable DeepFilterNet parameters. All travel as `f64` over the bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Param {
    AttenLim,
    PostFilter,
    MinThresh,
    MaxErbThresh,
    MaxDfThresh,
    MinBuffer,
    MaskReduce,
}

pub struct ParamSpec {
    pub param: Param,
    /// Key used on the bus, on the command line, and in the settings file.
    pub key: &'static str,
    pub min: f64,
    pub max: f64,
    /// Smallest meaningful increment, for a client's spinner or slider. Values are
    /// *not* rounded to it: `SetParam` rejects out-of-range and non-integral
    /// `min-buffer`/`mask-reduce`, and stores anything else as given.
    pub step: f64,
    /// libDF's recommended value.
    pub default: f64,
    pub unit: &'static str,
}

pub const PARAMS: [ParamSpec; 7] = [
    ParamSpec {
        param: Param::AttenLim,
        key: "atten-lim",
        min: 0.,
        max: 100.,
        step: 1.,
        default: 100.,
        unit: "dB",
    },
    ParamSpec {
        param: Param::PostFilter,
        key: "post-filter",
        min: 0.,
        max: 0.05,
        step: 0.001,
        default: 0.,
        unit: "",
    },
    ParamSpec {
        param: Param::MinThresh,
        key: "min-thresh",
        min: -15.,
        max: 35.,
        step: 1.,
        default: -10.,
        unit: "dB",
    },
    ParamSpec {
        param: Param::MaxErbThresh,
        key: "max-erb-thresh",
        min: -15.,
        max: 35.,
        step: 1.,
        default: 30.,
        unit: "dB",
    },
    ParamSpec {
        param: Param::MaxDfThresh,
        key: "max-df-thresh",
        min: -15.,
        max: 35.,
        step: 1.,
        default: 20.,
        unit: "dB",
    },
    ParamSpec {
        param: Param::MinBuffer,
        key: "min-buffer",
        min: 0.,
        max: 10.,
        step: 1.,
        // libDF's own default is zero: the adaptive rule already keeps one hop
        // and grows on demand, so anything more is reserve the user chose.
        default: 0.,
        unit: "hops",
    },
    ParamSpec {
        param: Param::MaskReduce,
        key: "mask-reduce",
        min: 0.,
        max: 2.,
        step: 1.,
        default: 0.,
        unit: "",
    },
];

/// `mask-reduce` values, in order: 0 = none, 1 = max, 2 = mean.
pub const MASK_REDUCE_NAMES: [&str; 3] = ["none", "max", "mean"];

impl Param {
    pub const ALL: [Self; 7] = [
        Self::AttenLim,
        Self::PostFilter,
        Self::MinThresh,
        Self::MaxErbThresh,
        Self::MaxDfThresh,
        Self::MinBuffer,
        Self::MaskReduce,
    ];

    pub fn spec(self) -> &'static ParamSpec {
        PARAMS
            .iter()
            .find(|s| s.param == self)
            .expect("every Param has a spec")
    }

    pub fn from_key(key: &str) -> Option<Self> {
        PARAMS.iter().find(|s| s.key == key).map(|s| s.param)
    }
}

/// Soft mode ("keep some room"): these values override the user's own while it
/// is on. Leaving soft mode restores the user's values exactly; while it is on,
/// `SetParam` on an overridden key is rejected with an error explaining why.
pub const SOFT_PRESET: [(Param, f64); 2] = [(Param::AttenLim, 12.), (Param::PostFilter, 0.)];

/// The interface. With zbus's `blocking-api` this generates `ControlProxy`
/// (async) and `ControlProxyBlocking`.
#[zbus::proxy(
    interface = "org.dfmic.Control1",
    default_service = "org.dfmic.Control",
    default_path = "/org/dfmic/Control"
)]
pub trait Control {
    // ---------------------------------------------------------------- methods

    /// Flip `Enabled`; returns the new value.
    fn toggle(&self) -> zbus::Result<bool>;
    fn set_enabled(&self, enabled: bool) -> zbus::Result<()>;
    /// Set one parameter of the current device's tuning, by `ParamSpec::key`.
    /// Out-of-range values are an error, not a clamp. Applies live.
    fn set_param(&self, key: &str, value: f64) -> zbus::Result<()>;
    /// Return the current device's tuning to `ParamSpec::default` (soft mode off).
    fn reset_params(&self) -> zbus::Result<()>;
    fn set_soft_mode(&self, on: bool) -> zbus::Result<()>;
    /// Switch capture to another device by node name, without dropping the
    /// virtual microphone or the takeover. Rejects the daemon's own node.
    fn set_source(&self, name: &str) -> zbus::Result<()>;
    fn set_agc(&self, on: bool) -> zbus::Result<()>;
    /// Ask PipeWire for a smaller graph quantum while filtering, and give it
    /// back when we stop. Off leaves the graph as it would be without dfmic.
    fn set_graph_quantum(&self, on: bool) -> zbus::Result<()>;
    /// Self-monitor. Enabling it is the caller's responsibility to confirm with
    /// the user when `MonitorTarget.kind` is not "headphones".
    fn set_monitor(&self, on: bool) -> zbus::Result<()>;
    fn set_take_default(&self, on: bool) -> zbus::Result<()>;
    /// Enable/disable the systemd user unit (start at login).
    fn set_autostart(&self, on: bool) -> zbus::Result<()>;
    /// Desktop notifications for state changes and failures.
    fn set_notify(&self, on: bool) -> zbus::Result<()>;
    /// Every setting back to its default, for every device (autostart excepted:
    /// that belongs to systemd, not to the settings file).
    fn reset_all(&self) -> zbus::Result<()>;
    /// Clear `Notices`.
    fn dismiss_notices(&self) -> zbus::Result<()>;
    /// Forget `LastFailure` (in memory and on disk).
    fn clear_failure(&self) -> zbus::Result<()>;
    /// Fetch the model into the data directory.
    ///
    /// Never automatic: this is the only way the daemon touches the network.
    /// Returns at once; `ModelDownloading` reports progress and `ModelPresent`
    /// the outcome. Downloading an already-present model is a no-op.
    fn download_model(&self) -> zbus::Result<()>;
    fn levels(&self) -> zbus::Result<Levels>;

    // ------------------------------------------------------------- properties

    /// `State::as_str`.
    #[zbus(property)]
    fn state(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn enabled(&self) -> zbus::Result<bool>;
    /// Something (an application or the self-monitor) is recording from us.
    #[zbus(property)]
    fn listening(&self) -> zbus::Result<bool>;

    /// Node name of the selected capture device (may be absent: see `State::NoDevice`).
    #[zbus(property)]
    fn source(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn source_description(&self) -> zbus::Result<String>;
    #[zbus(property)]
    fn source_available(&self) -> zbus::Result<bool>;
    /// The source setting is [`SOURCE_DEFAULT`]: follow the system default.
    #[zbus(property)]
    fn source_following(&self) -> zbus::Result<bool>;

    /// The model files are on disk, so filtering is possible.
    #[zbus(property)]
    fn model_present(&self) -> zbus::Result<bool>;
    /// The directory the daemon reads the model from and downloads into.
    ///
    /// Reported rather than assumed: `--models`, `DFMIC_MODELS` and the data
    /// directory are three different answers, and a client that guesses prints a
    /// path the files are not in.
    #[zbus(property)]
    fn model_path(&self) -> zbus::Result<String>;
    /// A `DownloadModel()` is in flight.
    #[zbus(property)]
    fn model_downloading(&self) -> zbus::Result<bool>;
    /// `DownloadModel()` could succeed: this build knows where to fetch from and
    /// the destination is writable. False means offering a button would be a
    /// dead end, so clients should explain instead.
    #[zbus(property)]
    fn model_download_possible(&self) -> zbus::Result<bool>;
    /// Live list, updated on hot-plug.
    #[zbus(property)]
    fn devices(&self) -> zbus::Result<Vec<Device>>;

    /// OpenVINO device the models are compiled for ("NPU").
    #[zbus(property)]
    fn compute_device(&self) -> zbus::Result<String>;
    /// Mean wall time of one hop through the NPU over the last second, in ms
    /// (the budget is 10 ms). 0 while the NPU is idle.
    #[zbus(property)]
    fn inference_ms(&self) -> zbus::Result<f64>;
    #[zbus(property)]
    fn inferences(&self) -> zbus::Result<u64>;
    /// Measured end-to-end latency added between the physical microphone and
    /// an application reading DeepFilter Mic, in ms: PipeWire graph latency on
    /// both streams + our buffering + algorithmic delay + inference time.
    /// Negative when it cannot be measured (nothing is listening).
    #[zbus(property)]
    fn latency_ms(&self) -> zbus::Result<f64>;
    /// The parts of `LatencyMs`, keyed "capture", "buffer", "model" and
    /// "output"; all in ms, and they sum to `LatencyMs`. Empty when
    /// unmeasurable. Inference time is reported separately (`InferenceMs`): it
    /// is absorbed by buffered audio rather than added to the delay.
    #[zbus(property)]
    fn latency_parts(&self) -> zbus::Result<HashMap<String, f64>>;
    /// Hops (10 ms) of captured audio dropped because the NPU fell behind.
    #[zbus(property)]
    fn overruns(&self) -> zbus::Result<u64>;
    /// Hops a consumer asked for that were not ready (silence inserted).
    #[zbus(property)]
    fn underruns(&self) -> zbus::Result<u64>;

    /// The current device's *own* tuning, keyed by `ParamSpec::key`. While
    /// `SoftMode` is on, the values in effect are these overlaid with `SOFT_PRESET`.
    #[zbus(property)]
    fn params(&self) -> zbus::Result<HashMap<String, f64>>;
    #[zbus(property)]
    fn soft_mode(&self) -> zbus::Result<bool>;
    #[zbus(property)]
    fn agc(&self) -> zbus::Result<bool>;
    #[zbus(property)]
    fn graph_quantum(&self) -> zbus::Result<bool>;
    #[zbus(property)]
    fn monitor(&self) -> zbus::Result<bool>;
    #[zbus(property)]
    fn monitor_target(&self) -> zbus::Result<MonitorTarget>;
    /// Setting: take over the default source while filtering.
    #[zbus(property)]
    fn take_default(&self) -> zbus::Result<bool>;
    /// Fact: DeepFilter Mic is the system default source right now (observed
    /// from PipeWire metadata, not assumed).
    #[zbus(property)]
    fn is_default(&self) -> zbus::Result<bool>;
    /// Fact: the systemd user unit is enabled.
    #[zbus(property)]
    fn autostart(&self) -> zbus::Result<bool>;
    #[zbus(property)]
    fn notify(&self) -> zbus::Result<bool>;
    /// Plain-language notes about settings corrected on load (out-of-range
    /// values, a device that is gone, ...). Cleared by `DismissNotices`.
    #[zbus(property)]
    fn notices(&self) -> zbus::Result<Vec<String>>;
    #[zbus(property)]
    fn last_failure(&self) -> zbus::Result<Failure>;
}
