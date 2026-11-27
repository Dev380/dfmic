//! Native zbus implementation of the daemon's `org.dfmic.Control1` contract.

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use zbus::fdo::{RequestNameFlags, RequestNameReply};
use zbus::zvariant::{DynamicType, OwnedValue, Value};

use crate::proto::{Device, Failure, Levels, MonitorTarget, BUS_NAME, INTERFACE, OBJECT_PATH};
use crate::pw::Shared;

#[derive(Clone)]
pub struct Published {
    pub state: String,
    pub enabled: bool,
    pub listening: bool,
    pub source: String,
    pub source_description: String,
    pub source_available: bool,
    /// The source setting is [`proto::SOURCE_DEFAULT`]: follow the system default.
    pub source_following: bool,
    /// The model files are on disk, so filtering is possible.
    pub model_present: bool,
    /// Where the daemon reads the model from and downloads into.
    pub model_path: String,
    /// A model download is in flight.
    pub model_downloading: bool,
    /// A download could succeed: a URL exists and the destination is writable.
    pub model_download_possible: bool,
    pub devices: Vec<Device>,
    pub compute_device: String,
    pub inference_ms: f64,
    pub inferences: u64,
    pub latency_ms: f64,
    pub latency_parts: HashMap<String, f64>,
    pub overruns: u64,
    pub underruns: u64,
    pub params: HashMap<String, f64>,
    pub soft_mode: bool,
    pub agc: bool,
    /// Whether dfmic asks PipeWire for a small graph quantum while filtering.
    pub graph_quantum: bool,
    pub monitor: bool,
    pub monitor_target: MonitorTarget,
    pub take_default: bool,
    pub is_default: bool,
    pub autostart: bool,
    pub notify: bool,
    pub notices: Vec<String>,
    pub last_failure: Failure,
}

impl Published {
    pub fn initial(compute_device: String) -> Self {
        Self {
            state: "starting".into(),
            // Matches `Settings::default()`: nothing filters until asked.
            enabled: false,
            listening: false,
            source: String::new(),
            source_description: String::new(),
            source_available: false,
            source_following: false,
            model_present: false,
            model_path: String::new(),
            model_downloading: false,
            model_download_possible: false,
            devices: Vec::new(),
            compute_device,
            inference_ms: 0.0,
            inferences: 0,
            latency_ms: -1.0,
            latency_parts: HashMap::new(),
            overruns: 0,
            underruns: 0,
            params: HashMap::new(),
            soft_mode: false,
            agc: true,
            graph_quantum: true,
            monitor: false,
            monitor_target: MonitorTarget {
                name: String::new(),
                description: String::new(),
                kind: "unknown".into(),
            },
            take_default: true,
            is_default: false,
            autostart: false,
            notify: true,
            notices: Vec::new(),
            last_failure: Failure {
                message: String::new(),
                when: 0,
                phase: String::new(),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum Command {
    Toggle,
    SetEnabled(bool),
    SetParam(String, f64),
    ResetParams,
    SetSoftMode(bool),
    SetSource(String),
    SetAgc(bool),
    SetGraphQuantum(bool),
    SetMonitor(bool),
    SetTakeDefault(bool),
    SetAutostart(bool),
    SetNotify(bool),
    ResetAll,
    DismissNotices,
    ClearFailure,
    /// Fetch the model. Never automatic; see the trait docs.
    DownloadModel,
    /// Not a request from anyone: a nudge to the event loop that something it
    /// watches has changed outside it (a PipeWire registry update, a finished
    /// download, a signal). Carries no reply and is never sent over the bus.
    Wake,
}

#[derive(Debug)]
pub enum Reply {
    None,
    Bool(bool),
}

pub struct Request {
    pub command: Command,
    reply: SyncSender<Result<Reply, String>>,
}

impl Request {
    pub fn succeed(self, reply: Reply) {
        let _ = self.reply.send(Ok(reply));
    }

    pub fn fail(self, message: String) {
        let _ = self.reply.send(Err(message));
    }
}

#[derive(Clone)]
struct ControlInterface {
    sender: Sender<Request>,
    published: Arc<RwLock<Published>>,
    shared: Arc<Shared>,
}

/// Nudge the daemon's event loop without asking it for anything.
///
/// The loop blocks on this channel rather than polling, so every asynchronous
/// source of change has to ring this bell: PipeWire's registry and metadata
/// callbacks, the worker, a finished download, and the signal handler.
pub fn wake(sender: &Sender<Request>) {
    let (reply, _) = mpsc::sync_channel(1);
    // Nobody is waiting for a reply; the loop drops it.
    let _ = sender.send(Request {
        command: Command::Wake,
        reply,
    });
}

pub fn request(sender: &Sender<Request>, command: Command) -> Result<Reply, String> {
    let (reply, result) = mpsc::sync_channel(1);
    sender
        .send(Request { command, reply })
        .map_err(|_| "The daemon is shutting down.".to_string())?;
    result
        .recv_timeout(Duration::from_secs(15))
        .map_err(|_| "The daemon did not answer the control request in time.".to_string())?
}

impl ControlInterface {
    fn request(&self, command: Command) -> zbus::fdo::Result<Reply> {
        request(&self.sender, command).map_err(zbus::fdo::Error::Failed)
    }

    fn unit(&self, command: Command) -> zbus::fdo::Result<()> {
        self.request(command).map(|_| ())
    }

    fn snapshot(&self) -> Published {
        self.published
            .read()
            .expect("published state lock poisoned")
            .clone()
    }
}

#[zbus::interface(name = "org.dfmic.Control1")]
impl ControlInterface {
    fn toggle(&self) -> zbus::fdo::Result<bool> {
        match self.request(Command::Toggle)? {
            Reply::Bool(enabled) => Ok(enabled),
            Reply::None => Err(zbus::fdo::Error::Failed("Invalid Toggle reply.".into())),
        }
    }

    fn set_enabled(&self, enabled: bool) -> zbus::fdo::Result<()> {
        self.unit(Command::SetEnabled(enabled))
    }

    fn set_param(&self, key: &str, value: f64) -> zbus::fdo::Result<()> {
        self.unit(Command::SetParam(key.into(), value))
    }

    fn reset_params(&self) -> zbus::fdo::Result<()> {
        self.unit(Command::ResetParams)
    }

    fn set_soft_mode(&self, on: bool) -> zbus::fdo::Result<()> {
        self.unit(Command::SetSoftMode(on))
    }

    fn set_source(&self, name: &str) -> zbus::fdo::Result<()> {
        self.unit(Command::SetSource(name.into()))
    }

    fn set_agc(&self, on: bool) -> zbus::fdo::Result<()> {
        self.unit(Command::SetAgc(on))
    }

    fn set_graph_quantum(&self, on: bool) -> zbus::fdo::Result<()> {
        self.unit(Command::SetGraphQuantum(on))
    }

    fn set_monitor(&self, on: bool) -> zbus::fdo::Result<()> {
        self.unit(Command::SetMonitor(on))
    }

    fn set_take_default(&self, on: bool) -> zbus::fdo::Result<()> {
        self.unit(Command::SetTakeDefault(on))
    }

    fn set_autostart(&self, on: bool) -> zbus::fdo::Result<()> {
        self.unit(Command::SetAutostart(on))
    }

    fn set_notify(&self, on: bool) -> zbus::fdo::Result<()> {
        self.unit(Command::SetNotify(on))
    }

    fn reset_all(&self) -> zbus::fdo::Result<()> {
        self.unit(Command::ResetAll)
    }

    fn dismiss_notices(&self) -> zbus::fdo::Result<()> {
        self.unit(Command::DismissNotices)
    }

    fn clear_failure(&self) -> zbus::fdo::Result<()> {
        self.unit(Command::ClearFailure)
    }

    fn download_model(&self) -> zbus::fdo::Result<()> {
        self.unit(Command::DownloadModel)
    }

    fn levels(&self) -> Levels {
        self.shared.levels()
    }

    #[zbus(property)]
    fn state(&self) -> String {
        self.snapshot().state
    }

    #[zbus(property)]
    fn enabled(&self) -> bool {
        self.snapshot().enabled
    }

    #[zbus(property)]
    fn listening(&self) -> bool {
        self.snapshot().listening
    }

    #[zbus(property)]
    fn source(&self) -> String {
        self.snapshot().source
    }

    #[zbus(property)]
    fn source_description(&self) -> String {
        self.snapshot().source_description
    }

    #[zbus(property)]
    fn source_available(&self) -> bool {
        self.snapshot().source_available
    }

    #[zbus(property)]
    fn source_following(&self) -> bool {
        self.snapshot().source_following
    }

    #[zbus(property)]
    fn model_present(&self) -> bool {
        self.snapshot().model_present
    }

    #[zbus(property)]
    fn model_path(&self) -> String {
        self.snapshot().model_path
    }

    #[zbus(property)]
    fn model_downloading(&self) -> bool {
        self.snapshot().model_downloading
    }

    #[zbus(property)]
    fn model_download_possible(&self) -> bool {
        self.snapshot().model_download_possible
    }

    #[zbus(property)]
    fn devices(&self) -> Vec<Device> {
        self.snapshot().devices
    }

    #[zbus(property)]
    fn compute_device(&self) -> String {
        self.snapshot().compute_device
    }

    #[zbus(property)]
    fn inference_ms(&self) -> f64 {
        self.snapshot().inference_ms
    }

    #[zbus(property)]
    fn inferences(&self) -> u64 {
        self.snapshot().inferences
    }

    #[zbus(property)]
    fn latency_ms(&self) -> f64 {
        self.snapshot().latency_ms
    }

    #[zbus(property)]
    fn latency_parts(&self) -> HashMap<String, f64> {
        self.snapshot().latency_parts
    }

    #[zbus(property)]
    fn overruns(&self) -> u64 {
        self.snapshot().overruns
    }

    #[zbus(property)]
    fn underruns(&self) -> u64 {
        self.snapshot().underruns
    }

    #[zbus(property)]
    fn params(&self) -> HashMap<String, f64> {
        self.snapshot().params
    }

    #[zbus(property)]
    fn soft_mode(&self) -> bool {
        self.snapshot().soft_mode
    }

    #[zbus(property)]
    fn agc(&self) -> bool {
        self.snapshot().agc
    }

    /// A smaller graph quantum means lower latency for every application, at a
    /// slightly higher CPU cost. Only requested while the filter is running.
    #[zbus(property)]
    fn graph_quantum(&self) -> bool {
        self.snapshot().graph_quantum
    }

    #[zbus(property)]
    fn monitor(&self) -> bool {
        self.snapshot().monitor
    }

    #[zbus(property)]
    fn monitor_target(&self) -> MonitorTarget {
        self.snapshot().monitor_target
    }

    #[zbus(property)]
    fn take_default(&self) -> bool {
        self.snapshot().take_default
    }

    #[zbus(property)]
    fn is_default(&self) -> bool {
        self.snapshot().is_default
    }

    #[zbus(property)]
    fn autostart(&self) -> bool {
        self.snapshot().autostart
    }

    #[zbus(property)]
    fn notify(&self) -> bool {
        self.snapshot().notify
    }

    #[zbus(property)]
    fn notices(&self) -> Vec<String> {
        self.snapshot().notices
    }

    #[zbus(property)]
    fn last_failure(&self) -> Failure {
        self.snapshot().last_failure
    }
}

fn owned<T>(value: T) -> OwnedValue
where
    T: Into<Value<'static>> + DynamicType,
{
    OwnedValue::try_from(Value::new(value)).expect("control properties contain only owned values")
}

const VOLATILE: [&str; 6] = [
    "LatencyMs",
    "LatencyParts",
    "InferenceMs",
    "Inferences",
    "Overruns",
    "Underruns",
];

pub struct ControlServer {
    connection: zbus::blocking::Connection,
    published: Arc<RwLock<Published>>,
    /// Volatile property changes waiting for the next slow publish.
    pending_volatile: Mutex<HashMap<String, OwnedValue>>,
}

impl ControlServer {
    /// Connect and take the bus name, without publishing anything yet.
    ///
    /// Split from [`Self::start`] so a second daemon is turned away *before* it
    /// touches the default source, the PipeWire graph or the NPU: all three are
    /// shared with the daemon already running.
    pub fn connect() -> Result<Self, String> {
        let connection = zbus::blocking::Connection::session()
            .map_err(|error| format!("could not connect to the session D-Bus: {error}"))?;
        Ok(Self {
            connection,
            published: Arc::new(RwLock::new(Published::initial(String::new()))),
            pending_volatile: Mutex::new(HashMap::new()),
        })
    }

    /// Take the bus name.
    ///
    /// Called *after* [`Self::start`] has exported the interface, never before:
    /// a name with nothing behind it is worse than no name at all. A client that
    /// calls during the gap gets no reply and waits for ever — which is what
    /// happened to the window when it was open before the daemon started, since
    /// the daemon claimed the name and then spent seconds on PipeWire setup
    /// before the object existed. Claiming last means the name appearing is the
    /// signal that the interface is ready.
    pub fn claim(self) -> Result<Self, String> {
        // Not `request_name`: zbus's *default* flags are `AllowReplacement |
        // ReplaceExisting | DoNotQueue`, so a second daemon would take the name
        // away from the running one and leave it alive but unreachable. Asking
        // for `DoNotQueue` alone makes the bus answer `Exists`, which zbus turns
        // into `Error::NameTaken`, and nobody can displace us afterwards.
        let reply = self
            .connection
            .request_name_with_flags(BUS_NAME, RequestNameFlags::DoNotQueue.into())
            .map_err(|error| {
                format!("could not own {BUS_NAME} (is another dfmic daemon running?): {error}")
            })?;
        if !matches!(
            reply,
            RequestNameReply::PrimaryOwner | RequestNameReply::AlreadyOwner
        ) {
            return Err(format!(
                "could not own {BUS_NAME} (is another dfmic daemon running?): {reply:?}"
            ));
        }
        Ok(self)
    }

    /// Export the control interface and adopt the initial published state.
    pub fn start(
        self,
        sender: Sender<Request>,
        shared: Arc<Shared>,
        initial: Published,
    ) -> Result<Self, String> {
        let interface = ControlInterface {
            sender,
            published: Arc::clone(&self.published),
            shared,
        };
        self.connection
            .object_server()
            .at(OBJECT_PATH, interface)
            .map_err(|error| format!("could not export {INTERFACE}: {error}"))?;
        *self
            .published
            .write()
            .expect("published state lock poisoned") = initial;
        Ok(self)
    }

    pub fn connection(&self) -> &zbus::blocking::Connection {
        &self.connection
    }

    /// Publish immediate setting changes and coalesce the six continuously
    /// changing diagnostic properties to the caller's 2 Hz cadence.
    pub fn publish(&self, next: Published, emit_volatile: bool) {
        let mut previous = self
            .published
            .write()
            .expect("published state lock poisoned");
        let mut immediate = HashMap::new();
        let mut volatile = HashMap::new();
        macro_rules! changed {
            ($field:ident, $name:literal) => {
                if previous.$field != next.$field {
                    if VOLATILE.contains(&$name) {
                        volatile.insert($name.to_string(), owned(next.$field.clone()));
                    } else {
                        immediate.insert($name.to_string(), owned(next.$field.clone()));
                    }
                }
            };
        }
        changed!(state, "State");
        changed!(enabled, "Enabled");
        changed!(listening, "Listening");
        changed!(source, "Source");
        changed!(source_description, "SourceDescription");
        changed!(source_available, "SourceAvailable");
        changed!(source_following, "SourceFollowing");
        // These three were missing from this list, so a model that appeared (or a
        // download that started) changed the published state without ever
        // telling a client: the window only found out on its periodic refresh.
        changed!(model_present, "ModelPresent");
        changed!(model_path, "ModelPath");
        changed!(model_downloading, "ModelDownloading");
        changed!(model_download_possible, "ModelDownloadPossible");
        changed!(devices, "Devices");
        changed!(compute_device, "ComputeDevice");
        changed!(inference_ms, "InferenceMs");
        changed!(inferences, "Inferences");
        changed!(latency_ms, "LatencyMs");
        changed!(latency_parts, "LatencyParts");
        changed!(overruns, "Overruns");
        changed!(underruns, "Underruns");
        changed!(params, "Params");
        changed!(soft_mode, "SoftMode");
        changed!(agc, "Agc");
        changed!(graph_quantum, "GraphQuantum");
        changed!(monitor, "Monitor");
        changed!(monitor_target, "MonitorTarget");
        changed!(take_default, "TakeDefault");
        changed!(is_default, "IsDefault");
        changed!(autostart, "Autostart");
        changed!(notify, "Notify");
        changed!(notices, "Notices");
        changed!(last_failure, "LastFailure");
        *previous = next;
        drop(previous);
        // Volatile properties are only emitted at the caller's slower cadence, so
        // a change that arrives on a fast publish has to be carried over rather
        // than dropped: `previous` has already absorbed it, so nothing would ever
        // report it again.
        let mut pending = self
            .pending_volatile
            .lock()
            .expect("pending state lock poisoned");
        pending.extend(volatile);
        if emit_volatile {
            immediate.extend(pending.drain());
        }
        drop(pending);
        if !immediate.is_empty() {
            let invalidated: Vec<String> = Vec::new();
            if let Err(error) = self.connection.emit_signal(
                None::<&str>,
                OBJECT_PATH,
                "org.freedesktop.DBus.Properties",
                "PropertiesChanged",
                &(INTERFACE, &immediate, &invalidated),
            ) {
                eprintln!("dfmic: could not emit PropertiesChanged: {error}");
            }
        }
    }
}

pub fn channel() -> (Sender<Request>, Receiver<Request>) {
    mpsc::channel()
}
