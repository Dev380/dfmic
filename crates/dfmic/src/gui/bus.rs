//! The window's connection to the daemon.
//!
//! Nothing here runs on the UI thread. Three kinds of background work feed the
//! window:
//!   * a refresher that reads every property in one `GetAll` whenever the daemon
//!     says something changed, appears or disappears (and every couple of
//!     seconds as a safety net), and hands the result to the UI thread;
//!   * a level poller that calls the stateless `Levels()` at about 30 Hz while
//!     the daemon is present;
//!   * a command runner that performs actions in order and reports failures.
//!
//! Signals are matched by path and well-known name rather than through a proxy
//! bound to one daemon instance, so the window keeps working when the daemon
//! restarts underneath it.

use std::collections::HashMap;
use std::fmt::Display;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use zbus::blocking::fdo::{DBusProxy, PropertiesProxy};
use zbus::blocking::{Connection, MessageIterator};
use zbus::names::InterfaceName;
use zbus::zvariant::OwnedValue;
use zbus::MatchRule;

use crate::proto::{self, ControlProxyBlocking, Device, Failure, Levels, MonitorTarget};
use crate::service;

/// Everything the daemon publishes, read in one go.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub state: String,
    pub enabled: bool,
    pub listening: bool,
    pub source: String,
    pub source_description: String,
    pub source_available: bool,
    /// The source setting is "default": follow the system default.
    pub source_following: bool,
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

/// What can be known while the daemon is not on the bus.
#[derive(Clone, Debug)]
pub struct Absent {
    pub unit_installed: bool,
    /// systemd says the unit is running (or starting) even though the bus name
    /// is not there yet: the daemon is compiling models.
    pub unit_active: bool,
    pub autostart: Option<bool>,
    pub failure: Option<Failure>,
}

#[derive(Clone, Debug)]
pub enum Link {
    /// The first look has not completed yet.
    Connecting,
    /// No session bus at all; nothing can work.
    NoBus(String),
    Absent(Absent),
    Present(Box<Snapshot>),
}

/// Actions the window asks for. Each maps onto one method of the interface
/// (or, while the daemon is down, onto the `service` helpers the CLI uses too).
#[derive(Clone, Debug)]
pub enum Command {
    SetEnabled(bool),
    SetParam(&'static str, f64),
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
    StartService,
}

/// A command the daemon (or systemd) refused, in words for the user.
pub struct Refusal {
    pub command: Command,
    pub message: String,
}

pub struct Handles {
    pub commands: Sender<Command>,
    pub levels: Arc<Mutex<Option<(Instant, Levels)>>>,
}

/// Start the background work. `on_link` and `on_refusal` are called from
/// background threads; they must hand off to the UI thread themselves.
pub fn start(
    on_link: impl Fn(Link) + Send + Sync + 'static,
    on_refusal: impl Fn(Refusal) + Send + Sync + 'static,
) -> Handles {
    let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();
    let levels = Arc::new(Mutex::new(None));
    let handles = Handles {
        commands: cmd_tx,
        levels: Arc::clone(&levels),
    };

    let conn = match Connection::session() {
        Ok(c) => c,
        Err(e) => {
            on_link(Link::NoBus(format!("Cannot reach the session bus: {e}")));
            // Still run commands that do not need the bus (starting the service).
            std::thread::spawn(move || run_commands(None, cmd_rx, &|_| {}, &on_refusal));
            return handles;
        }
    };

    let present = Arc::new(AtomicBool::new(false));
    let (wake_tx, wake_rx) = mpsc::channel::<()>();

    // Wake the refresher on any property change from the daemon...
    spawn_signal_watch(
        &conn,
        MatchRule::builder()
            .msg_type(zbus::message::Type::Signal)
            .interface("org.freedesktop.DBus.Properties")
            .and_then(|b| b.member("PropertiesChanged"))
            .and_then(|b| b.path(proto::OBJECT_PATH))
            .map(|b| b.build()),
        wake_tx.clone(),
    );
    // ...and whenever it appears on or leaves the bus.
    spawn_signal_watch(
        &conn,
        MatchRule::builder()
            .msg_type(zbus::message::Type::Signal)
            .sender("org.freedesktop.DBus")
            .and_then(|b| b.interface("org.freedesktop.DBus"))
            .and_then(|b| b.member("NameOwnerChanged"))
            .and_then(|b| b.arg(0, proto::BUS_NAME))
            .map(|b| b.build()),
        wake_tx.clone(),
    );

    {
        let conn = conn.clone();
        let present = Arc::clone(&present);
        std::thread::Builder::new()
            .name("dfmic-refresh".into())
            .spawn(move || refresh_loop(&conn, &present, wake_rx, &on_link))
            .expect("spawning the refresh thread");
    }
    {
        let conn = conn.clone();
        let present = Arc::clone(&present);
        std::thread::Builder::new()
            .name("dfmic-levels".into())
            .spawn(move || levels_loop(&conn, &present, &levels))
            .expect("spawning the level thread");
    }
    {
        let conn = conn.clone();
        std::thread::Builder::new()
            .name("dfmic-commands".into())
            .spawn(move || {
                let wake = move |_: &Command| {
                    let _ = wake_tx.send(());
                };
                run_commands(Some(&conn), cmd_rx, &wake, &on_refusal)
            })
            .expect("spawning the command thread");
    }
    handles
}

fn spawn_signal_watch(conn: &Connection, rule: zbus::Result<MatchRule<'static>>, wake: Sender<()>) {
    let conn = conn.clone();
    std::thread::Builder::new()
        .name("dfmic-signals".into())
        .spawn(move || {
            let iter = rule.and_then(|rule| MessageIterator::for_match_rule(rule, &conn, Some(64)));
            match iter {
                Ok(iter) => {
                    for _ in iter {
                        if wake.send(()).is_err() {
                            return;
                        }
                    }
                }
                // Without change signals the periodic refresh still keeps the
                // window correct, only slower; say so on stderr.
                Err(e) => eprintln!("dfmic gui: cannot subscribe to daemon signals: {e}"),
            }
        })
        .expect("spawning a signal thread");
}

fn refresh_loop(
    conn: &Connection,
    present: &AtomicBool,
    wake: Receiver<()>,
    on_link: &(dyn Fn(Link) + Send + Sync),
) {
    let bus = DBusProxy::new(conn);
    let props = PropertiesProxy::builder(conn)
        .destination(proto::BUS_NAME)
        .and_then(|b| b.path(proto::OBJECT_PATH))
        .and_then(|b| b.cache_properties(zbus::proxy::CacheProperties::No).build());
    let (bus, props) = match (bus, props) {
        (Ok(b), Ok(p)) => (b, p),
        (Err(e), _) | (_, Err(e)) => {
            on_link(Link::NoBus(format!("Cannot talk to the session bus: {e}")));
            return;
        }
    };

    loop {
        let link = look(&bus, &props);
        present.store(matches!(link, Link::Present(_)), Ordering::SeqCst);
        // While absent there are no signals to wait for besides the name
        // appearing, but systemd's view (starting, autostart) can still change.
        let period = if matches!(link, Link::Present(_)) {
            Duration::from_secs(3)
        } else {
            Duration::from_millis(1500)
        };
        on_link(link);

        match wake.recv_timeout(period) {
            Ok(()) => {
                // Coalesce bursts (a slider drag produces many changes).
                std::thread::sleep(Duration::from_millis(25));
                while wake.try_recv().is_ok() {}
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn look(bus: &DBusProxy<'_>, props: &PropertiesProxy<'_>) -> Link {
    let owned = proto::BUS_NAME
        .try_into()
        .map_err(|e: zbus::names::Error| e.to_string())
        .and_then(|name| bus.name_has_owner(name).map_err(|e| e.to_string()));
    match owned {
        Ok(true) => match read_snapshot(props) {
            Ok(snapshot) => Link::Present(Box::new(snapshot)),
            // The daemon vanished between the two calls, or is mid-start.
            Err(e) => {
                eprintln!("dfmic gui: reading daemon state: {e}");
                Link::Absent(absent())
            }
        },
        Ok(false) => Link::Absent(absent()),
        Err(e) => Link::NoBus(e),
    }
}

fn absent() -> Absent {
    Absent {
        unit_installed: service::unit_installed(),
        unit_active: service::is_active(),
        autostart: service::autostart_enabled().ok(),
        failure: service::last_failure(),
    }
}

fn take<T>(map: &mut HashMap<String, OwnedValue>, key: &str) -> Result<T, String>
where
    T: TryFrom<OwnedValue>,
    T::Error: Display,
{
    let value = map
        .remove(key)
        .ok_or_else(|| format!("the daemon does not publish {key}"))?;
    T::try_from(value).map_err(|e| format!("{key}: {e}"))
}

fn read_snapshot(props: &PropertiesProxy<'_>) -> Result<Snapshot, String> {
    let interface = InterfaceName::from_static_str(proto::INTERFACE).map_err(|e| e.to_string())?;
    let mut m = props.get_all(interface).map_err(|e| e.to_string())?;
    Ok(Snapshot {
        state: take(&mut m, "State")?,
        enabled: take(&mut m, "Enabled")?,
        listening: take(&mut m, "Listening")?,
        source: take(&mut m, "Source")?,
        source_description: take(&mut m, "SourceDescription")?,
        source_available: take(&mut m, "SourceAvailable")?,
        source_following: take(&mut m, "SourceFollowing")?,
        devices: take(&mut m, "Devices")?,
        compute_device: take(&mut m, "ComputeDevice")?,
        inference_ms: take(&mut m, "InferenceMs")?,
        inferences: take(&mut m, "Inferences")?,
        latency_ms: take(&mut m, "LatencyMs")?,
        latency_parts: take(&mut m, "LatencyParts")?,
        overruns: take(&mut m, "Overruns")?,
        underruns: take(&mut m, "Underruns")?,
        params: take(&mut m, "Params")?,
        soft_mode: take(&mut m, "SoftMode")?,
        agc: take(&mut m, "Agc")?,
        graph_quantum: take(&mut m, "GraphQuantum")?,
        monitor: take(&mut m, "Monitor")?,
        monitor_target: take(&mut m, "MonitorTarget")?,
        take_default: take(&mut m, "TakeDefault")?,
        is_default: take(&mut m, "IsDefault")?,
        autostart: take(&mut m, "Autostart")?,
        notify: take(&mut m, "Notify")?,
        notices: take(&mut m, "Notices")?,
        last_failure: take(&mut m, "LastFailure")?,
    })
}

fn control(conn: &Connection) -> zbus::Result<ControlProxyBlocking<'static>> {
    ControlProxyBlocking::builder(conn)
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
}

fn levels_loop(conn: &Connection, present: &AtomicBool, slot: &Mutex<Option<(Instant, Levels)>>) {
    let proxy = match control(conn) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("dfmic gui: no level meters: {e}");
            return;
        }
    };
    let period = Duration::from_millis(33);
    loop {
        let started = Instant::now();
        if present.load(Ordering::SeqCst) {
            // A failed call means the daemon is going away; the refresher sees
            // that through NameOwnerChanged, and the meters go stale on their own.
            if let Ok(levels) = proxy.levels() {
                if let Ok(mut slot) = slot.lock() {
                    *slot = Some((Instant::now(), levels));
                }
            }
        }
        std::thread::sleep(period.saturating_sub(started.elapsed()));
    }
}

fn run_commands(
    conn: Option<&Connection>,
    rx: Receiver<Command>,
    after: &dyn Fn(&Command),
    on_refusal: &(dyn Fn(Refusal) + Send + Sync),
) {
    let proxy = conn.map(control).transpose();
    let proxy = match proxy {
        Ok(p) => p,
        Err(e) => {
            eprintln!("dfmic gui: cannot create the control proxy: {e}");
            None
        }
    };
    while let Ok(first) = rx.recv() {
        // Drain what has queued up and drop superseded parameter changes, so a
        // fast slider drag sends its latest position rather than its history.
        let mut batch = vec![first];
        batch.extend(rx.try_iter());
        let mut keep = vec![true; batch.len()];
        for i in 0..batch.len() {
            if let Command::SetParam(key, _) = batch[i] {
                if batch[i + 1..]
                    .iter()
                    .any(|later| matches!(later, Command::SetParam(k, _) if *k == key))
                {
                    keep[i] = false;
                }
            }
        }
        for (command, keep) in batch.into_iter().zip(keep) {
            if !keep {
                continue;
            }
            if let Err(message) = execute(proxy.as_ref(), &command) {
                on_refusal(Refusal {
                    command: command.clone(),
                    message,
                });
            }
            after(&command);
        }
    }
}

fn execute(proxy: Option<&ControlProxyBlocking<'_>>, command: &Command) -> Result<(), String> {
    // These work whether or not the daemon is running.
    match command {
        Command::StartService => return service::start(),
        Command::SetAutostart(on) if proxy.is_none() => return service::set_autostart(*on),
        _ => {}
    }
    let proxy = proxy.ok_or("DeepFilter Mic is not running.")?;
    let result = match command {
        Command::SetEnabled(on) => proxy.set_enabled(*on),
        Command::SetParam(key, value) => proxy.set_param(key, *value),
        Command::ResetParams => proxy.reset_params(),
        Command::SetSoftMode(on) => proxy.set_soft_mode(*on),
        Command::SetSource(name) => proxy.set_source(name),
        Command::SetAgc(on) => proxy.set_agc(*on),
        Command::SetGraphQuantum(on) => proxy.set_graph_quantum(*on),
        Command::SetMonitor(on) => proxy.set_monitor(*on),
        Command::SetTakeDefault(on) => proxy.set_take_default(*on),
        Command::SetAutostart(on) => {
            // The daemon may be gone by now; systemd does not need it.
            return match proxy.set_autostart(*on) {
                Err(zbus::Error::MethodError(name, _, _))
                    if name.as_str() == "org.freedesktop.DBus.Error.ServiceUnknown" =>
                {
                    service::set_autostart(*on)
                }
                other => other.map_err(describe),
            };
        }
        Command::SetNotify(on) => proxy.set_notify(*on),
        Command::ResetAll => proxy.reset_all(),
        Command::DismissNotices => proxy.dismiss_notices(),
        Command::ClearFailure => proxy.clear_failure(),
        Command::StartService => unreachable!("handled above"),
    };
    result.map_err(describe)
}

/// The daemon's error messages are written for people; pass them through and
/// only rephrase transport failures.
fn describe(error: zbus::Error) -> String {
    match error {
        zbus::Error::MethodError(name, Some(message), _) if !message.is_empty() => {
            if name.as_str() == "org.freedesktop.DBus.Error.ServiceUnknown" {
                "DeepFilter Mic is not running.".into()
            } else {
                message
            }
        }
        zbus::Error::MethodError(name, _, _) => format!("The daemon refused: {name}"),
        other => format!("Could not reach DeepFilter Mic: {other}"),
    }
}
