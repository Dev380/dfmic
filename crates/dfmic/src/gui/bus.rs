//! The window's connection to the daemon.
//!
//! Nothing here runs on the UI thread, including the connection itself: a slow
//! or missing session bus delays the first `Link`, never the window. Four kinds
//! of background work feed the window:
//!   * a connector that opens the session bus and starts the rest, so that the
//!     first frame is drawn while it is still connecting;
//!   * a refresher that reads every property in one `GetAll` whenever the daemon
//!     says something changed, appears or disappears (and every couple of
//!     seconds as a safety net), and hands the result to the UI thread. The
//!     `systemctl --user` facts behind it are cached, because the daemon being
//!     absent is the normal case and each of those is a fork;
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
    /// The model files are on disk, so filtering is possible.
    pub model_present: bool,
    /// A model download is in flight.
    pub model_downloading: bool,
    /// A download could succeed at all.
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

/// The unit's systemd facts, and when each was asked for.
///
/// Every one of them is a `systemctl --user` fork, and the refresher runs
/// whether or not the daemon exists — which is most of the time, since the
/// window is designed to work with it absent. Measured with the daemon down, the
/// uncached refresher forked twice a second for as long as the window was open.
/// None of these facts change except when this window or the user acts, so a
/// couple of seconds of staleness is invisible and a fork per fact per refresh
/// is not.
///
/// The TTLs differ because the facts do: `is-active` has to notice a daemon that
/// is compiling its models, while the unit file and the enablement only change
/// when someone installs or enables them. Anything that changes either is
/// invalidated outright through [`changes_systemd`], so the seconds after the
/// user presses Start are never answered from here.
#[derive(Default)]
struct FactCache {
    facts: Mutex<Facts>,
}

#[derive(Default)]
struct Facts {
    /// `systemctl --user cat`: does the unit exist at all?
    installed: Option<(Instant, bool)>,
    /// `systemctl --user is-active`: running or starting, bus name or not.
    active: Option<(Instant, bool)>,
    /// `systemctl --user is-enabled`; `None` in the entry means the query failed.
    autostart: Option<(Instant, Option<bool>)>,
}

/// How long `is-active` may stand. Long enough that a window left open with the
/// daemon down costs one fork every two seconds rather than eight.
const ACTIVE_TTL: Duration = Duration::from_secs(2);
/// How long the unit file and the enablement may stand: they only change when
/// the package is reinstalled or someone enables the unit, and both of those go
/// through the command runner.
const INSTALLED_TTL: Duration = Duration::from_secs(30);

impl FactCache {
    fn invalidate(&self) {
        let mut facts = self.facts.lock().unwrap_or_else(|e| e.into_inner());
        *facts = Facts::default();
    }

    /// The facts, re-asked only as often as each is allowed to be stale.
    ///
    /// The lock is held across the `systemctl` calls on purpose: the refresher is
    /// the only caller, and two of them forking the same query is exactly the
    /// waste this exists to remove.
    fn take(&self) -> (bool, bool, Option<bool>) {
        fn is_fresh<T>(taken: Option<(Instant, T)>, now: Instant, ttl: Duration) -> bool {
            taken.is_some_and(|(at, _)| now.duration_since(at) < ttl)
        }

        let mut facts = self.facts.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        if !is_fresh(facts.installed, now, INSTALLED_TTL) {
            facts.installed = Some((now, service::unit_installed()));
        }
        if !is_fresh(facts.active, now, ACTIVE_TTL) {
            facts.active = Some((now, service::is_active()));
        }
        if !is_fresh(facts.autostart, now, INSTALLED_TTL) {
            facts.autostart = Some((now, service::autostart_enabled().ok()));
        }
        // Each entry was just filled, so the unwraps are on values this call
        // wrote or on ones already fresh.
        let installed = facts.installed.map(|(_, v)| v).unwrap_or(false);
        let active = facts.active.map(|(_, v)| v).unwrap_or(false);
        let autostart = facts.autostart.and_then(|(_, v)| v);
        (installed, active, autostart)
    }
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
    /// Ask the daemon to fetch the model. It is the only thing that downloads.
    DownloadModel,
}

/// A command the daemon (or systemd) refused, in words for the user.
pub struct Refusal {
    pub command: Command,
    pub message: String,
}

pub struct Handles {
    pub commands: Sender<Command>,
    pub levels: Arc<Mutex<Option<(Instant, Levels)>>>,
    /// Set by the UI thread: would a level reading change anything on screen?
    ///
    /// The poller runs at 30 Hz, and 30 D-Bus round trips a second is real work
    /// for both processes. It is only worth it while the output meter is actually
    /// showing something — that is, while the daemon is filtering and something is
    /// listening — and while the window is on screen. `Levels()` is stateless, so
    /// not asking costs the daemon nothing either.
    pub want_levels: Arc<AtomicBool>,
}

/// Start the background work. `on_link` and `on_refusal` are called from
/// background threads; they must hand off to the UI thread themselves.
///
/// Returns at once: the session bus is connected on a thread of its own, so a
/// slow or missing bus delays the first `Link`, not the window. The first thing
/// the UI hears is therefore whatever `on_link` says once the connection has
/// been tried; until then the view shows its own "Connecting…".
pub fn start(
    on_link: impl Fn(Link) + Send + Sync + 'static,
    on_refusal: impl Fn(Refusal) + Send + Sync + 'static,
) -> Handles {
    let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();
    let levels = Arc::new(Mutex::new(None));
    let want_levels = Arc::new(AtomicBool::new(false));
    let handles = Handles {
        commands: cmd_tx,
        levels: Arc::clone(&levels),
        want_levels: Arc::clone(&want_levels),
    };

    let present = Arc::new(AtomicBool::new(false));
    let facts = Arc::new(FactCache::default());
    // The connect is a socket round trip and can fail on a machine with no bus;
    // none of that belongs in front of the first paint. If the thread cannot be
    // spawned at all — a thread table that is already exhausted — the window
    // still works as well as a bus client can without a connection, so it says
    // so instead of panicking.
    if let Err(error) = std::thread::Builder::new()
        .name("dfmic-connect".into())
        .spawn(move || {
            connect(
                present,
                facts,
                levels,
                want_levels,
                cmd_rx,
                on_link,
                on_refusal,
            )
        })
    {
        // Reported rather than silently swallowed: the tile will sit on
        // "Connecting…" for ever, and this is why.
        eprintln!("dfmic gui: could not spawn the bus connection thread: {error}");
    }
    handles
}

/// Connect to the session bus and start everything that needs it.
///
/// Runs on the connection thread: the connect is a socket round trip, and on a
/// machine with no session bus it is a failure that must not be shown by a
/// window that never appears.
fn connect(
    present: Arc<AtomicBool>,
    facts: Arc<FactCache>,
    levels: Arc<Mutex<Option<(Instant, Levels)>>>,
    want_levels: Arc<AtomicBool>,
    cmd_rx: Receiver<Command>,
    on_link: impl Fn(Link) + Send + Sync + 'static,
    on_refusal: impl Fn(Refusal) + Send + Sync + 'static,
) {
    let conn = match Connection::session() {
        Ok(c) => c,
        Err(e) => {
            on_link(Link::NoBus(format!("Cannot reach the session bus: {e}")));
            // Still run commands that do not need the bus (starting the service).
            run_commands(None, cmd_rx, &|_| {}, &on_refusal);
            return;
        }
    };

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
        let facts = Arc::clone(&facts);
        std::thread::Builder::new()
            .name("dfmic-refresh".into())
            .spawn(move || refresh_loop(&conn, &present, &facts, wake_rx, &on_link))
            .expect("spawning the refresh thread");
    }
    {
        let conn = conn.clone();
        let present = Arc::clone(&present);
        let want_levels = Arc::clone(&want_levels);
        std::thread::Builder::new()
            .name("dfmic-levels".into())
            .spawn(move || levels_loop(&conn, &present, &want_levels, &levels))
            .expect("spawning the level thread");
    }
    {
        let conn = conn.clone();
        std::thread::Builder::new()
            .name("dfmic-commands".into())
            .spawn(move || {
                let wake = move |command: &Command| {
                    if changes_systemd(command) {
                        // The answer the cache holds is now known to be wrong.
                        facts.invalidate();
                    }
                    let _ = wake_tx.send(());
                };
                run_commands(Some(&conn), cmd_rx, &wake, &on_refusal)
            })
            .expect("spawning the command thread");
    }
}

/// Whether a command can change what `systemctl --user` would answer.
///
/// These two are the reason the cache can be trusted for long: the window is
/// the only thing that starts the unit or changes autostart, and when it does,
/// it says so here instead of waiting out the TTL.
fn changes_systemd(command: &Command) -> bool {
    matches!(command, Command::StartService | Command::SetAutostart(_))
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
    facts: &FactCache,
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
        let link = look(&bus, &props, facts);
        present.store(matches!(link, Link::Present(_)), Ordering::SeqCst);
        // While absent there are no signals to wait for besides the name
        // appearing, but systemd's view (starting, autostart) can still change.
        // The wait only decides when to *look* again: the systemd facts behind
        // it are cached for longer (see [`FactCache`]), so this cadence costs one
        // D-Bus round trip, not four forks.
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

fn look(bus: &DBusProxy<'_>, props: &PropertiesProxy<'_>, facts: &FactCache) -> Link {
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
                Link::Absent(absent(facts))
            }
        },
        Ok(false) => Link::Absent(absent(facts)),
        Err(e) => Link::NoBus(e),
    }
}

fn absent(facts: &FactCache) -> Absent {
    let (unit_installed, unit_active, autostart) = facts.take();
    Absent {
        unit_installed,
        unit_active,
        autostart,
        // Not cached: the file is written by a daemon that is not on the bus, and
        // a read of a small file costs nothing next to a point of truth that has
        // to be current the moment the daemon dies.
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
        model_present: take(&mut m, "ModelPresent")?,
        model_downloading: take(&mut m, "ModelDownloading")?,
        model_download_possible: take(&mut m, "ModelDownloadPossible")?,
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

fn levels_loop(
    conn: &Connection,
    present: &AtomicBool,
    want_levels: &AtomicBool,
    slot: &Mutex<Option<(Instant, Levels)>>,
) {
    let proxy = match control(conn) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("dfmic gui: no level meters: {e}");
            return;
        }
    };
    /// While the window is on screen: one poll per frame.
    const FRAME: Duration = Duration::from_millis(33);
    /// While it is not: slow enough to cost nothing, quick enough that the meters
    /// are alive again within a couple of frames of the window coming back.
    const IDLE: Duration = Duration::from_millis(500);
    loop {
        let started = Instant::now();
        let wanted = want_levels.load(Ordering::Relaxed);
        let period = if wanted { FRAME } else { IDLE };
        if wanted && present.load(Ordering::SeqCst) {
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
        Command::DownloadModel => proxy.download_model(),
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
