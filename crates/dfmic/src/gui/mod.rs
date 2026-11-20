//! The control window: a view onto the daemon, never a second owner of state.
//!
//! The daemon's published state (see `proto`) is the truth. The only thing the
//! window adds is *optimism*: when the user flips a control the window shows the
//! new value at once, marks it as pending, and drops the override as soon as
//! the daemon confirms it, reports something else, or a short deadline passes.
//! A refused action reverts immediately and says why.

mod bus;
mod meters;
mod record;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

use slint::{ComponentHandle, ModelRc, SharedString, Timer, TimerMode, VecModel};

use crate::proto::{self, Device, Param, State, PARAMS, SOFT_PRESET};
use bus::{Absent, Command, Link, Refusal, Snapshot};
use meters::{Ballistics, Reading};

slint::include_modules!();

/// How long an optimistic value may stand without the daemon confirming it.
const PENDING_DEADLINE: Duration = Duration::from_secs(3);
/// Levels older than this mean the poller has stopped getting answers.
const LEVELS_STALE: Duration = Duration::from_millis(400);
/// Meter frame rate: a steady 30 fps.
///
/// Slowing this down when the meters settle would save CPU, but a meter that
/// jumps in eighth-of-a-second steps looks broken, so the rate stays fixed and
/// the cost is taken out of the renderer instead (see `select_renderer`).
const METER_FRAME: Duration = Duration::from_millis(33);
const TOAST_TIME: Duration = Duration::from_secs(5);
const TOAST_TIME_WITH_ACTION: Duration = Duration::from_secs(9);
/// Above this share of the 10 ms budget the NPU is flagged as struggling.
const NPU_STRAIN_MS: f64 = 8.0;

pub fn run() -> Result<(), String> {
    // The app id is stored on the platform context, which only exists once a
    // backend has been selected, so select it before naming the window.
    select_renderer().select().map_err(|e| e.to_string())?;
    // Matches the desktop entry, so the compositor shows the right name and icon.
    slint::set_xdg_app_id("dfmic").map_err(|e| e.to_string())?;
    let window = MainWindow::new().map_err(|e| e.to_string())?;
    let view = Rc::new(RefCell::new(View::new(window.as_weak())));

    let link_target = window.as_weak();
    let refusal_target = window.as_weak();
    let views = Rc::downgrade(&view);
    // The view lives on the UI thread; background threads reach it by posting
    // closures to the event loop, which look it up through this thread-local.
    VIEW.with(|slot| *slot.borrow_mut() = Some(views));

    let handles = bus::start(
        move |link| {
            let _ = link_target.upgrade_in_event_loop(move |_| with_view(|v| v.on_link(link)));
        },
        move |refusal| {
            let _ =
                refusal_target.upgrade_in_event_loop(move |_| with_view(|v| v.on_refusal(refusal)));
        },
    );
    view.borrow_mut().commands = Some(handles.commands);
    wire_callbacks(&window);
    // Fixed for the life of the binary: ranges, defaults and the soft preset.
    let ui = window.global::<Ui>();
    ui.set_specs(ModelRc::new(VecModel::from(
        PARAMS
            .iter()
            .map(|s| ParamSpec {
                key: s.key.into(),
                minimum: s.min as f32,
                maximum: s.max as f32,
                step: s.step as f32,
                default_value: s.default as f32,
            })
            .collect::<Vec<_>>(),
    )));
    ui.set_soft_summary(soft_summary().into());
    view.borrow().render();

    // Meters: a steady 30 fps on the UI thread, reading whatever the poller
    // fetched last. Properties only change (and the window only redraws) when
    // the picture actually moves.
    let meter_timer = Timer::default();
    {
        let levels = handles.levels;
        let mut last = Instant::now();
        meter_timer.start(TimerMode::Repeated, METER_FRAME, move || {
            let now = Instant::now();
            let dt = now.duration_since(last).as_secs_f32().min(0.25);
            last = now;
            let sample = levels
                .lock()
                .ok()
                .and_then(|slot| *slot)
                .filter(|(at, _)| now.duration_since(*at) < LEVELS_STALE)
                .map(|(_, l)| l);
            with_view(|v| {
                v.poll_recorder();
                v.poll_playback();
                v.tick_meters(now, dt, sample);
            });
        });
    }

    window.run().map_err(|e| e.to_string())
}

thread_local! {
    static VIEW: RefCell<Option<std::rc::Weak<RefCell<View>>>> = const { RefCell::new(None) };
}

/// Choose a renderer, falling back to Slint's software rasteriser when OpenGL
/// would be software too.
///
/// Slint's OpenGL renderer (femtovg) repaints the whole window for every frame
/// that changes, so its cost is the window area times the frame rate. On a GPU
/// that is a fraction of a millisecond a frame (measured here at 0.3–0.5 ms for
/// the 921x1176 physical pixels this window gets on a 1.6x display). Through a
/// CPU-side OpenGL implementation it is 3.5x worse than Slint's software
/// renderer, which tracks dirty regions instead of repainting everything:
/// forced to 60 fps, femtovg cost 573 ms of CPU per second against 163 ms for
/// the software renderer, so a software-GL session would burn a third of a core
/// just drawing the meters.
///
/// Hence: use the GPU when there is one, and the software renderer when there
/// is not. `SLINT_BACKEND` overrides either way, as it does for any Slint
/// program.
fn select_renderer() -> slint::BackendSelector {
    let selector = slint::BackendSelector::new();
    if std::env::var_os("SLINT_BACKEND").is_some() {
        // An explicit choice from the environment wins, including an empty
        // value, which asks Slint to pick for itself.
        return selector;
    }
    if gl_would_be_software() {
        selector.renderer_name("software".into())
    } else {
        selector
    }
}

/// Whether OpenGL on this machine is a CPU rasteriser.
///
/// Two signals, both cheap and both checked before any window exists: an
/// explicit request for software rendering, and the absence of a DRM render
/// node, which means there is no GPU driver to draw with at all.
///
/// This deliberately does not try to catch a GPU that Mesa refuses to drive and
/// silently falls back from; that would need an EGL context to ask, which is the
/// very thing being avoided. `SLINT_BACKEND=winit-software` covers that case.
fn gl_would_be_software() -> bool {
    let env = |key: &str| std::env::var(key).unwrap_or_default().to_lowercase();
    if matches!(env("LIBGL_ALWAYS_SOFTWARE").as_str(), "1" | "true" | "yes") {
        return true;
    }
    if matches!(
        env("MESA_LOADER_DRIVER_OVERRIDE").as_str(),
        "llvmpipe" | "softpipe" | "swrast" | "kms_swrast"
    ) {
        return true;
    }
    !std::fs::read_dir("/dev/dri")
        .map(|entries| {
            entries.filter_map(Result::ok).any(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with("renderD"))
            })
        })
        .unwrap_or(false)
}

/// A take left by a previous run, so the play buttons work from the start.
fn existing_take() -> Option<std::path::PathBuf> {
    let dir = record::take_dir();
    let has = dir.join("raw.wav").is_file() && dir.join("filtered.wav").is_file();
    has.then_some(dir)
}

fn with_view(f: impl FnOnce(&mut View)) {
    let view = VIEW.with(|slot| slot.borrow().as_ref().and_then(|w| w.upgrade()));
    if let Some(view) = view {
        f(&mut view.borrow_mut());
    }
}

fn wire_callbacks(window: &MainWindow) {
    let ui = window.global::<Ui>();
    ui.on_toggle(|| with_view(|v| v.toggle()));
    ui.on_start_service(|| with_view(|v| v.start_service()));
    ui.on_set_param(|index, value| with_view(|v| v.set_param(index as usize, value as f64)));
    ui.on_reset_param(|index| {
        with_view(|v| {
            if let Some(spec) = PARAMS.get(index as usize) {
                v.set_param(index as usize, spec.default);
            }
        })
    });
    ui.on_reset_params(|| with_view(|v| v.reset_params()));
    ui.on_set_soft(|on| with_view(|v| v.set_flag(Flag::Soft, on)));
    ui.on_set_source(|name| with_view(|v| v.set_source(name.to_string())));
    ui.on_set_agc(|on| with_view(|v| v.set_flag(Flag::Agc, on)));
    ui.on_set_graph_quantum(|on| with_view(|v| v.set_flag(Flag::GraphQuantum, on)));
    ui.on_set_monitor(|on| with_view(|v| v.set_flag(Flag::Monitor, on)));
    ui.on_set_take_default(|on| with_view(|v| v.set_flag(Flag::TakeDefault, on)));
    ui.on_set_autostart(|on| with_view(|v| v.set_autostart(on)));
    ui.on_set_notify(|on| with_view(|v| v.set_flag(Flag::Notify, on)));
    ui.on_reset_all(|| with_view(|v| v.reset_all()));
    ui.on_dismiss_notices(|| with_view(|v| v.send(Command::DismissNotices)));
    ui.on_clear_failure(|| with_view(|v| v.send(Command::ClearFailure)));
    ui.on_copied(|| with_view(|v| v.show_toast("Node name copied.".into(), None)));
    ui.on_reset_clip(|output| with_view(|v| v.reset_clip(output)));
    ui.on_toggle_recording(|| with_view(|v| v.toggle_recording()));
    ui.on_play_take(|which| with_view(|v| v.play_take(which.as_str())));
    ui.on_toast_activate(|| with_view(|v| v.toast_activate()));
    ui.on_toast_dismiss(|| with_view(|v| v.hide_toast()));
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Flag {
    Enabled,
    Soft,
    Agc,
    GraphQuantum,
    Monitor,
    TakeDefault,
    Autostart,
    Notify,
}

impl Flag {
    fn command(self, on: bool) -> Command {
        match self {
            Flag::Enabled => Command::SetEnabled(on),
            Flag::Soft => Command::SetSoftMode(on),
            Flag::Agc => Command::SetAgc(on),
            Flag::GraphQuantum => Command::SetGraphQuantum(on),
            Flag::Monitor => Command::SetMonitor(on),
            Flag::TakeDefault => Command::SetTakeDefault(on),
            Flag::Autostart => Command::SetAutostart(on),
            Flag::Notify => Command::SetNotify(on),
        }
    }

    fn of(self, s: &Snapshot) -> bool {
        match self {
            Flag::Enabled => s.enabled,
            Flag::Soft => s.soft_mode,
            Flag::Agc => s.agc,
            Flag::GraphQuantum => s.graph_quantum,
            Flag::Monitor => s.monitor,
            Flag::TakeDefault => s.take_default,
            Flag::Autostart => s.autostart,
            Flag::Notify => s.notify,
        }
    }
}

/// A value shown before the daemon has confirmed it.
struct Pending<T> {
    value: T,
    until: Instant,
}

impl<T> Pending<T> {
    fn new(value: T) -> Self {
        Self {
            value,
            until: Instant::now() + PENDING_DEADLINE,
        }
    }
}

/// What a toast's button does.
enum ToastAction {
    RestoreParams {
        params: HashMap<String, f64>,
        soft: bool,
    },
    Autostart(bool),
}

struct View {
    window: slint::Weak<MainWindow>,
    commands: Option<std::sync::mpsc::Sender<Command>>,
    link: Link,
    starting: bool,

    flags: HashMap<Flag, Pending<bool>>,
    params: HashMap<usize, Pending<f64>>,
    source: Option<Pending<String>>,

    input: Ballistics,
    output: Ballistics,
    last_meters: Option<(MeterView, MeterView, SharedString, bool, bool)>,

    toast_action: Option<ToastAction>,
    toast_timer: Timer,
    /// A take in progress. The window owns it: the daemon is not involved.
    recorder: Option<record::Recorder>,
    /// Where the last take was written, so it can be played back.
    take_dir: Option<std::path::PathBuf>,
    /// The comparison track being played, if any. The child is polled by the
    /// meter timer: `pw-play` exits when the file ends, which is the only
    /// signal needed to put the buttons back.
    playing: Option<(String, std::process::Child)>,
}

impl View {
    fn new(window: slint::Weak<MainWindow>) -> Self {
        Self {
            window,
            commands: None,
            link: Link::Connecting,
            starting: false,
            flags: HashMap::new(),
            params: HashMap::new(),
            source: None,
            input: Ballistics::new(),
            output: Ballistics::new(),
            last_meters: None,
            toast_action: None,
            toast_timer: Timer::default(),
            recorder: None,
            take_dir: existing_take(),
            playing: None,
        }
    }

    fn snapshot(&self) -> Option<&Snapshot> {
        match &self.link {
            Link::Present(s) => Some(s),
            _ => None,
        }
    }

    fn send(&self, command: Command) {
        if let Some(tx) = &self.commands {
            // The command thread only stops when the process does.
            let _ = tx.send(command);
        }
    }

    // ------------------------------------------------------------ incoming

    fn on_link(&mut self, link: Link) {
        if !matches!(link, Link::Absent(_) | Link::Connecting) {
            self.starting = false;
        }
        if let Link::Absent(a) = &link {
            if !a.unit_active && self.starting && matches!(self.link, Link::Absent(_)) {
                // systemd gave up on the start we asked for; the failure (if
                // any) is on screen from `last-failure`.
                self.starting = false;
            }
        }
        self.link = link;
        self.settle_pending();
        self.render();
    }

    fn on_refusal(&mut self, refusal: Refusal) {
        // Drop the optimistic value so the daemon's truth shows again.
        match &refusal.command {
            Command::SetEnabled(_) => drop(self.flags.remove(&Flag::Enabled)),
            Command::SetSoftMode(_) => drop(self.flags.remove(&Flag::Soft)),
            Command::SetAgc(_) => drop(self.flags.remove(&Flag::Agc)),
            Command::SetGraphQuantum(_) => drop(self.flags.remove(&Flag::GraphQuantum)),
            Command::SetMonitor(_) => drop(self.flags.remove(&Flag::Monitor)),
            Command::SetTakeDefault(_) => drop(self.flags.remove(&Flag::TakeDefault)),
            Command::SetAutostart(_) => drop(self.flags.remove(&Flag::Autostart)),
            Command::SetNotify(_) => drop(self.flags.remove(&Flag::Notify)),
            Command::SetParam(key, _) => {
                if let Some(i) = PARAMS.iter().position(|s| s.key == *key) {
                    self.params.remove(&i);
                }
            }
            Command::SetSource(_) => self.source = None,
            Command::StartService => self.starting = false,
            Command::ResetParams
            | Command::ResetAll
            | Command::DismissNotices
            | Command::ClearFailure => {}
        }
        self.show_toast(refusal.message, None);
        self.render();
    }

    /// Forget optimistic values the daemon has confirmed, or that waited too long.
    fn settle_pending(&mut self) {
        let now = Instant::now();
        let snapshot = match &self.link {
            Link::Present(s) => Some(s.as_ref().clone()),
            _ => None,
        };
        let autostart_absent = match &self.link {
            Link::Absent(a) => a.autostart,
            _ => None,
        };
        self.flags.retain(|flag, p| {
            let confirmed = match (&snapshot, flag) {
                (Some(s), f) => f.of(s) == p.value,
                (None, Flag::Autostart) => autostart_absent == Some(p.value),
                (None, _) => true, // nothing to be optimistic about without a daemon
            };
            !confirmed && now < p.until
        });
        self.params.retain(|i, p| {
            let confirmed = snapshot
                .as_ref()
                .map(|s| {
                    own_param(s, *i).is_some_and(|v| (v - p.value).abs() < PARAMS[*i].step / 2.0)
                })
                .unwrap_or(true);
            !confirmed && now < p.until
        });
        if let Some(p) = &self.source {
            let confirmed = snapshot
                .as_ref()
                .map(|s| s.source == p.value)
                .unwrap_or(true);
            if confirmed || now >= p.until {
                self.source = None;
            }
        }
    }

    // ------------------------------------------------------------- actions

    fn flag(&self, flag: Flag) -> bool {
        if let Some(p) = self.flags.get(&flag) {
            return p.value;
        }
        match &self.link {
            Link::Present(s) => flag.of(s),
            Link::Absent(a) if flag == Flag::Autostart => a.autostart.unwrap_or(false),
            _ => false,
        }
    }

    fn set_flag(&mut self, flag: Flag, on: bool) {
        if self.snapshot().is_none() && flag != Flag::Autostart {
            return;
        }
        self.flags.insert(flag, Pending::new(on));
        self.send(flag.command(on));
        self.render();
    }

    fn toggle(&mut self) {
        if self.snapshot().is_none() {
            return;
        }
        // Send the explicit target rather than `Toggle`, so a second press
        // before the first is confirmed does what the screen shows.
        let target = !self.flag(Flag::Enabled);
        self.set_flag(Flag::Enabled, target);
    }

    fn set_autostart(&mut self, on: bool) {
        let was = self.flag(Flag::Autostart);
        self.set_flag(Flag::Autostart, on);
        if was && !on {
            self.show_toast(
                "DeepFilter Mic will no longer start at login.".into(),
                Some(("Undo", ToastAction::Autostart(true))),
            );
        }
    }

    fn start_service(&mut self) {
        self.starting = true;
        self.send(Command::StartService);
        self.render();
    }

    fn set_param(&mut self, index: usize, value: f64) {
        let Some(spec) = PARAMS.get(index) else {
            return;
        };
        if self.snapshot().is_none() {
            return;
        }
        let value = snap(value, spec);
        if self.param_shown(index) == Some(value) {
            return;
        }
        self.params.insert(index, Pending::new(value));
        self.send(Command::SetParam(spec.key, value));
        self.render();
    }

    fn reset_params(&mut self) {
        let Some(s) = self.snapshot() else { return };
        let before = s.params.clone();
        let soft = s.soft_mode;
        self.params.clear();
        for (i, spec) in PARAMS.iter().enumerate() {
            self.params.insert(i, Pending::new(spec.default));
        }
        self.flags.insert(Flag::Soft, Pending::new(false));
        self.send(Command::ResetParams);
        self.show_toast(
            "Fine-tuning is back to the recommended values.".into(),
            Some((
                "Undo",
                ToastAction::RestoreParams {
                    params: before,
                    soft,
                },
            )),
        );
        self.render();
    }

    fn reset_all(&mut self) {
        if self.snapshot().is_none() {
            return;
        }
        self.flags.retain(|f, _| *f == Flag::Autostart);
        self.params.clear();
        self.source = None;
        self.send(Command::ResetAll);
        self.show_toast("All settings are back to their defaults.".into(), None);
        self.render();
    }

    fn set_source(&mut self, name: String) {
        if self.snapshot().is_none() || name == proto::NODE_NAME {
            return;
        }
        self.source = Some(Pending::new(name.clone()));
        self.send(Command::SetSource(name));
        self.render();
    }

    fn reset_clip(&mut self, output: bool) {
        if output {
            self.output.reset_clip();
        } else {
            self.input.reset_clip();
        }
    }

    // ----------------------------------------------------------- comparison

    /// Record raw and filtered audio so they can be compared after the fact.
    /// Start a take, or end the running one.
    ///
    /// A take stops by itself after a few seconds, but the button stays useful
    /// while it runs: stopping early keeps everything captured so far.
    fn toggle_recording(&mut self) {
        if self.recorder.is_some() {
            self.stop_recording();
            return;
        }
        if self.playing.is_some() {
            return;
        }
        let Some(s) = self.snapshot() else {
            return;
        };
        if !s.enabled {
            self.show_toast(
                "Turn the filter on first, so there is something to compare against.".into(),
                None,
            );
            return;
        }
        let target = s.source.clone();
        match record::Recorder::start(&target, &record::take_dir()) {
            Ok(recorder) => self.recorder = Some(recorder),
            Err(error) => self.show_toast(error, None),
        }
        self.render();
    }

    /// End the take and keep what was captured.
    fn stop_recording(&mut self) {
        if let Some(recorder) = self.recorder.take() {
            self.take_dir = Some(recorder.stop());
        }
        self.render();
    }

    /// Play one of the recorded tracks, or stop the one playing.
    ///
    /// Playback is a separate player process rather than an output stream of
    /// our own: PipeWire already ships one, and a second stream inside the GUI
    /// would need its own device selection, volume and lifetime rules for no
    /// gain. The child is kept so the window can show what is playing and offer
    /// to stop it.
    fn play_take(&mut self, which: &str) {
        // Pressing the track that is playing stops it; pressing the other one
        // switches to it rather than doing nothing.
        match self.playing.as_ref() {
            Some((playing, _)) if playing == which => {
                self.stop_playback();
                return;
            }
            Some(_) => self.stop_playback(),
            None => {}
        }
        if self.recorder.is_some() {
            return;
        }
        let Some(dir) = self.take_dir.clone() else {
            return;
        };
        let file = match which {
            "raw" => "raw.wav",
            "filtered" => "filtered.wav",
            _ => return,
        };
        let path = dir.join(file);
        match std::process::Command::new("pw-play").arg(&path).spawn() {
            Ok(child) => self.playing = Some((which.to_string(), child)),
            Err(error) => self.show_toast(format!("Could not play the recording: {error}"), None),
        }
        self.render();
    }

    fn stop_playback(&mut self) {
        if let Some((_, mut child)) = self.playing.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.render();
    }

    /// Retire a finished player, and a take whose capture stopped on its own.
    fn poll_recorder(&mut self) {
        // A child exits by itself only when its node went away (the microphone
        // was unplugged, or the daemon stopped exporting), so the take cannot
        // continue: keep what was recorded rather than recording nothing.
        let dead = match self.recorder.as_mut() {
            Some(recorder) => !recorder.alive(),
            None => return,
        };
        if dead {
            self.stop_recording();
        }
    }

    fn poll_playback(&mut self) {
        let finished = match self.playing.as_mut() {
            Some((_, child)) => !matches!(child.try_wait(), Ok(None)),
            None => return,
        };
        if finished {
            self.playing = None;
            self.render();
        }
    }

    // --------------------------------------------------------------- toast

    fn show_toast(&mut self, text: String, action: Option<(&str, ToastAction)>) {
        let Some(window) = self.window.upgrade() else {
            return;
        };
        let ui = window.global::<Ui>();
        ui.set_toast(text.into());
        ui.set_toast_action(
            action
                .as_ref()
                .map(|(label, _)| *label)
                .unwrap_or("")
                .into(),
        );
        let time = if action.is_some() {
            TOAST_TIME_WITH_ACTION
        } else {
            TOAST_TIME
        };
        self.toast_action = action.map(|(_, a)| a);
        self.toast_timer.start(TimerMode::SingleShot, time, || {
            with_view(|v| v.hide_toast())
        });
    }

    fn hide_toast(&mut self) {
        self.toast_timer.stop();
        self.toast_action = None;
        if let Some(window) = self.window.upgrade() {
            let ui = window.global::<Ui>();
            ui.set_toast(SharedString::new());
            ui.set_toast_action(SharedString::new());
        }
    }

    fn toast_activate(&mut self) {
        let action = self.toast_action.take();
        self.hide_toast();
        match action {
            Some(ToastAction::RestoreParams { params, soft }) => {
                for (i, spec) in PARAMS.iter().enumerate() {
                    if let Some(&v) = params.get(spec.key) {
                        if (v - spec.default).abs() >= spec.step / 2.0 {
                            self.params.insert(i, Pending::new(v));
                            self.send(Command::SetParam(spec.key, v));
                        }
                    }
                }
                if soft {
                    self.set_flag(Flag::Soft, true);
                }
                self.render();
            }
            Some(ToastAction::Autostart(on)) => self.set_flag(Flag::Autostart, on),
            None => {}
        }
    }

    // ------------------------------------------------------------- meters

    fn tick_meters(&mut self, now: Instant, dt: f32, levels: Option<proto::Levels>) {
        let Some(window) = self.window.upgrade() else {
            return;
        };
        let snapshot = self.snapshot();
        let live = snapshot.is_some();

        let (in_status, in_reading) = match (live, levels) {
            (false, _) | (true, None) => (MeterStatus::Unavailable, Reading::Nothing),
            (true, Some(l)) if !l.input_active => (MeterStatus::NoSignal, Reading::Nothing),
            (true, Some(l)) if l.input_peak <= 0.0 => (MeterStatus::Silent, Reading::Silence),
            (true, Some(l)) => (
                MeterStatus::Live,
                Reading::Signal {
                    peak: l.input_peak,
                    rms: l.input_rms,
                    clip: l.input_clip,
                },
            ),
        };
        // The bar is only meaningful while the daemon is filtering *and*
        // something is pulling the node. Anything else means the samples in the
        // ring are stale or irrelevant, and showing them would claim audio is
        // being sent when it is not.
        let sending = snapshot.is_some_and(|s| s.listening && s.state == State::Filtering.as_str());
        let (out_status, out_reading) = match (live, levels) {
            (false, _) | (true, None) => (MeterStatus::Unavailable, Reading::Nothing),
            (true, Some(l)) if !sending || !l.output_active => {
                (MeterStatus::Unused, Reading::Nothing)
            }
            (true, Some(l)) if l.output_peak <= 0.0 => (MeterStatus::Silent, Reading::Silence),
            (true, Some(l)) => (
                MeterStatus::Live,
                Reading::Signal {
                    peak: l.output_peak,
                    rms: l.output_rms,
                    clip: l.output_clip,
                },
            ),
        };
        self.input.step(now, dt, in_reading);
        self.output.step(now, dt, out_reading);

        let view = |b: &Ballistics, status, reading| MeterView {
            status,
            rms_db: b.rms_db,
            peak_db: b.peak_db,
            hold_db: b.hold_db,
            readout: b.readout(reading).into(),
            clip: b.clipped(now),
        };
        let input = view(&self.input, in_status, in_reading);
        let output = view(&self.output, out_status, out_reading);
        let gain = levels.map(|l| l.agc_gain_db).unwrap_or(0.0);
        let agc_text: SharedString = format!("{} dB", signed(gain, 1)).into();
        let agc_active = gain.abs() >= 0.5;
        let voice = levels.is_some_and(|l| l.voice);

        let next = (input, output, agc_text, agc_active, voice);
        if self.last_meters.as_ref() == Some(&next) {
            return;
        }
        let ui = window.global::<Ui>();
        ui.set_input(next.0.clone());
        ui.set_output(next.1.clone());
        ui.set_agc_readout(next.2.clone());
        ui.set_agc_active(next.3);
        ui.set_voice(next.4);
        self.last_meters = Some(next);
    }

    // ------------------------------------------------------------- render

    fn param_shown(&self, index: usize) -> Option<f64> {
        if let Some(p) = self.params.get(&index) {
            return Some(p.value);
        }
        let s = self.snapshot()?;
        let own = own_param(s, index)?;
        // Soft mode overrides some values; show what is in effect.
        if self.flag(Flag::Soft) {
            if let Some((_, v)) = SOFT_PRESET.iter().find(|(p, _)| *p == PARAMS[index].param) {
                return Some(*v);
            }
        }
        Some(own)
    }

    fn render(&self) {
        let Some(window) = self.window.upgrade() else {
            return;
        };
        let ui = window.global::<Ui>();

        ui.set_starting(self.starting);

        match &self.link {
            Link::Present(s) => self.render_present(&ui, s),
            other => self.render_absent(&ui, other),
        }
    }

    fn render_absent(&self, ui: &Ui<'_>, link: &Link) {
        ui.set_live(false);
        ui.set_switch_on(false);
        ui.set_switch_pending(false);
        ui.set_devices(ModelRc::default());
        ui.set_source_missing(false);
        ui.set_monitor(false);
        ui.set_notices_count(0);
        ui.set_input_note("".into());
        ui.set_output_note("".into());
        ui.set_latency_total("—".into());
        ui.set_latency_parts(ModelRc::default());
        for setter in [
            Ui::set_compute_device,
            Ui::set_inference_text,
            Ui::set_overruns_text,
            Ui::set_underruns_text,
            Ui::set_inferences_text,
        ] {
            setter(ui, "—".into());
        }
        ui.set_default_detail("Unavailable while DeepFilter Mic is not running.".into());
        ui.set_values(ModelRc::new(VecModel::from(
            PARAMS.iter().map(|s| s.default as f32).collect::<Vec<_>>(),
        )));
        ui.set_locked(ModelRc::new(VecModel::from(vec![false; PARAMS.len()])));
        ui.set_params_at_default(true);

        let tile = match link {
            Link::Connecting => Tile::plain(
                Mood::Busy,
                "Connecting…",
                "Looking for the DeepFilter Mic service.".into(),
            ),
            Link::NoBus(e) => Tile::plain(
                Mood::Failed,
                "No session bus",
                format!("{e}. The window cannot reach DeepFilter Mic."),
            ),
            Link::Absent(a) => absent_tile(a, self.starting),
            Link::Present(_) => unreachable!("handled by render_present"),
        };
        ui.set_mood(tile.mood);
        ui.set_headline(tile.headline.into());
        ui.set_explain(tile.explain.into());
        ui.set_no_service_hint(tile.hint.into());
        ui.set_can_start(tile.can_start);
        ui.set_start_label(tile.start_label.into());

        let autostart = match link {
            Link::Absent(a) => a.autostart,
            _ => None,
        };
        ui.set_autostart_known(autostart.is_some());
        ui.set_autostart(
            self.flags
                .get(&Flag::Autostart)
                .map(|p| p.value)
                .or(autostart)
                .unwrap_or(false),
        );
        ui.set_take_default(false);
        ui.set_is_default(false);
        ui.set_agc(false);
        ui.set_soft(false);
        ui.set_notify(false);

        // While the daemon is down a failure is told on the tile itself.
        ui.set_failure_title("".into());
        ui.set_failure_body("".into());
    }

    fn render_present(&self, ui: &Ui<'_>, s: &Snapshot) {
        ui.set_live(true);
        ui.set_can_start(false);
        ui.set_no_service_hint("".into());

        let enabled = self.flag(Flag::Enabled);
        let state = State::parse(&s.state);
        // Predict the state an optimistic toggle leads to; the daemon corrects it.
        let shown_state = match (self.flags.get(&Flag::Enabled), state) {
            (Some(p), Some(State::NoDevice)) if p.value => State::NoDevice,
            (Some(p), _) if !p.value => State::Off,
            (Some(_), _) if s.listening => State::Filtering,
            (Some(_), _) => State::Idle,
            (None, Some(st)) => st,
            (None, None) => State::Starting,
        };
        let source_name = self
            .source
            .as_ref()
            .map(|p| p.value.clone())
            .unwrap_or_else(|| s.source.clone());
        let device = s.devices.iter().find(|d| d.name == source_name);
        let source_label = device
            .map(|d| d.description.clone())
            .unwrap_or_else(|| non_empty(&s.source_description, &s.source));
        let source_following = s.source_following;

        let (mood, headline, explain) = match shown_state {
            State::Filtering => (
                Mood::Filtering,
                "Filtering",
                if s.monitor {
                    "Noise suppression is on, and you are listening to it.".to_string()
                } else {
                    "Noise suppression is on and an app is listening.".to_string()
                },
            ),
            State::Idle => (
                Mood::Ready,
                "Ready",
                "On, but nothing is recording from DeepFilter Mic yet, so the NPU is resting."
                    .to_string(),
            ),
            State::Off => (
                Mood::Off,
                "Off",
                "Not filtering. Apps reading DeepFilter Mic get your raw microphone.".to_string(),
            ),
            State::NoDevice => (
                Mood::Warning,
                "Microphone missing",
                format!(
                    "“{source_label}” is disconnected. DeepFilter Mic is silent until it returns."
                ),
            ),
            State::Starting => (
                Mood::Busy,
                "Starting…",
                "Preparing the models on the NPU.".to_string(),
            ),
        };
        ui.set_mood(mood);
        ui.set_headline(headline.into());
        ui.set_explain(explain.into());
        ui.set_switch_on(enabled);
        ui.set_switch_pending(self.flags.contains_key(&Flag::Enabled));

        // Facts on the tile.
        let (kind, icon, _) =
            device
                .map(describe_device)
                .unwrap_or(("Microphone", IconKind::Mic, None));
        ui.set_device_chip(if s.source_available || device.is_some() {
            format!("{source_label} · {kind}").into()
        } else {
            format!("{source_label} · not connected").into()
        });
        ui.set_device_icon(icon);
        ui.set_latency_chip(if s.latency_ms >= 0.0 {
            format!("{:.0} ms measured", s.latency_ms).into()
        } else {
            "Latency: no listener".into()
        });
        let drops = s.overruns + s.underruns;
        ui.set_drops_chip(match drops {
            0 => "No dropouts".into(),
            1 => "1 dropout".into(),
            n => format!("{n} dropouts").into(),
        });
        ui.set_drops_bad(drops > 0);
        let strained = s.inference_ms > NPU_STRAIN_MS;
        ui.set_npu_chip(if s.inference_ms > 0.0 {
            format!("{} {:.1} ms", s.compute_device, s.inference_ms).into()
        } else {
            format!("{} idle", s.compute_device).into()
        });
        ui.set_npu_bad(strained);

        // Meter notes.
        ui.set_input_note(
            if !s.source_available {
                "The selected microphone is not connected."
            } else {
                ""
            }
            .into(),
        );
        ui.set_output_note(
            if shown_state == State::Off {
                "The filter is off, so this is your raw microphone."
            } else if !s.listening {
                "No app is recording from DeepFilter Mic. Turn on “Hear myself” to test it."
            } else {
                ""
            }
            .into(),
        );

        // Devices: physical first, then other software's sources; never our own.
        let mut devices: Vec<&Device> = s
            .devices
            .iter()
            .filter(|d| d.name != proto::NODE_NAME)
            .collect();
        devices.sort_by(|a, b| {
            a.virtual_node.cmp(&b.virtual_node).then_with(|| {
                a.description
                    .to_lowercase()
                    .cmp(&b.description.to_lowercase())
            })
        });
        // Following the system default is a choice like any other device, and
        // it leads the list because it is the one that keeps working when you
        // plug something else in.
        let mut rows: Vec<DeviceRow> = vec![DeviceRow {
            name: proto::SOURCE_DEFAULT.into(),
            description: "System default".into(),
            kind: if source_following && !source_label.is_empty() {
                source_label.as_str().into()
            } else {
                "whatever the system picks".into()
            },
            icon: IconKind::Auto,
            caution: String::new().into(),
            selected: source_following,
            system_default: false,
            is_virtual: false,
        }];
        rows.extend(devices.iter().map(|d| {
            let (kind, icon, caution) = describe_device(d);
            DeviceRow {
                name: d.name.as_str().into(),
                description: d.description.as_str().into(),
                kind: kind.into(),
                icon,
                caution: caution.unwrap_or("").into(),
                selected: !source_following && d.name == source_name,
                system_default: d.is_system_default,
                is_virtual: d.virtual_node,
            }
        }));
        ui.set_devices(ModelRc::new(VecModel::from(rows)));
        ui.set_source_name(source_name.as_str().into());
        ui.set_source_description(source_label.as_str().into());
        ui.set_source_missing(!s.source_available && self.source.is_none());

        // Comparison.
        ui.set_recording(self.recorder.is_some());
        ui.set_has_recording(self.take_dir.is_some());
        let playing = self.playing.as_ref().map(|(which, _)| which.clone());
        ui.set_playing(
            match playing.as_deref() {
                Some("raw") => "raw",
                Some(_) => "filtered",
                None => "",
            }
            .into(),
        );
        ui.set_playing_label(
            match playing.as_deref() {
                Some("raw") => "Playing the raw microphone",
                Some(_) => "Playing what apps hear",
                None => "",
            }
            .into(),
        );

        // Sound.
        ui.set_soft(self.flag(Flag::Soft));
        ui.set_agc(self.flag(Flag::Agc));
        ui.set_graph_quantum(self.flag(Flag::GraphQuantum));
        ui.set_monitor(self.flag(Flag::Monitor));
        ui.set_monitor_target(s.monitor_target.description.as_str().into());
        ui.set_monitor_headphones(s.monitor_target.kind == "headphones");
        ui.set_monitor_speakers(s.monitor_target.kind == "speakers");

        // Tuning.
        let soft = self.flag(Flag::Soft);
        let values: Vec<f32> = (0..PARAMS.len())
            .map(|i| self.param_shown(i).unwrap_or(PARAMS[i].default) as f32)
            .collect();
        let locked: Vec<bool> = PARAMS
            .iter()
            .map(|spec| soft && SOFT_PRESET.iter().any(|(p, _)| *p == spec.param))
            .collect();
        let at_default = !soft
            && PARAMS.iter().enumerate().all(|(i, spec)| {
                let own = self
                    .params
                    .get(&i)
                    .map(|p| p.value)
                    .or_else(|| own_param(s, i));
                own.map_or(true, |v| (v - spec.default).abs() < spec.step / 2.0)
            });
        ui.set_values(ModelRc::new(VecModel::from(values)));
        ui.set_locked(ModelRc::new(VecModel::from(locked)));
        ui.set_params_at_default(at_default);

        // Diagnostics.
        ui.set_latency_total(if s.latency_ms >= 0.0 {
            format!("{:.0} ms, measured", s.latency_ms).into()
        } else {
            "not measurable right now".into()
        });
        let parts: Vec<LatencyPart> = LATENCY_PARTS
            .iter()
            .filter_map(|(key, label, explain)| {
                s.latency_parts.get(*key).map(|ms| LatencyPart {
                    label: (*label).into(),
                    value: format!("{ms:.1} ms").into(),
                    explain: (*explain).into(),
                })
            })
            .collect();
        ui.set_latency_parts(ModelRc::new(VecModel::from(parts)));
        ui.set_compute_device(format!("{} (Intel AI Boost)", s.compute_device).into());
        ui.set_inference_text(if s.inference_ms > 0.0 {
            format!(
                "{:.2} ms{}",
                s.inference_ms,
                if strained {
                    " — close to the limit"
                } else {
                    ""
                }
            )
            .into()
        } else {
            "idle".into()
        });
        ui.set_overruns_text(count_text(s.overruns, "frame", "frames").into());
        ui.set_underruns_text(count_text(s.underruns, "gap", "gaps").into());
        ui.set_inferences_text(group_digits(s.inferences).into());

        // System.
        let take = self.flag(Flag::TakeDefault);
        ui.set_take_default(take);
        ui.set_is_default(s.is_default);
        ui.set_default_detail(default_detail(take, s.is_default, shown_state).into());
        ui.set_autostart_known(true);
        ui.set_autostart(self.flag(Flag::Autostart));
        ui.set_notify(self.flag(Flag::Notify));

        // Messages.
        ui.set_notices_count(s.notices.len() as i32);
        ui.set_notices_text(
            if s.notices.len() == 1 {
                s.notices[0].clone()
            } else {
                s.notices
                    .iter()
                    .map(|n| format!("• {n}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            .into(),
        );
        let (title, body) = failure_banner(&s.last_failure);
        ui.set_failure_title(title.into());
        ui.set_failure_body(body.into());
    }
}

// ------------------------------------------------------------------ wording

const LATENCY_PARTS: [(&str, &str, &str); 4] = [
    (
        "capture",
        "Capture",
        "PipeWire buffering between the microphone and DeepFilter Mic.",
    ),
    (
        "buffer",
        "Queue",
        "Audio waiting its turn for the NPU. It has to cover one inference pass, which is why it is not zero.",
    ),
    (
        "model",
        "Filter look-ahead",
        "Built into DeepFilterNet; the same on any hardware.",
    ),
    (
        "output",
        "Delivery",
        "PipeWire buffering between DeepFilter Mic and the app.",
    ),
];

/// What the tile says while the daemon is not on the bus.
struct Tile {
    mood: Mood,
    headline: &'static str,
    explain: String,
    hint: String,
    can_start: bool,
    start_label: &'static str,
}

impl Tile {
    fn plain(mood: Mood, headline: &'static str, explain: String) -> Self {
        Self {
            mood,
            headline,
            explain,
            hint: String::new(),
            can_start: false,
            start_label: "",
        }
    }
}

fn absent_tile(a: &Absent, starting: bool) -> Tile {
    if starting || a.unit_active {
        return Tile::plain(
            Mood::Busy,
            "Starting…",
            "Preparing the models on the NPU. This takes a few seconds.".into(),
        );
    }
    let hint = if a.unit_installed {
        String::new()
    } else {
        "The dfmic user service is not installed. Enable services.dfmic in your NixOS configuration, or run “dfmic run” in a terminal.".into()
    };
    match a.failure.as_ref().filter(|f| !f.message.is_empty()) {
        Some(f) => {
            let when = ago(f.when);
            let explain = if f.phase == "startup" {
                format!(
                    "It could not start {when}: {}. Apps use your real microphone, unfiltered.",
                    trim_period(&f.message)
                )
            } else {
                format!(
                    "It stopped {when}: {}. Your real microphone is the default again.",
                    trim_period(&f.message)
                )
            };
            Tile {
                mood: Mood::Failed,
                headline: "Filter failed",
                explain,
                hint,
                can_start: a.unit_installed,
                start_label: "Try again",
            }
        }
        None => Tile {
            mood: Mood::Absent,
            headline: "Not running",
            explain: "Apps use your microphone directly, without noise suppression.".into(),
            hint,
            can_start: a.unit_installed,
            start_label: "Start DeepFilter Mic",
        },
    }
}

fn failure_banner(f: &proto::Failure) -> (String, String) {
    if f.message.is_empty() {
        return (String::new(), String::new());
    }
    let when = ago(f.when);
    if f.phase == "startup" {
        (
            format!("The last start failed {when}"),
            format!(
                "{} It is running now; dismiss this once you have read it.",
                sentence(&f.message)
            ),
        )
    } else {
        (
            format!("The filter stopped {when}"),
            format!(
                "{} Your real microphone was restored as the default and the service restarted.",
                sentence(&f.message)
            ),
        )
    }
}

fn default_detail(take: bool, is_default: bool, state: State) -> String {
    match (take, is_default) {
        (true, true) => {
            "In effect: apps that use the default microphone get DeepFilter Mic.".into()
        }
        (true, false) => match state {
            State::Off => "Paused while the filter is off; apps use your real microphone.".into(),
            State::NoDevice => "Paused while the microphone is missing.".into(),
            _ => "Not in effect yet: another source is the default right now.".into(),
        },
        (false, true) => {
            "Off here, but DeepFilter Mic is still the default (it was chosen elsewhere).".into()
        }
        (false, false) => "Off: choose DeepFilter Mic in each app that should use it.".into(),
    }
}

/// Type label, icon and a caution for a device, from what PipeWire says about it.
fn describe_device(d: &Device) -> (&'static str, IconKind, Option<&'static str>) {
    let form = d.form_factor.as_str();
    if d.virtual_node {
        return ("Virtual", IconKind::Virtual, None);
    }
    if d.bus == "bluetooth" {
        return (
            "Bluetooth headset",
            IconKind::Bluetooth,
            Some("Using a Bluetooth microphone switches the headset to call-quality audio."),
        );
    }
    match (form, d.bus.as_str()) {
        ("webcam", _) => ("Webcam", IconKind::Webcam, None),
        ("headset" | "handset" | "headphone", "usb") => ("USB headset", IconKind::Headset, None),
        ("headset" | "handset" | "headphone", _) => ("Headset", IconKind::Headset, None),
        (_, "usb") => ("USB", IconKind::Usb, None),
        ("internal", _) | (_, "pci") => ("Built-in", IconKind::Laptop, None),
        _ => ("Microphone", IconKind::Mic, None),
    }
}

fn soft_summary() -> String {
    SOFT_PRESET
        .iter()
        .map(|(param, value)| match param {
            Param::AttenLim => format!("the noise reduction limit to {value:.0} dB"),
            Param::PostFilter if *value == 0.0 => "extra cleanup off".to_string(),
            Param::PostFilter => format!("extra cleanup to {value:.3}"),
            other => format!("{} to {value}", other.spec().key),
        })
        .collect::<Vec<_>>()
        .join(" and ")
}

fn own_param(s: &Snapshot, index: usize) -> Option<f64> {
    s.params.get(PARAMS.get(index)?.key).copied()
}

fn snap(value: f64, spec: &proto::ParamSpec) -> f64 {
    let steps = ((value - spec.min) / spec.step).round();
    let v = (spec.min + steps * spec.step).clamp(spec.min, spec.max);
    // Keep decimal steps exact (0.001 * 20 is not 0.02 in binary).
    let decimals = (-spec.step.log10()).ceil().max(0.0) as i32;
    let scale = 10f64.powi(decimals);
    (v * scale).round() / scale
}

fn non_empty(a: &str, b: &str) -> String {
    if a.is_empty() {
        b.to_string()
    } else {
        a.to_string()
    }
}

fn signed(value: f64, decimals: usize) -> String {
    if value.abs() < 0.05 {
        format!("{:.*}", decimals, 0.0)
    } else if value > 0.0 {
        format!("+{:.*}", decimals, value)
    } else {
        meters::minus(value, decimals)
    }
}

fn count_text(n: u64, one: &str, many: &str) -> String {
    match n {
        0 => "none".into(),
        1 => format!("1 {one}"),
        n => format!("{} {many}", group_digits(n)),
    }
}

fn group_digits(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push('\u{2009}'); // thin space
        }
        out.push(c);
    }
    out
}

fn trim_period(s: &str) -> &str {
    s.trim().trim_end_matches('.')
}

fn sentence(s: &str) -> String {
    let s = s.trim();
    let mut out: String = s
        .chars()
        .take(1)
        .flat_map(char::to_uppercase)
        .chain(s.chars().skip(1))
        .collect();
    if !out.ends_with(['.', '!', '?']) {
        out.push('.');
    }
    out
}

/// "2 minutes ago", "at 14:03", "on 3 Nov at 14:03".
fn ago(unix: i64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(unix);
    let secs = (now - unix).max(0);
    match secs {
        0..=59 => "just now".into(),
        60..=119 => "a minute ago".into(),
        120..=3599 => format!("{} minutes ago", secs / 60),
        3600..=7199 => "an hour ago".into(),
        7200..=86399 => format!("{} hours ago", secs / 3600),
        86400..=172799 => "yesterday".into(),
        _ => format!("{} days ago", secs / 86400),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapping_keeps_decimal_steps_exact() {
        let post = Param::PostFilter.spec();
        assert_eq!(snap(0.0204, post), 0.02);
        assert_eq!(snap(1.0, post), 0.05);
        let atten = Param::AttenLim.spec();
        assert_eq!(snap(12.4, atten), 12.0);
        assert_eq!(snap(-3.0, atten), 0.0);
    }

    #[test]
    fn digits_are_grouped() {
        assert_eq!(group_digits(999), "999");
        assert_eq!(group_digits(1234567), "1\u{2009}234\u{2009}567");
    }
}
