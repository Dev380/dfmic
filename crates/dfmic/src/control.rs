//! Rust side of the D-Bus control interface (see `c/dbus_ctl.c`).
//!
//! This is what the GUI and `dfmic status`/`dfmic toggle` talk to. Properties are
//! read through `org.freedesktop.DBus.Properties`, and `Changed` is emitted when
//! the daemon's state moves.

use std::ffi::{c_char, c_int, c_void, CString};

pub struct Control {
    raw: *mut DfmicDb,
}

enum DfmicDb {}

/// Commands the daemon accepts over the bus.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Toggle,
    Enable,
    Disable,
    Shutdown,
    /// Live parameter change: `atten-lim` or `post-filter`.
    SetParam(String, f64),
}

pub type CommandSink = std::sync::mpsc::Sender<Command>;

#[link(name = "dfmic_pw")]
extern "C" {
    fn dfmic_dbus_start(
        device: *const c_char,
        source: *const c_char,
        on_method: extern "C" fn(*const c_char, *const c_char, f64, *mut c_void),
        user: *mut c_void,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut DfmicDb;
    fn dfmic_dbus_update(
        ctl: *mut DfmicDb,
        enabled: c_int,
        level: f64,
        out_level: f64,
        latency_ms: f64,
        inferences: u32,
        dropped: u32,
        error: *const c_char,
    );
    fn dfmic_dbus_emit_changed(ctl: *mut DfmicDb);
    fn dfmic_dbus_set_params(ctl: *mut DfmicDb, atten_lim: f64, post_filter: f64);
    fn dfmic_dbus_set_free_hook(ctl: *mut DfmicDb, free_user: extern "C" fn(*mut c_void));
    fn dfmic_dbus_stop(ctl: *mut DfmicDb);
}

extern "C" fn on_method(method: *const c_char, arg: *const c_char, value: f64, user: *mut c_void) {
    let sink = unsafe { &*(user as *const CommandSink) };
    if method.is_null() {
        return;
    }
    let name = unsafe { std::ffi::CStr::from_ptr(method) }.to_string_lossy().into_owned();
    let arg = if arg.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(arg) }.to_string_lossy().into_owned()
    };
    let command = match name.as_str() {
        "Toggle" => Command::Toggle,
        "Enable" => Command::Enable,
        "Disable" => Command::Disable,
        "SetParam" => Command::SetParam(arg, value),
        // SetSource needs a stream restart; the daemon logs and ignores it.
        _ => return,
    };
    let _ = sink.send(command);
}

/// Frees the boxed `CommandSink` when the interface shuts down.
extern "C" fn drop_sink(user: *mut c_void) {
    unsafe { drop(Box::from_raw(user as *mut CommandSink)) };
}

/// Live values pushed into the D-Bus object.
#[derive(Default, Clone)]
pub struct Status {
    pub enabled: bool,
    pub atten_lim: f64,
    pub post_filter: f64,
    pub level: f64,
    pub out_level: f64,
    pub latency_ms: f64,
    pub inferences: u32,
    pub dropped: u32,
    pub error: String,
}

impl Control {
    pub fn start(device: &str, source: &str, sink: CommandSink) -> Result<Self, String> {
        let device = CString::new(device).map_err(|e| e.to_string())?;
        let source = CString::new(source).map_err(|e| e.to_string())?;
        // The sink must outlive the object: it is stored by the C side as user data.
        let boxed = Box::into_raw(Box::new(sink));
        let mut err = [0i8; 256];
        let raw = unsafe {
            dfmic_dbus_start(
                device.as_ptr(),
                source.as_ptr(),
                on_method,
                boxed as *mut c_void,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if raw.is_null() {
            unsafe { drop(Box::from_raw(boxed)) };
            let message = unsafe { std::ffi::CStr::from_ptr(err.as_ptr() as *const c_char) }
                .to_string_lossy()
                .into_owned();
            return Err(message);
        }
        unsafe { dfmic_dbus_set_free_hook(raw, drop_sink) };
        Ok(Self { raw })
    }

    pub fn update(&self, status: &Status) {
        let error = CString::new(status.error.as_str()).unwrap_or_default();
        unsafe {
            dfmic_dbus_update(
                self.raw,
                status.enabled as c_int,
                status.level,
                status.out_level,
                status.latency_ms,
                status.inferences,
                status.dropped,
                error.as_ptr(),
            )
        };
    }

    /// Publish the tunables the daemon is actually running with.
    pub fn set_params(&self, atten_lim: f64, post_filter: f64) {
        unsafe { dfmic_dbus_set_params(self.raw, atten_lim, post_filter) };
    }

    pub fn notify_changed(&self) {
        unsafe { dfmic_dbus_emit_changed(self.raw) };
    }
}

impl Drop for Control {
    fn drop(&mut self) {
        unsafe { dfmic_dbus_stop(self.raw) };
    }
}
