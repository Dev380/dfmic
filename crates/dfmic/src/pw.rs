//! Rust side of the PipeWire bridge (see `c/pw_bridge.c`).

use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

pub struct Bridge {
    raw: *mut DfmicPw,
    /// Keeps the callback state alive for as long as PipeWire may call into it.
    _shared: Arc<Shared>,
}

enum DfmicPw {}

type CaptureCb = extern "C" fn(*const f32, u32, *mut c_void);
type PlaybackCb = extern "C" fn(*mut f32, u32, *mut c_void);
type StateCb = extern "C" fn(c_int, *const c_char, *mut c_void);
type SourceStartedCb = extern "C" fn(*mut c_void);
type SourceStoppedCb = extern "C" fn(*mut c_void);

#[link(name = "dfmic_pw")]
extern "C" {
    fn dfmic_pw_start(
        node_name: *const c_char,
        node_description: *const c_char,
        target_source: *const c_char,
        on_capture: CaptureCb,
        on_playback: PlaybackCb,
        on_state: StateCb,
        on_source_started: SourceStartedCb,
        on_source_stopped: SourceStoppedCb,
        user: *mut c_void,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut DfmicPw;
    fn dfmic_pw_ready(bridge: *mut DfmicPw) -> c_int;
    fn dfmic_pw_stop(bridge: *mut DfmicPw);
}

/// Everything the realtime callbacks touch. Kept lock-free on purpose: these run
/// on PipeWire's data thread and must never wait for the NPU.
pub struct Shared {
    /// Samples captured from the microphone, waiting for the worker.
    pub input: ring::Ring,
    /// Enhanced samples waiting for the source node.
    pub output: ring::Ring,
    /// Set when a stream reports an error.
    pub failed: AtomicBool,
    pub error: std::sync::Mutex<Option<String>>,
    /// Frames the microphone delivered, for status reporting.
    pub captured: AtomicU64,
    /// Frames the source node consumed.
    pub played: AtomicU64,
    /// Frames dropped because the worker could not keep up.
    pub dropped: AtomicU64,
    /// Peak absolute sample captured, in units of 1e-6, for level reporting.
    pub peak: AtomicU64,
    /// Peak absolute sample emitted to apps, in units of 1e-6.
    pub output_peak: AtomicU64,
    /// Whether anything is recording from our node.
    pub consumers: AtomicBool,
}

impl Shared {
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            input: ring::Ring::new(capacity),
            output: ring::Ring::new(capacity),
            failed: AtomicBool::new(false),
            error: std::sync::Mutex::new(None),
            captured: AtomicU64::new(0),
            played: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            output_peak: AtomicU64::new(0),
            consumers: AtomicBool::new(false),
        })
    }

    fn set_error(&self, message: String) {
        if let Ok(mut slot) = self.error.lock() {
            if slot.is_none() {
                *slot = Some(message);
            }
        }
        self.failed.store(true, Ordering::SeqCst);
    }
}

extern "C" fn on_capture(samples: *const f32, frames: u32, user: *mut c_void) {
    let shared = unsafe { &*(user as *const Shared) };
    let data = unsafe { std::slice::from_raw_parts(samples, frames as usize) };
    shared.captured.fetch_add(frames as u64, Ordering::Relaxed);
    let peak = data.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let scaled = (peak * 1e6) as u64;
    if scaled > shared.peak.load(Ordering::Relaxed) {
        shared.peak.store(scaled, Ordering::Relaxed);
    }
    if !shared.input.push(data) {
        shared.dropped.fetch_add(frames as u64, Ordering::Relaxed);
    }
}

extern "C" fn on_playback(samples: *mut f32, frames: u32, user: *mut c_void) {
    let shared = unsafe { &*(user as *const Shared) };
    let data = unsafe { std::slice::from_raw_parts_mut(samples, frames as usize) };
    let got = shared.output.pop(data);
    shared.played.fetch_add(got as u64, Ordering::Relaxed);
    let peak = data.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let scaled = (peak * 1e6) as u64;
    if scaled > shared.output_peak.load(Ordering::Relaxed) {
        shared.output_peak.store(scaled, Ordering::Relaxed);
    }
}

extern "C" fn on_source_started(user: *mut c_void) {
    let shared = unsafe { &*(user as *const Shared) };
    // Drop whatever was captured while nobody was listening.
    shared.output.clear();
    shared.consumers.store(true, Ordering::SeqCst);
}

extern "C" fn on_source_stopped(user: *mut c_void) {
    let shared = unsafe { &*(user as *const Shared) };
    shared.consumers.store(false, Ordering::SeqCst);
}

extern "C" fn on_state(error: c_int, message: *const c_char, user: *mut c_void) {
    let shared = unsafe { &*(user as *const Shared) };
    let text = if message.is_null() {
        "unknown PipeWire error".to_string()
    } else {
        unsafe { std::ffi::CStr::from_ptr(message) }.to_string_lossy().into_owned()
    };
    shared.set_error(text);
    let _ = error;
}

impl Bridge {
    pub fn start(
        node_name: &str,
        description: &str,
        target: Option<&str>,
        shared: Arc<Shared>,
    ) -> Result<Self, String> {
        let node = CString::new(node_name).map_err(|e| e.to_string())?;
        let desc = CString::new(description).map_err(|e| e.to_string())?;
        let target = match target {
            Some(t) => CString::new(t).map_err(|e| e.to_string())?,
            None => CString::new("").unwrap(),
        };
        let mut err = [0i8; 256];
        let user = Arc::as_ptr(&shared) as *mut c_void;
        let raw = unsafe {
            dfmic_pw_start(
                node.as_ptr(),
                desc.as_ptr(),
                if target.as_bytes().is_empty() {
                    std::ptr::null()
                } else {
                    target.as_ptr()
                },
                on_capture,
                on_playback,
                on_state,
                on_source_started,
                on_source_stopped,
                user,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if raw.is_null() {
            let message = unsafe { std::ffi::CStr::from_ptr(err.as_ptr() as *const c_char) }
                .to_string_lossy()
                .into_owned();
            return Err(message);
        }
        Ok(Self { raw, _shared: shared })
    }

    pub fn ready(&self) -> bool {
        unsafe { dfmic_pw_ready(self.raw) != 0 }
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        unsafe { dfmic_pw_stop(self.raw) };
    }
}

/// Single-producer single-consumer sample ring, lock-free.
pub mod ring {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct Ring {
        buffer: Box<[f32]>,
        write: AtomicUsize,
        read: AtomicUsize,
    }

    impl Ring {
        pub fn new(capacity: usize) -> Self {
            Self {
                buffer: vec![0.0; capacity].into_boxed_slice(),
                write: AtomicUsize::new(0),
                read: AtomicUsize::new(0),
            }
        }

        pub fn capacity(&self) -> usize {
            self.buffer.len()
        }

        pub fn len(&self) -> usize {
            let w = self.write.load(Ordering::Acquire);
            let r = self.read.load(Ordering::Acquire);
            w.wrapping_sub(r)
        }

        /// Push samples; returns false if there was not enough room.
        pub fn push(&self, samples: &[f32]) -> bool {
            if self.len() + samples.len() > self.capacity() {
                return false;
            }
            let mut w = self.write.load(Ordering::Relaxed);
            for &s in samples {
                let index = w % self.capacity();
                // Safety: only the producer writes below `write`.
                unsafe {
                    let ptr = self.buffer.as_ptr().add(index) as *mut f32;
                    ptr.write(s);
                }
                w = w.wrapping_add(1);
            }
            self.write.store(w, Ordering::Release);
            true
        }

        /// Discard everything buffered.
        pub fn clear(&self) {
            let w = self.write.load(Ordering::Acquire);
            self.read.store(w, Ordering::Release);
        }

        /// Pop up to `samples.len()` samples; zero-fills the remainder.
        pub fn pop(&self, samples: &mut [f32]) -> usize {
            let available = self.len().min(samples.len());
            let mut r = self.read.load(Ordering::Relaxed);
            for slot in samples.iter_mut().take(available) {
                let index = r % self.capacity();
                *slot = self.buffer[index];
                r = r.wrapping_add(1);
            }
            for slot in samples.iter_mut().skip(available) {
                *slot = 0.0;
            }
            self.read.store(r, Ordering::Release);
            available
        }
    }
}
