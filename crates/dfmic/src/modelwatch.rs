//! Notice a model appearing on disk.
//!
//! A model that is not installed cannot announce itself: it is either
//! downloaded by this daemon (which knows when it finishes) or dropped into
//! place by someone else — `dfmic models download` run by hand, a file copied
//! in, a mount appearing. Polling for that was the last thing keeping the daemon
//! awake, so instead this watches the directory and sleeps.
//!
//! `inotify` rather than a general-purpose watcher crate: it is in the kernel,
//! `libc` already exposes it, and the whole job is "tell me when these names
//! appear in this directory". The watch is on the *parent* directory, because the
//! model directory itself usually does not exist yet.
//!
//! The thread blocks in `poll()` with no timeout, so it costs nothing while
//! waiting. Shutdown goes through a pipe of its own: writing a byte wakes the
//! poll, which is the only way to interrupt a blocking wait without racing a
//! `close()` against it.

use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Watches a model directory and calls `on_change` when its contents change.
pub struct ModelWatch {
    stop: Arc<AtomicBool>,
    /// Read end of the shutdown pipe; the thread blocks on this and the watch.
    wake_read: i32,
    /// Write end, kept so `Drop` can poke the reader.
    wake_write: i32,
    inotify: i32,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl ModelWatch {
    /// Watch `dir`, calling `on_change` whenever a file appears in it or in its
    /// parent directory.
    ///
    /// `on_change` runs on the watcher's own thread, so it must be cheap and must
    /// not block — the daemon's wake call is exactly that.
    pub fn start(dir: &Path, on_change: impl Fn() + Send + 'static) -> Option<Self> {
        let parent = dir.parent()?.to_path_buf();
        let inotify = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
        if inotify < 0 {
            return None;
        }
        // The parent is where a download lands (it is renamed into place from a
        // sibling), and where the model directory itself appears.
        if !add_watch(inotify, &parent) {
            // Nothing to watch: the parent does not exist yet either. Not an
            // error — the caller falls back to looking periodically.
            unsafe { libc::close(inotify) };
            return None;
        }
        // If the directory is already there, watch inside it too, so files being
        // filled in are seen even if the directory was not recreated.
        let _ = add_watch(inotify, dir);

        let mut fds = [0i32; 2];
        if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } < 0 {
            unsafe { libc::close(inotify) };
            return None;
        }
        let (wake_read, wake_write) = (fds[0], fds[1]);

        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let dir = dir.to_path_buf();
        let thread = std::thread::Builder::new()
            .name("dfmic-model-watch".into())
            .spawn(move || watch_loop(inotify, wake_read, &dir, &flag, on_change))
            .ok()?;
        Some(Self {
            stop,
            wake_read,
            wake_write,
            inotify,
            thread: Some(thread),
        })
    }
}

impl Drop for ModelWatch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Poke the reader out of `poll`, then wait for it: closing the
        // descriptor underneath a thread blocked on it is a race, this is not.
        let byte = 1u8;
        unsafe { libc::write(self.wake_write, (&byte as *const u8).cast(), 1) };
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        unsafe {
            libc::close(self.inotify);
            libc::close(self.wake_read);
            libc::close(self.wake_write);
        }
    }
}

fn add_watch(fd: i32, path: &Path) -> bool {
    let Ok(c_path) = CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    // Names appearing or moving in, plus the directory itself being created or
    // deleted. `IN_ONLYDIR` keeps a file with the same name from being watched.
    let mask = libc::IN_CREATE
        | libc::IN_MOVED_TO
        | libc::IN_CLOSE_WRITE
        | libc::IN_DELETE
        | libc::IN_MOVED_FROM
        | libc::IN_ONLYDIR;
    unsafe { libc::inotify_add_watch(fd, c_path.as_ptr(), mask) >= 0 }
}

fn watch_loop(inotify: i32, wake_read: i32, dir: &Path, stop: &AtomicBool, on_change: impl Fn()) {
    let mut buffer = [0u8; 4096];
    let mut watched_inside = false;
    while !stop.load(Ordering::SeqCst) {
        // No timeout: this thread sleeps until something actually happens.
        let mut polled = [
            libc::pollfd {
                fd: inotify,
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: wake_read,
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        let ready = unsafe { libc::poll(polled.as_mut_ptr(), polled.len() as libc::nfds_t, -1) };
        if ready < 0 {
            // `EINTR` from a signal is normal; anything else would spin, so give
            // it a moment before trying again.
            if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            continue;
        }
        if polled[1].revents != 0 {
            return; // Shutdown.
        }
        if polled[0].revents == 0 {
            continue;
        }
        let read = unsafe {
            libc::read(
                inotify,
                buffer.as_mut_ptr().cast::<libc::c_void>(),
                buffer.len(),
            )
        };
        if read <= 0 {
            continue;
        }
        // The directory may have just appeared, which is when it becomes worth
        // watching inside it as well.
        if !watched_inside && dir.is_dir() {
            watched_inside = add_watch(inotify, dir);
        }
        on_change();
    }
}

/// Where a watch should be placed for a model directory: the directory itself,
/// or its nearest existing ancestor while it does not exist yet.
pub fn watch_root(dir: &Path) -> PathBuf {
    let mut probe = dir.to_path_buf();
    while !probe.exists() {
        match probe.parent() {
            Some(parent) if parent != probe => probe = parent.to_path_buf(),
            _ => return dir.to_path_buf(),
        }
    }
    probe
}
