//! Recording the two taps, from the window's side.
//!
//! The window records like any other application: it captures the exported
//! `dfmic-source` node for the filtered track and the physical microphone node
//! for the raw one. The daemon does not know a take is happening, which is why
//! there is nothing to add to its control interface — it just sees two more
//! clients.
//!
//! Capture is delegated to `pw-record`, one child per track. Doing it in-process
//! would mean a second PipeWire event loop inside the UI process, with its own
//! thread, buffer lifetime and error handling, to produce two WAV files: the
//! standard tool already does exactly that and is what a user would reach for by
//! hand. The children are stopped with SIGINT, which is how `pw-record` is meant
//! to be interrupted and what makes it finalise the file header.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// Where takes are kept.
///
/// One stable directory rather than one per take: the point is to listen to the
/// last thing you recorded, and a record button overwriting it is the expected
/// behaviour.
pub fn take_dir() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from(".local/share"))
        .join("dfmic/compare")
}

/// A take in progress: one child per track, both writing into `dir`.
pub struct Recorder {
    raw: Child,
    filtered: Child,
    dir: PathBuf,
}

impl Recorder {
    /// Start capturing `raw_target` and the exported node into `dir`.
    ///
    /// `raw_target` is the physical microphone, so the raw track is what the
    /// microphone actually heard; the filtered track is `dfmic-source`, which is
    /// what applications receive.
    pub fn start(raw_target: &str, dir: &Path) -> Result<Self, String> {
        if raw_target.is_empty() {
            return Err("No microphone is selected, so there is nothing to record.".into());
        }
        std::fs::create_dir_all(dir)
            .map_err(|error| format!("could not create {}: {error}", dir.display()))?;
        let raw = spawn(raw_target, &dir.join("raw.wav"))?;
        // If the second child fails the first would be left recording, so it is
        // stopped before the error is reported.
        let filtered = match spawn(crate::proto::NODE_NAME, &dir.join("filtered.wav")) {
            Ok(child) => child,
            Err(error) => {
                let mut raw = raw;
                stop_child(&mut raw);
                return Err(error);
            }
        };
        Ok(Self {
            raw,
            filtered,
            dir: dir.to_path_buf(),
        })
    }

    /// Whether both children are still running.
    ///
    /// `pw-record` exits on its own only when its node disappears (the
    /// microphone was unplugged, or the daemon stopped exporting), so a dead
    /// child means the take cannot continue.
    pub fn alive(&mut self) -> bool {
        let raw = matches!(self.raw.try_wait(), Ok(None));
        let filtered = matches!(self.filtered.try_wait(), Ok(None));
        raw && filtered
    }

    /// Stop both children and let them finalise their files.
    pub fn stop(mut self) -> PathBuf {
        stop_child(&mut self.raw);
        stop_child(&mut self.filtered);
        self.dir.clone()
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        // Whatever happens to the window, no `pw-record` is left behind.
        stop_child(&mut self.raw);
        stop_child(&mut self.filtered);
    }
}

fn spawn(target: &str, path: &Path) -> Result<Child, String> {
    Command::new("pw-record")
        .arg("--target")
        .arg(target)
        .arg("--rate")
        .arg("48000")
        .arg("--channels")
        .arg("1")
        .arg("--format")
        .arg("f32")
        .arg(path)
        // A failing child reports through its exit status; its output is of no
        // use to the window and would end up mixed into the terminal.
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("could not run pw-record: {error}"))
}

/// Ask a child to finish, then make sure it is gone.
fn stop_child(child: &mut Child) {
    if matches!(child.try_wait(), Ok(None)) {
        // SIGINT is what `pw-record` expects: it stops the stream and writes the
        // WAV header with the final length.
        unsafe { libc::kill(child.id() as i32, libc::SIGINT) };
        // Half a second is generous for closing a file; anything slower is not
        // worth blocking the UI thread over.
        for _ in 0..50 {
            match child.try_wait() {
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(10)),
                _ => return,
            }
        }
        let _ = child.kill();
    }
    let _ = child.wait();
}
