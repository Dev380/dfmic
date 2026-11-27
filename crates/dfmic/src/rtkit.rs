//! Realtime scheduling, granted by RealtimeKit.
//!
//! The NPU worker's worst case is what decides whether the output ring starves:
//! measured over 4000 hops of real audio, realtime scheduling moved p99.9 from
//! 8.3 ms to 6.8 ms and the worst single hop from 50 ms to 20 ms. A long stall
//! there clicks, so this buys dropout margin — it does not lower latency, since
//! the queue floor comes from bursty production rather than from jitter.
//!
//! **Why rtkit rather than `pthread_setschedparam`.** Raising a thread to
//! `SCHED_FIFO` needs a non-zero `RLIMIT_RTPRIO`, and an unprivileged
//! `systemd --user` manager cannot provide one: a user unit's `LimitRTPRIO=` is
//! accepted by systemd and then silently ignored, because the manager itself runs
//! with `RLIMIT_RTPRIO 0`. A capability wrapper would work, but it makes
//! `CAP_SYS_NICE` ambient in everything that touches the binary — a much larger
//! hammer than this needs. RealtimeKit asks the same question through polkit and
//! answers "yes" for an active local session, with no prompt.
//!
//! **What rtkit requires**, verified against `rtkit-daemon.c` and by experiment:
//!
//!   * the priority must be below rtkit's own (`our_realtime_priority`, 21 by
//!     default), so 18 leaves it able to preempt us while still sitting under
//!     PipeWire's audio threads;
//!   * `RLIMIT_RTTIME` must not exceed rtkit's `RTTimeUSecMax`, or the request is
//!     refused outright. Zero is accepted.
//!
//! **What the kernel requires** is the other half, and it is the one that bit:
//! `RLIMIT_RTTIME` is a budget on how long a realtime thread may run *without
//! blocking*, after which the kernel SIGKILLs it — and the signal takes the whole
//! process with it. Two things follow, and both are load-bearing:
//!
//!   * the worker is promoted only *after* the NPU graphs are compiled, because
//!     that is about half a second of uninterrupted CPU;
//!   * `libpipewire-module-rt` is refused when the context is created (see
//!     `Bridge::start`). It disables `RLIMIT_RTTIME` for the whole process while
//!     probing for realtime privileges, and on an unprivileged process the
//!     attempt to restore the default (`-1`, i.e. infinity) fails with `EPERM`,
//!     leaving 0/0. A realtime thread with a zero budget is killed on its first
//!     slice of CPU, which looked exactly like an out-of-memory kill.

/// Priority the worker asks for.
///
/// Below rtkit's own (21) so it can always preempt us, and below PipeWire's audio
/// threads so a full output ring is drained before the NPU refills it.
const PRIORITY: u32 = 18;

/// The RT-time budget in microseconds: rtkit's `RTTimeUSecMax`. Kept because
/// [`reserve_rt_time`] sets it, which is still worth doing — a plain
/// `dfmic run` outside the daemon may load other PipeWire-adjacent libraries.
///
/// Far more than one hop needs — a hop is ~10 ms and the worker blocks on a futex
/// between hops — and far less than the ~0.5 s an NPU compile takes, which is
/// exactly why promotion waits for the compile.
const RTTIME_USEC: libc::rlim_t = 200_000;

/// `SCHED_RESET_ON_FORK`, which libc does not expose as a constant.
const SCHED_RESET_ON_FORK: libc::c_int = 0x4000_0000;

/// Attempts at the handshake: slack for rtkit still starting, or a transient
/// D-Bus error. Every attempt after the first costs one `RETRY_DELAY`.
const ATTEMPTS: u32 = 3;
const RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(250);

#[zbus::proxy(
    interface = "org.freedesktop.RealtimeKit1",
    default_service = "org.freedesktop.RealtimeKit1",
    default_path = "/org/freedesktop/RealtimeKit1"
)]
trait RealtimeKit {
    /// Promote one thread of the caller's own process.
    fn make_thread_realtime(&self, thread: u64, priority: u32) -> zbus::Result<()>;
}

/// Give the process an RT-time budget before anything can take it away.
///
/// libpipewire's `module-rt` sets `RLIMIT_RTTIME` to 0/0 when it initialises,
/// and raising a hard limit that is already 0 fails with `EPERM` for an
/// unprivileged process. Calling this before `Bridge::start` is therefore the only
/// chance to set it.
///
/// It is not enough on its own — the kernel still kills a realtime thread that
/// never blocks — which is why [`promote_current_thread`] is only called after the
/// NPU graphs are compiled.
pub fn reserve_rt_time() -> Result<(), String> {
    let limit = libc::rlimit {
        rlim_cur: RTTIME_USEC,
        rlim_max: RTTIME_USEC,
    };
    // SAFETY: `setrlimit` reads the two integers in `limit` and nothing else.
    if unsafe { libc::setrlimit(libc::RLIMIT_RTTIME, &limit) } != 0 {
        return Err(format!(
            "could not set RLIMIT_RTTIME: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

/// Ask rtkit to put the calling thread on `SCHED_FIFO`.
///
/// Must be called *from* the thread that wants it: rtkit identifies the caller
/// over D-Bus and only promotes threads of the calling process. It also has to be
/// called only when that thread is about to become mostly idle — see the module
/// docs on `RLIMIT_RTTIME`.
///
/// Returns the reason it failed, for the caller to report once. Every failure mode
/// here is expected somewhere (no rtkit installed, an inactive session), so
/// nothing is fatal.
pub fn promote_current_thread() -> Result<(), String> {
    // Already realtime? A capability wrapper or an explicit `chrt` got there
    // first and there is nothing to ask for.
    if is_realtime() {
        return Ok(());
    }

    let mut last = String::new();
    for attempt in 0..ATTEMPTS {
        if attempt > 0 {
            std::thread::sleep(RETRY_DELAY);
        }
        match request() {
            Ok(()) => return Ok(()),
            Err(error) => last = error,
        }
    }
    Err(last)
}

/// One `MakeThreadRealtime` for this thread.
fn request() -> Result<(), String> {
    let connection = zbus::blocking::Connection::system()
        .map_err(|error| format!("could not reach the system bus: {error}"))?;
    let proxy = RealtimeKitProxyBlocking::new(&connection)
        .map_err(|error| format!("could not reach RealtimeKit: {error}"))?;
    // SAFETY: `gettid` takes no arguments and cannot fail.
    let tid = unsafe { libc::gettid() } as u64;
    proxy
        .make_thread_realtime(tid, PRIORITY)
        .map_err(|error| match error {
            zbus::Error::MethodError(_, Some(message), _) => message,
            other => other.to_string(),
        })?;

    // rtkit leaves the thread alone in some polkit configurations, so the answer
    // is only "granted" if the scheduler actually changed.
    if is_realtime() {
        Ok(())
    } else {
        Err("RealtimeKit accepted the request but the scheduler is unchanged".into())
    }
}

/// The scheduler policy of the calling thread, without its flags.
///
/// `sched_getscheduler` returns the policy OR'd with `SCHED_RESET_ON_FORK`
/// (0x40000000) when that flag is set, which rtkit always sets. Comparing the raw
/// value against `SCHED_FIFO` therefore fails on a thread rtkit just promoted —
/// which is exactly the bug this mask exists to avoid.
fn policy() -> libc::c_int {
    // SAFETY: `sched_getscheduler` takes a pid and only reads kernel state.
    let raw = unsafe { libc::sched_getscheduler(0) };
    raw & !SCHED_RESET_ON_FORK
}

/// Whether the calling thread is on a realtime policy.
pub fn is_realtime() -> bool {
    let policy = policy();
    policy == libc::SCHED_FIFO || policy == libc::SCHED_RR
}
