# dfmic: DeepFilterNet low-latency noise suppression on Intel NPUs

Note: The following is an AI vibeslopped README that I don't really have time to fix right now sorry :( but it is working!

---


DeepFilterNet3_**ll** noise suppression as a virtual microphone, running on the
**Intel NPU** (Intel AI Boost) rather than the CPU.

The converted DeepFilterNet3_ll graphs run on the Intel NPU in a streaming Rust
pipeline. The `dfmic run` daemon exports `DeepFilter Mic` as a PipeWire
`Audio/Source`; the CLI and other clients control it over the session D-Bus.

## Why this is not just "run the ONNX on the NPU"

The upstream `DeepFilterNet3_ll_onnx.tar.gz` graphs are *stateless*: the GRU
recurrence runs along the sequence axis, and libDF relies on tract's `pulse`
transform to hold state between calls. OpenVINO has no equivalent, and the
graphs are stateful in **two** ways:

1. **GRUs** — 6 of them (1 encoder, 2 ERB decoder, 3 deep-filter decoder), each
   a single 512-unit layer.
2. **Causal convolutions along time** — every conv is causal, so a frame's output
   depends on its predecessors. Measured receptive fields: 6 frames for the
   encoder (its `e3` head only converges at a 6-frame window), 3 for each
   decoder.

`tools/dfnpy/convert.py` makes both explicit:

- each GRU is replaced with a single primitive cell (`MatMul`/`Add`/`Slice`/
  `Sigmoid`/`Tanh`) whose hidden state is a graph input and output. The GRU
  advances **exactly one frame per inference** — feeding it a padded window
  would advance it over frames that do not exist yet, which desynchronises it
  (a GRU with non-zero biases does not sit still on zero input);
- each GRU's *output history* is threaded as state too, so the causal
  convolutions downstream still see a full window of context;
- the stream axis is pinned to a static length, which the NPU requires.

The runtime therefore carries two kinds of state per graph: a window of past
input frames (for the convs) and a per-frame history of each GRU's outputs.

The two numbers in this document that are easy to confuse: **0.96–1.13 ms** is
what the **Python** runtime drives through the converted graphs (measured
offline, not on the daemon's path), while `dfmic doctor` reports the **Rust**
pipeline end to end and reads 2.2–2.6 ms per frame on this machine (see
[Status](#status-stage-3-complete-gui)).

Measured cost of that choice, driving the real graphs on the NPU, 48 kHz mono,
480-sample hop, through the Python runtime:

| metric | value |
|---|---|
| mean per 10 ms frame | **0.96 – 1.13 ms** |
| p99 / max | 1.30 – 1.72 ms |
| NPU duty cycle | **10–11%** |
| realtime factor | 8.8 – 10.4× |
| host CPU (Python runtime) | 0.38 ms/frame |

For reference, the same weights on the CPU via tract (nixpkgs `deepfilternet`
LADSPA plugin) cost 0.25 CPU-seconds per audio-second, i.e. 4× realtime at 25%
of a core. The NPU is not faster — it is *freeing*, which is the point.

## Correctness gate

`tools/dfnpy/verify_convert.py` runs the original stateless ONNX graphs over a
whole sequence (OpenVINO CPU, f32) and compares them against the converted IR
driven frame by frame. Both sides of that comparison are **CPU**: the reference
is fp32 CPU, so the candidate has to be the same kind of device and the same
precision, and the check has to run on machines with no NPU. The flake runs it
with `--device CPU` for exactly that reason.

Current result (default seed, 60 frames, `--settle 10`, as measured on
2026-11-26):

```
emb    SDR 48.98 dB      c0    SDR 76.34 dB
lsnr   SDR 37.76 dB      e0    SDR 71.74 dB
e3     SDR 66.79 dB      m     SDR 44.98 dB
coefs  SDR 40.14 dB
```

The bound is per output, not global: the convolution heads (`e0`, `e3`, `c0`)
never consume a GRU output and are deterministic to within a dB across seeds, so
their floor is 60 dB; the GRU-carried heads (`emb`, `lsnr`, `m`, `coefs`) amplify
float reordering chaotically on random-noise input (22–60 dB across 60 seeds) and
only get a 25 dB "did not collapse" floor.

The IR is built with `compress_to_fp16`, so even the convolution heads are not
bit-exact against the fp32 reference: `e0` reads ~71.7 dB, not `inf dB`. Build
with `--fp32` and those three read `inf dB`, which confirms the difference is the
built precision and not the graph rewrite. The GRU path's residual is chaotic
amplification of float reordering on the random noise used as test input; real
speech does not do that, which is what the next gate (`--settle`, real audio vs
upstream `deep-filter`) is for.

This check is wired up as `nix build .#checks.x86_64-linux.parity`, and
`nix flake check` builds it along with the package and its tests.

## Usage

```sh
nix develop            # python + onnx + openvino + rust + ffmpeg + ladspa-sdk
nix build .#models     # converted IR in ./result (manifest.json + config.ini)

# convert by hand
python tools/dfnpy/convert.py \
  --model-tar DeepFilterNet3_ll_onnx.tar.gz --out /tmp/ir

# prove the conversion is faithful (CPU: the reference is fp32 CPU)
python tools/dfnpy/verify_convert.py \
  --model-tar DeepFilterNet3_ll_onnx.tar.gz --ir /tmp/ir \
  --device CPU --frames 60 --settle 10

# check the GRU decomposition on its own
python tools/dfnpy/check_gru.py --model-tar DeepFilterNet3_ll_onnx.tar.gz
```

`manifest.json` in the IR directory describes, per graph, the static input
shapes, the window length and each state tensor (name, shape, output index).

## Design decisions worth remembering

- **NPU only, no silent fallback.** If the NPU is missing or a graph will not
  compile, the daemon will fail loudly and leave the real microphone as the
  default. No CPU inference path exists.
- **10 ms blocks.** The `_ll` model's own latency is ~10 ms (`hop_size` 480,
  `df_lookahead` 0); the windowed inference keeps that. On the Rust pipeline the
  NPU duty is the 2.2–3.0 ms `doctor` figure against a 10 ms hop, i.e. 22–30%
  (see [Status](#status-stage-3-complete-gui)); the 11% in the Python timing
  table above is that runtime, not the daemon.
- **The system audio graph is only touched on request.** A small PipeWire
  quantum makes every application's graph shorter, so it is a user setting
  (`graph-quantum`, on by default) that is requested *only while filtering* and
  handed back the moment the filter stops. Status reports measured capture,
  buffering, model and output latency separately, and sums them, instead of
  treating the 10 ms model delay as the whole cost.
- **It costs a few percent of one core, and that is real work.** The numbers in
  this file predate the current pipeline and could not be reproduced for this
  revision, so treat them as an indication rather than a measurement: 8% of a
  core for the process while filtering, 6% of that in the NPU worker, 0% idle or
  passing audio through. What can be reproduced is the offline cost, which uses
  the same DSP: `dfmic enhance` on 60 s of audio ran 11.4-11.8 s, i.e. **5.2-5.3x
  realtime** on 2026-11-26. A `perf` profile of the worker shows it is not
  OpenVINO overhead but the DSP itself: rustfft's butterflies (the STFT and
  ISTFT), the deep-filter synthesis, AGC, and the `memmove`s that marshal tensors
  in and out of OpenVINO. The release build matters: a debug build is several
  times more expensive, which is what `perf` will show you if you profile
  `cargo run` without `--release`.
- **The queue depth adapts, the way libDF's does.** libDF keeps a `proc_delay`
  that grows a frame when the worker falls behind and shrinks when it has
  headroom, floored at `frame_size * (1 + min_buffer)`. Ours does the same, with
  one difference forced by the architecture: libDF's audio callback blocks until
  a block is ready, so a failed shrink merely stalls it, whereas a PipeWire
  callback must return immediately and a failed shrink shows up as a real
  dropout. The wait between shrink attempts therefore doubles on every failure
  (5 s up to 2 min), so the depth converges instead of oscillating. The floor is
  two hops rather than libDF's one, because the callback cannot block: measured
  at one hop the queue dropped thousands of buffers, at one hop plus a quantum
  hundreds, at two hops none.
- **The worker is woken by a futex, not a poll.** It used to sleep 1 ms when
  the input ring held less than a hop, which cost a full 10 ms of latency and
  caused hundreds of dropped buffers. Busy-waiting fixed the latency but burned
  4x the CPU. The capture callback now bumps an atomic and calls
  `FUTEX_WAKE_PRIVATE` — one syscall, no lock, safe from the realtime thread —
  so the worker wakes within microseconds of a hop completing at essentially no
  CPU cost. Measured: 41 ms and 828 dropouts polling, 31 ms and 4 dropouts with
  the futex.
- **The audio bridge is Rust.** The virtual microphone uses the official
  `pipewire-rs` bindings, so the realtime process callbacks are ordinary Rust
  rather than careful C. `pw_bridge.c` is gone; the only hand-written FFI left
  is OpenVINO's.
- **The NPU runs on its own thread, and that is load-bearing.** Processing
  inline in PipeWire's realtime callback would be lower latency, but the worker
  has a long tail: over 4000 hops of real audio the p99.9 was 8.3 ms and the
  worst single hop 50 ms. Inline, a 50 ms stall breaks the graph and glitches
  every client. Decoupling through the rings turns that into a dropped buffer at
  worst. (The same measurement with `CAP_SYS_NICE` gives 6.8 ms and 20 ms, which
  is why `services.dfmic.realtime` exists — it buys margin, not latency.)
- **Where the ~40 ms goes.** As measured on the development machine on
  2026-11-26 with a consumer attached, `dfmic status` reports the parts
  separately and sums them: ~6 ms of capture-stream buffering, 20 ms of queue,
  10 ms of model look-ahead and ~5 ms of output-stream buffering. The parts are
  measured from `pw_stream_get_time_n` and the ring lengths, not estimated. Two
  of them follow from code, so they are not machine-specific: the look-ahead is
  `HOP_SIZE / RATE` = 10 ms (`MODEL_DELAY_MS`), and the queue floor is
  `MIN_HOPS = 2` hops = 20 ms. The stream-buffering figures move with the
  quantum and the device. Inference is deliberately *not* added: it is absorbed
  by the queue that plays while the NPU works, so counting it would count the
  same milliseconds twice. `doctor` reports the per-frame cost with the SNR
  gates opened, because the DSP skips the ERB gains and the deep filter both on
  clean input and on plain noise, which used to make its figure about three
  times too optimistic.
- **Stateful artifacts are built, never generated at runtime.** The Nix
  derivation produces the IR; nothing at runtime touches ONNX. The daemon does
  **not** cache a compiled NPU blob today: `NpuRuntime::load` takes a
  `cache_dir`, but every caller passes `None`, so each start pays the NPU
  compile (0.1–0.2 s here, reported by `doctor`). The plumbing for
  `CACHE_DIR` + `CACHE_MODE=OPTIMIZE_SPEED` is in `dfn-npu` and unused.
- **Vendored DSP.** Stage 2 vendors `DFState` from `libDF` (MIT/Apache-2.0,
  credited) rather than forking the whole training repo.

## The control panel

`dfmic gui` opens a Slint window that is a remote control and a meter for the
daemon; the audio engine stays in the service. Everything it shows comes from
`org.dfmic.Control` on the session bus, and every control is a method call, so
the same interface drives keybinds:

```
dfmic toggle                  # flips the filter, prints on/off
gdbus call --session --dest org.dfmic.Control --object-path /org/dfmic/Control \
  --method org.dfmic.Control.SetParam 'atten-lim' 60.0
```

The window shows the state in one glance (Filtering/Off), a real dBFS meter for
the microphone and for what apps receive, the device in use, latency and drop
counters, and an Advanced section with the two knobs that are worth touching —
each labelled with what it does. Expert parameters apply live: the daemon reads
them from atomics at the top of every block, so nothing on the audio path takes
a lock.

Meters are measured, not decorative: the daemon tracks the peak of the samples
it captures and of the samples it hands to the source node, and the GUI maps
those to a −60..0 dBFS scale.

The window only repaints when a value actually changed, so a panel with no audio
flowing costs almost nothing: measured with meters live and a consumer attached
(2026-11-26, release build, femtovg on the GPU), the window process used
0.9-1.1 s of CPU per 10 s, i.e. **9-11% of one core** at a steady 26-27 fps, or
roughly 3.5-4 ms of CPU per frame. Hidden or idle it drops to ~1 fps and ~1% of a
core. (A debug build is several times worse, which is worth knowing before
blaming the UI.)

What each frame costs is up to the renderer. Slint's default is OpenGL
(femtovg), which repaints the entire window every frame *regardless of how much
changed* - the cost is in scene traversal, glyph runs and path tessellation, not
in the rasterised area. That is why hiding part of the window does not make
frames cheaper under femtovg, and why Slint's software renderer, which does track
dirty regions, is the cheaper choice when OpenGL itself would be software. The
same 8 s measurement with `SLINT_BACKEND` forced: software 1.6 s of CPU, femtovg
1.2 s, on this machine's GPU. So the window asks for the software renderer when
OpenGL would be software (`LIBGL_ALWAYS_SOFTWARE`, a `MESA_LOADER_DRIVER_OVERRIDE`
naming llvmpipe, or no `/dev/dri` render node at all) and uses the GPU otherwise.
`SLINT_BACKEND` overrides the choice:

```
SLINT_BACKEND=winit-software dfmic gui     # never touch OpenGL
SLINT_BACKEND=winit-femtovg  dfmic gui     # always use the GPU
```

The software renderer cannot transform items, so the UI draws its disclosure
arrows as two separate glyphs rather than rotating one, and it needs Slint's
`software-renderer-path` feature (enabled here) to draw the icon set at all.

### Following the system default

The microphone setting has a `default` value that means "whichever device the
system default points at", so a hot-plugged headset or a desktop that moves the
default is picked up without touching dfmic:

```
dfmic source default      # follow the system default from now on
dfmic source              # shows the device it currently resolves to
```

While the takeover is active dfmic *is* the default source, so following it
literally would record our own output; the daemon resolves to the device it
displaced instead, and steps aside as soon as something else claims the default.

### Comparing raw against filtered

Testing by talking and listening live is misleading: what you hear through the
self-monitor arrives a few milliseconds after you speak, which makes every
setting sound wrong. The window's **Compare** card records both taps at once so
they can be played back afterwards, raw against filtered, from the same take.

It is a convenience for trying settings out, so it lives in the window only:
press **Record**, speak, then **Play raw** and **Play filtered** to hear the
difference. A take stops after six seconds, or when you press **Stop**. While it
runs, the *Sent to apps* meter shows what is being captured, so the picture and
the recording agree.

Files live in `~/.local/share/dfmic/compare/` as `raw.wav` and `filtered.wav`. A
take makes the daemon filter on its own, so nothing has to be recording from the
virtual microphone for it to work.

## Not done yet

- **Sample-exact parity** with upstream (see above; needs per-layer conv state).

## Roadmap

1. **Model + parity (done, except real-audio gate)** — remaining: a numpy port
   of libDF's `DFState` (STFT/ERB/normalisation/deep-filter/synthesis) so the
   harness can run real speech against upstream `deep-filter`. The offline CLI
   already exists: `dfmic enhance in.wav out.wav` (positional output, no `-o`).
2. **Engine + PipeWire** — Rust, vendored DSP, OpenVINO behind a runtime trait,
   exported `Audio/Source` node, systemd user service, default-source takeover
   with restore, loud failure handling.
3. **GUI** — Slint + D-Bus, full DeepFilterNet knob set with defaults marked,
   "keep music/room" soft mode, self-monitor, AGC (VAD-gated), latency readout.

## Licence

Code here is MIT/Apache-2.0. DeepFilterNet and its model weights are by Hendrik
Schröter et al., MIT/Apache-2.0 — see `NOTICE`.

## Status: stage 3 complete (GUI)

The daemon, the PipeWire node and the systemd integration work. Verified on this
machine (Core Ultra 9 285H, Intel AI Boost NPU, PipeWire 1.6.11 + WirePlumber) on
2026-11-26:

```
$ dfmic doctor
openvino devices: ["CPU", "GPU", "NPU"]
models: ~/.local/share/dfmic/models/DeepFilterNet3_ll-0.5.6
compiled on NPU in 0.11 s
steady state: 2.96 ms per 10 ms frame (3.4x realtime, 29.6% duty)
```

That figure is the whole Rust pipeline (DSP + all three graphs on the NPU) per
10 ms hop, and it is noisy: five 500-frame runs on the same machine gave
2.2–2.6 ms, and `DFMIC_DOCTOR_PACED=1` (pacer on, like the daemon) gave 5.1 ms
because the sleeps and the NPU contend. Quote it as "a few ms", not as a
headline. With 10 ms of hop that is 3–5× realtime, so 20–35% NPU duty.

The daemon prints very little: it reports a model load or a missing model, a
failed `pw-metadata` restore and a `CAP_SYS_NICE` note (if realtime priority was
not granted), and otherwise works through the bus. Operators watch it with
`dfmic status`, `journalctl --user -u dfmic` and the window, not with its stdout.
The startup path worth knowing is the unpleasant one: after an unclean stop it
reads `$XDG_STATE_HOME/dfmic/default-source`, restores the default source *before*
creating any stream, and reports `could not restore PipeWire default source ...`
if it cannot — which needs `pw-metadata` on `PATH` (see [NixOS](#nixos)).

What the daemon does:

- captures the physical microphone on a PipeWire input stream pinned to an
  explicit device (never "follow the default", see below),
- exports an `Audio/Source` node called `DeepFilter Mic`,
- runs the NPU on a worker thread between two lock-free rings, so the realtime
  callbacks never block,
- takes over the default source while running and restores it on exit, including
  after `SIGKILL` (the previous default is written to
  `$XDG_STATE_HOME/dfmic/default-source` and restored on the next start),
- writes no status file: `dfmic status` reads `org.dfmic.Control` over the
  session bus, and works only while the daemon owns the name,
- and **idles the NPU when nothing is recording** from the node: on a laptop that
  is the difference between filtering into the void and actually saving power.

### Two bugs worth remembering

Both were found by testing against the real system rather than in isolation:

1. **Self-capture.** The capture stream first followed the *default* source.
   Taking over the default then made it capture our own output: silence, zero
   inferences, no error. The capture target is now resolved *before* the takeover
   and pinned with `node.dont-reconnect`.
2. **Missing autoconnect.** Without `PW_STREAM_FLAG_AUTOCONNECT` WirePlumber
   never linked the capture stream to the microphone, so the node sat in
   `suspended` and produced silence.

## Honest note on parity with upstream

The DSP is exact: replaying libDF's own gains/coefficients through `dfn-dsp`
(`dfmic replay-dsp`) reproduces upstream's audio at **51 dB SDR**, and the
converted graphs match the upstream ONNX graphs to ~1e-3 per element when given
the same recurrent state (the `max|diff|` column of `verify_convert.py`; the
GRU-carried heads are 1e-2 to 1, because random noise makes the recurrence
chaotic).

End-to-end, however, our output is *not* sample-identical to upstream
`deep-filter`. The reason is the recurrent state: upstream (tract's `pulse`)
keeps per-layer convolution state, while we run a sliding window and can only
keep the GRU state. At the start of a stream the two therefore diverge, and
DFN3's encoder GRU has long enough memory that the difference never fully decays.

Measured consequence: on speech-like audio with a known clean reference, upstream
and dfmic behave the same (output rms 0.0139 vs 0.0142, same suppression), but the
waveforms differ. That comparison was made through `enhance`, on a fixed input,
before the AGC existed; the daemon's AGC adds a second, deliberate difference.
Making it sample-exact would mean exposing per-layer conv state in the IR, which
is a larger change than it is worth for a live microphone.

## Usage

```sh
# check the NPU and the models
dfmic doctor

# clean a file offline
# --agc runs the automatic gain exactly as the daemon does, so the gain path can
# be measured on a fixed recording; --agc-trace prints one line per hop
dfmic enhance in.wav out.wav [--atten-lim 100] [--post-filter 0.02] [--agc [--agc-trace]]

# the virtual microphone (becomes the default source, restores on exit)
dfmic run [--source <node name>] [--default-source never]

# full status; --json includes every setting, meter and diagnostic
dfmic status [--json]

# state and settings
dfmic toggle
dfmic on | off
dfmic soft on|off
dfmic agc on|off
dfmic monitor on|off
dfmic default on|off
dfmic autostart on|off
dfmic notify on|off

# enumerate or select capture nodes without removing DeepFilter Mic
dfmic source list
dfmic source alsa_input....source

# all seven libDF tuning values; out-of-range input is an error
dfmic set atten-lim 60
dfmic set post-filter 0.01
dfmic set min-buffer 3
dfmic set mask-reduce mean

dfmic reset              # current device's parameters, soft mode off
dfmic reset --all        # global settings and every device, not autostart
dfmic dismiss-notices
dfmic clear-failure
dfmic service start      # uses systemctl --user, no daemon required

# one-paragraph usage summary
dfmic help
```

### NixOS

```nix
{
  inputs.dfmic.url = "path:~/Code/dfmic"; # or git+...
  # ...
  imports = [inputs.dfmic.nixosModules.default];
  services.dfmic = {
    enable = true;
    # source = "alsa_input...HiFi__Mic1__source"; # pin the microphone
  };
}
```

The NixOS module installs the package with `environment.systemPackages` and its
systemd *user* unit with `systemd.packages`. It does not force the service to
start for every user. Choose autostart per login user:

```sh
systemctl --user enable dfmic.service
systemctl --user start dfmic.service
# Or, equivalently, with the daemon running:
dfmic autostart on
```

The module overrides the unit's `PATH` (both its drop-in and the one the package
ships) to include what the daemon actually launches: `pw-metadata`
(`pkgs.pipewire`), `curl`, `sha256sum` (`pkgs.coreutils`), `tar` + `gzip`
(`pkgs.gnutar`, `pkgs.gzip`) and `systemctl` (`pkgs.systemd`). NixOS user units
get only its minimal default `PATH` otherwise, which holds none of these —
without `pw-metadata` the daemon cannot restore the default source after an
unclean stop and refuses to start, and `Restart=on-failure` then loops. `tar -xzf`
shells out to `gzip`, so it is listed explicitly rather than assumed.

`services.dfmic.source` is a node **name**, and names can contain spaces. The
module passes the whole argv through `utils.escapeSystemdExecArgs`, so
`source = "Built-in Audio Analog Stereo"` arrives as one argument instead of a
target plus four stray parameters.

Set `services.dfmic.realtime = false` if you do not want the `CAP_SYS_NICE`
wrapper; everything else is unaffected.

The unit restarts on failure with a fixed `RestartSec=5`, throttled to five starts
per minute: without a start limit, a start-up that always fails would repeat
every five seconds forever. `dfmic service start`, `Autostart` and `SetAutostart`
operate without depending on daemon-owned state. The module is not a
home-manager module and does not use `home.packages`.

### Realtime priority while developing

The NPU worker asks for `CAP_SYS_NICE` and prints a note when it cannot have it;
without it the worker's worst case grows from ~20 ms to ~50 ms. `sudo` is the
wrong tool for this, because the daemon has to stay inside your session (it
speaks to your session bus and your PipeWire), and both `sudo -E` and a file
capability break the GUI: glibc enters secure-execution mode and discards
`LD_LIBRARY_PATH`, which is the only way Slint's `dlopen`ed libwayland and
libEGL resolve from the dev shell. Use the wrapper, which grants the capability
with ambient capabilities *after* exec, so nothing is discarded:

```sh
tools/dfmic-rt run --source alsa_input...HiFi__Mic1__source
tools/dfmic-rt gui
```

The installed package does not need this: the NixOS module grants the same
capability through a `security.wrappers` entry.

### Daemon and D-Bus

`org.dfmic.Control` owns `/org/dfmic/Control`, interface `org.dfmic.Control1`.
The Rust contract in `crates/dfmic/src/proto.rs` defines every method and
property. `PropertiesChanged` is emitted for setting changes and diagnostics
are coalesced to at most 2 Hz. `Levels()` reads a timestamped 50 ms history, so
multiple clients can poll it without stealing each other's peaks.

* **Off** keeps `DeepFilter Mic` present and passes raw audio unchanged; it
  releases the default-source takeover. `NoDevice` outputs silence and resumes
  only when the selected node returns, never a different fallback node.
* Devices and default-source facts come from PipeWire registry/default metadata.
  Source changes reconnect only the capture stream. The previous real default
  is restored on graceful exit and from an atomic backup on the next start
  after an unclean exit.
* AGC levels the signal the suppressor **produced**, not the microphone signal,
  to -18 dBFS RMS: it cuts in 0.2 s and boosts over 1 s, bounded to +18 dB into
  a -1 dBFS limiter, and additionally capped so the amplified room never exceeds
  -30 dBFS however the gate behaves. Its gate is a level test against a room
  floor tracked as the minimum of the smoothed input over 10 s, plus 12 dB, with
  the model's own zero-mask threshold as a veto — the local SNR alone is not a
  voice detector (measured over a quiet room: median -9 dB, above zero in a
  third of hops). The gain holds across the gaps inside a sentence and releases
  to unity at 4 dB/s after them, so a quiet room comes out quieter than the
  microphone instead of 12 dB above it. `dfmic enhance --agc` runs the same path
  on a file. Self-monitor
  follows the default sink and also counts as a processing consumer. It is off
  by default; a saved monitor request starts only for a headphone target.
* The window stops working when there is nothing to look at. Hidden or minimised,
  the meter tick drops to 4 Hz and the microphone capture stops; scrolled so the
  Levels card is out of the viewport, the meters stop updating and the daemon is
  no longer polled for levels at all (`Levels()` only feeds the output meter,
  which means nothing unless something is being sent). Measured 111 → 16 ms/s.
  Occlusion is left to Slint's renderer, which skips the draw. Its `pw-record`
  children carry `PR_SET_PDEATHSIG`, so a killed window cannot leave an orphan
  holding the microphone.
* The input meter works without the daemon: with no device list to consult, the
  window captures the **system default source** (no `--target`), so the mic bar
  still moves when the service is down.
* The daemon meters nothing on its input: the **window** captures the physical
  microphone itself for the input meter, so an idle daemon pauses its capture
  stream entirely (measured: 0 ms/s of CPU and **0 wakeups/s** when off). The
  event loop does not poll at all: every asynchronous source of change rings a
  wake channel — D-Bus commands, PipeWire's registry and metadata callbacks, the
  worker, a finished download and the signal handler — so it sleeps until there
  is something to do, with a 30 s timeout purely as a safety net. The one
  exception is a missing model, which is polled at 2 Hz because a file appearing
  on disk cannot announce itself.
* The daemon **always starts with the filter off**, and does not take the system
  default until it is turned on. Whether it is on is runtime state, not a
  setting: nothing is written to `settings.toml` for it, so a saved `enabled`
  from an older version is ignored and a login can never grab the microphone
  unattended.
* `source = "default"` in the settings file means follow the system default. The
  daemon re-resolves it on every registry change and retargets only the capture
  stream, so switching microphones costs no restart and no re-negotiation of the
  exported node.
* Where the model comes from depends on how dfmic was installed. The **Nix
  package** builds the IR as its own derivation (`nix/models.nix`), links it at
  `$out/share/dfmic/models` and points the unit's `DFMIC_MODELS` there — so it
  *is* in the closure, and an installed daemon never downloads anything (the
  target is read-only, so `can_download` is false and the window offers no
  button). A **from-source build** has neither, so it searches
  `$XDG_DATA_HOME/dfmic/models/<version>/` and can fetch the `.tar.gz` release
  asset **on request**, never behind your back (SHA-256 checked before anything
  is unpacked). Missing model is a state (`no-model`), not a failure: the daemon
  runs, owns the bus, lists devices and passes audio through untouched. The
  **daemon** is the only thing that downloads — the window asks it over
  `DownloadModel()`, and `dfmic models download` uses the same call when a daemon
  is running (and downloads directly when one is not). `ModelPresent`,
  `ModelDownloading` and `ModelDownloadPossible` report it, so a client can offer
  the button only when it could work. `--models DIR` and `DFMIC_MODELS` win over
  both, which is how the dev shell supplies it.
* Comparison takes are written to `$XDG_DATA_HOME/dfmic/compare/` as
  `raw.wav` and `filtered.wav`. `StartRecording(seconds)` returns at once,
  `StopRecording()` ends a take early keeping what it has, and `Recording`
  reports progress. The take counts as a consumer, so the NPU runs for it even
  when nothing is recording from the node, and the output meter is fed from the
  worker for the same reason.
* Global settings live in `$XDG_CONFIG_HOME/dfmic/settings.toml`. Params and
  soft mode are keyed by PipeWire node name. Invalid saved values are corrected
  and reported in `Notices`; writes are atomic. Soft mode preserves the user's
  underlying values and rejects edits to overridden keys.
* Fatal failures are saved in `$XDG_STATE_HOME/dfmic/last-failure`, notified
  from the daemon and exposed after a restart. `dfmic status` also prints this
  record when no daemon is running. Desktop notification actions are handled by
  the daemon; set `dfmic notify off` to disable notifications.
* `min-buffer` implements libDF's minimum processing-buffer shortening rule.
  `mask-reduce` is libDF's cross-channel max/mean mask reduction and therefore
  has no audible effect in this mono engine. Defaults preserve the old DSP path.

## Development

```sh
nix develop                # exports DFMIC_MODELS, RUSTFLAGS and LD_LIBRARY_PATH
cargo build                # daemon and CLI, no GUI feature required
cargo test                 # DSP, AGC, replay and GUI-logic tests (17 today)
cargo clippy -p dfmic --no-deps -- -D warnings
nix build .#dfmic          # the daemon; its tests run in the build (doCheck)
nix build .#models         # converted NPU artifacts
nix flake check            # package + its tests, and the parity gate
```

The dev shell exports `DFMIC_MODELS` pointing at the built model derivation, so
both `cargo run` and `tools/dfmic-rt` find the IR without a download. `cargo
build`/`cargo test` there still need the shell's `RUSTFLAGS` and
`LD_LIBRARY_PATH` for `pipewire-sys` and the GUI.

Diagnostics that paid for themselves:

- `dfmic dump-features file.wav [--models DIR]` — DSP features, and the encoder's
  per-frame outputs, for comparison against `libDF/examples/dump_features.rs`.
- `DFMIC_DUMP=1 dfmic enhance ...` — per-frame gains/coefficients in the same
  format as libDF's `DF_DUMP`, feedable to `dfmic replay-dsp`.
- `dfmic enhance --agc --agc-trace ...` — per-hop input level, the
  post-suppression level the gain was computed from, the gain, and whether the
  gate was open.
- `DFMIC_DEBUG=1 dfmic run` — PipeWire stream state transitions.
