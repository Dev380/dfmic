# dfmic

DeepFilterNet3_**ll** noise suppression as a virtual microphone, running on the
**Intel NPU** (Intel AI Boost) rather than the CPU.

Status: **stage 1 of 3** — the model conversion, the streaming runtime and the
parity gate exist and pass. The Rust engine, PipeWire node and GUI are not
written yet. See [Roadmap](#roadmap).

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

Measured cost of that choice, driving the real graphs on the NPU, 48 kHz mono,
480-sample hop:

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
driven frame by frame. Current result on the NPU:

```
emb    SDR 48.66 dB      c0    SDR 69.35 dB
e0     SDR 67.29 dB      e3    SDR 63.45 dB
m      SDR 44.97 dB      coefs SDR 39.64 dB
```

With an f32 IR the convolution heads are *bit-exact* (`inf dB`); the residual on
the GRU path is chaotic amplification of float reordering on the random noise
used as test input. Real speech does not do that, which is what the next gate
(`--settle`, real audio vs upstream `deep-filter`) is for.

This check is wired up as `nix build .#checks.x86_64-linux.parity`.

## Usage

```sh
nix develop            # python + onnx + openvino + rust + ffmpeg + ladspa-sdk
nix build .#models     # converted IR in ./result (manifest.json + config.ini)

# convert by hand
python tools/dfnpy/convert.py \
  --model-tar DeepFilterNet3_ll_onnx.tar.gz --out /tmp/ir

# prove the conversion is faithful
python tools/dfnpy/verify_convert.py \
  --model-tar DeepFilterNet3_ll_onnx.tar.gz --ir /tmp/ir \
  --device NPU --frames 60 --settle 10
```

`manifest.json` in the IR directory describes, per graph, the static input
shapes, the window length and each state tensor (name, shape, output index).

## Design decisions worth remembering

- **NPU only, no silent fallback.** If the NPU is missing or a graph will not
  compile, the daemon will fail loudly and leave the real microphone as the
  default. No CPU inference path exists.
- **10 ms blocks.** The `_ll` model's own latency is ~10 ms (`hop_size` 480,
  `df_lookahead` 0); the windowed inference keeps that, at 11% NPU duty.
- **The system audio graph is not touched.** No global quantum changes; the
  filter adds ~10 ms and the GUI reports the measured end-to-end figure.
- **Stateful artifacts are built, never generated at runtime.** The Nix
  derivation produces the IR; the daemon will load it and cache an
  `EXPORT_IMPORT`-compiled NPU blob so restarts skip the compile.
- **Vendored DSP.** Stage 2 vendors `DFState` from `libDF` (MIT/Apache-2.0,
  credited) rather than forking the whole training repo.

## Not done yet

- **GUI** (stage 3): Slint window over D-Bus, the full knob set, soft mode,
  self-monitor, AGC.
- **AGC** (automatic gain), which the model can leave quieter than the raw mic.
- **Sample-exact parity** with upstream (see above; needs per-layer conv state).

## Roadmap

1. **Model + parity (done, except real-audio gate)** — remaining: a numpy port
   of libDF's `DFState` (STFT/ERB/normalisation/deep-filter/synthesis) so the
   harness can run real speech against upstream `deep-filter`, plus the offline
   `dfmic enhance in.wav -o out.wav` CLI.
2. **Engine + PipeWire** — Rust, vendored DSP, OpenVINO behind a runtime trait,
   exported `Audio/Source` node, systemd user service, default-source takeover
   with restore, loud failure handling.
3. **GUI** — Slint + D-Bus, full DeepFilterNet knob set with defaults marked,
   "keep music/room" soft mode, self-monitor, AGC (VAD-gated), latency readout.

## Licence

Code here is MIT/Apache-2.0. DeepFilterNet and its model weights are by Hendrik
Schröter et al., MIT/Apache-2.0 — see `NOTICE`.

## Status: stage 2 complete

The daemon, the PipeWire node and the systemd integration work. Verified on this
machine (Core Ultra 9 285H, Intel AI Boost NPU, PipeWire 1.6.11 + WirePlumber):

```
dfmic doctor
  openvino devices: ["CPU", "GPU", "NPU"]
  compiled on NPU in 0.07 s
  steady state: 1.9 ms per 10 ms frame (5.2x realtime, 10% duty)

dfmic run
  dfmic: capturing from alsa_input...HiFi__Mic1__source
  dfmic: 'DeepFilter Mic' is now the default source
  ... recordings from dfmic-source: rms 0.0038, peak 0.025, 0 dropped frames
  ... on exit: restored default source to alsa_input...HiFi__Mic1__source
```

What the daemon does:

- captures the physical microphone on a PipeWire input stream pinned to an
  explicit device (never "follow the default", see below),
- exports an `Audio/Source` node called `DeepFilter Mic`,
- runs the NPU on a worker thread between two lock-free rings, so the realtime
  callbacks never block,
- takes over the default source while running and restores it on exit, including
  after `SIGKILL` (the previous default is written to
  `$XDG_RUNTIME_DIR/dfmic/previous-default` and restored on the next start),
- writes `$XDG_RUNTIME_DIR/dfmic/status` with live counters for `dfmic status`,
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
converted graphs match the upstream ONNX graphs to ~1e-3 when given the same
recurrent state.

End-to-end, however, our output is *not* sample-identical to upstream
`deep-filter`. The reason is the recurrent state: upstream (tract's `pulse`)
keeps per-layer convolution state, while we run a sliding window and can only
keep the GRU state. At the start of a stream the two therefore diverge, and
DFN3's encoder GRU has long enough memory that the difference never fully decays.

Measured consequence: on speech-like audio with a known clean reference, upstream
and dfmic behave the same (output rms 0.0139 vs 0.0142, same suppression), but the
waveforms differ. Making it sample-exact would mean exposing per-layer conv state
in the IR, which is a larger change than it is worth for a live microphone.

## Usage

```sh
# check the NPU and the models
dfmic doctor

# clean a file offline
dfmic enhance in.wav out.wav [--atten-lim 100] [--post-filter 0.02]

# the virtual microphone (becomes the default source, restores on exit)
dfmic run [--source <node name>] [--default-source never]

# what is it doing?
dfmic status
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

The module installs a systemd *user* service that restarts on failure with a
backoff; the daemon restores the previous default source before exiting, so a
crash never leaves the system without a microphone.

## Development

```sh
nix develop
cargo test                 # DSP unit tests (STFT round trip, ERB bank, ...)
nix build .#dfmic          # the daemon
nix build .#models         # converted NPU artifacts
nix flake check            # conversion parity gate
```

Diagnostics that paid for themselves:

- `dfmic dump-features file.wav [--models DIR]` — DSP features, and the encoder's
  per-frame outputs, for comparison against `libDF/examples/dump_features.rs`.
- `DFMIC_DUMP=1 dfmic enhance ...` — per-frame gains/coefficients in the same
  format as libDF's `DF_DUMP`, feedable to `dfmic replay-dsp`.
- `DFMIC_DEBUG=1 dfmic run` — PipeWire stream state transitions.
