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
| mean per 10 ms frame | **1.13 ms** |
| p99 / max | 1.72 ms / 1.82 ms |
| NPU duty cycle | **11%** |
| realtime factor | 8.8× |
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
