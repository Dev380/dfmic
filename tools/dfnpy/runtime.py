"""Streaming wrapper around the converted DeepFilterNet3_ll graphs.

Contract with the converter (see `convert.py`): each graph takes a fixed window
of `frames` frames plus every GRU's hidden state *as of the window start*, and
returns outputs for the window, the state after each frame in the window, and
the state after the last frame.

Two kinds of state exist and both are handled here:

  * **conv history** — the causal convolutions along time need their previous
    frames of input, so a window of `frames` frames is fed and only the last
    frame of output is used;
  * **GRU hidden state** — threaded per frame, so the state fed for a window is
    the state as of the frame just before the window.

The net cost is `frames`x the compute of a single-frame pass; the README records
why we accept that instead of rewriting the conv stack.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import openvino as ov

GRAPH_ORDER = ("enc", "erb_dec", "df_dec")


class Graph:
    """One converted graph: compiled model, input history, per-frame GRU states."""

    def __init__(self, core: ov.Core, ir: Path, spec: dict, device: str, precision: str):
        self.spec = spec
        self.name = spec["graph"]
        self.frames = spec["frames"]
        model = core.read_model(str(ir))
        config = {"INFERENCE_PRECISION_HINT": precision} if device == "NPU" else {}
        self.compiled = core.compile_model(model, device, config)
        self.request = self.compiled.create_infer_request()
        self.output_names = {
            self.compiled.output(i): self.compiled.output(i).get_any_name()
            for i in range(len(self.compiled.outputs))
        }
        # State outputs are addressed positionally: OpenVINO may rename outputs it
        # folds, but it preserves their order.
        self.state_outputs = {
            self.compiled.output(s["out_index"]).get_any_name(): s for s in spec["states"]
        }
        state_ports = {s["in"] for s in spec["states"]} | set(self.state_outputs)
        self.input_names = [
            i.get_any_name() for i in self.compiled.inputs if i.get_any_name() not in state_ports
        ]
        self.reset()

    def reset(self) -> None:
        self.history: dict[str, collections.deque] = {
            name: collections.deque(maxlen=self.frames) for name in self.input_names
        }
        # States are indexed by absolute frame number: `state_at[f]` is the
        # hidden state after frame f (-1 meaning "before the first frame").
        # Per-frame hidden states, keyed by absolute frame index. Entries before
        # frame 0 are zeros: the stream starts from silence.
        self.state_at: dict[str, dict[int, np.ndarray]] = {
            s["in"]: {-1: np.zeros(s["shape"][1:], np.float32)} for s in self.spec["states"]
        }
        self.state_frames = self.frames - 1  # h history length per GRU
        self.next_frame = 0

    def push(self, **frame: np.ndarray) -> dict[str, np.ndarray] | None:
        """Feed one frame; returns that frame's outputs.

        The window is left-padded with zeros while it fills up. For a live stream
        that padding is the truth (there is no audio before the first frame), so
        the graph runs from the very first frame and its GRU state is primed with
        real data instead of being left at zero until the window fills.
        """
        for name, value in frame.items():
            self.history[name].append(np.ascontiguousarray(value, np.float32))

        end = self.next_frame  # index of the newest frame in the window
        start = end - self.frames + 1

        feed: dict[str, np.ndarray] = {}
        for name in self.input_names:
            axis = self._stream_axis(name)
            frames = list(self.history[name])
            missing = self.frames - len(frames)
            if missing:
                shape = np.shape(frames[0])
                pad = [np.zeros(shape, np.float32)] * missing
                frames = pad + frames
            feed[name] = np.concatenate(frames, axis=axis)
        # Each GRU advances one frame per call and is fed the history of its own
        # outputs for the frames before this one, so its convolutions have context.
        for spec in self.spec["states"]:
            hist = self.state_at[spec["in"]]
            frames = [
                hist.get(f, np.zeros(spec["shape"][1:], np.float32))
                for f in range(end - self.frames + 1, end)
            ]
            feed[spec["in"]] = np.stack(frames).astype(np.float32)

        raw = self.request.infer(feed)
        # Copy: OpenVINO hands back views into the request's internal tensors.
        result = {self.output_names[port]: np.array(v, copy=True) for port, v in raw.items()}
        for out_name, spec in self.state_outputs.items():
            self.state_at[spec["in"]][end] = result[out_name]
            # Keep only what the next window can still need.
            for f in [f for f in self.state_at[spec["in"]] if f < start]:
                del self.state_at[spec["in"]][f]

        self.next_frame += 1
        return {k: v for k, v in result.items() if not self._is_state(k)}

    def _is_state(self, name: str) -> bool:
        return name in self.state_outputs

    def _stream_axis(self, name: str) -> int:
        # [ch, feat, time, freq] for conv features, [ch, time, feat] for emb.
        return 1 if name == "emb" else 2


class Runtime:
    """The three graphs of DeepFilterNet3_ll, driven one frame at a time."""

    def __init__(self, ir_dir: Path, device: str = "NPU", precision: str = "f16"):
        core = ov.Core()
        manifest = json.loads((Path(ir_dir) / "manifest.json").read_text())
        self.specs = {g["graph"]: g for g in manifest["graphs"]}
        self.graphs = {
            name: Graph(
                core, Path(ir_dir) / self.specs[name]["ir"], self.specs[name], device, precision
            )
            for name in GRAPH_ORDER
        }
        self.reset()

    def reset(self) -> None:
        for graph in self.graphs.values():
            graph.reset()

    def push(self, feat_erb: np.ndarray, feat_spec: np.ndarray):
        """Push one frame of features through the encoder.

        Returns the encoder outputs for the whole window, or None while warming
        up. Only the last frame of the window is valid.
        """
        return self.graphs["enc"].push(feat_erb=feat_erb, feat_spec=feat_spec)

    def push_decoders(self, enc: dict[str, np.ndarray]):
        """Feed the encoder's current-frame outputs to both decoders.

        Returns {name: tensor} for the current frame once both decoders have
        enough history, else None.
        """
        dec_in = {
            k: np.take(enc[k], [-1], axis=self._axis(k))
            for k in ("emb", "e0", "e1", "e2", "e3", "c0")
        }
        m = self.graphs["erb_dec"].push(
            emb=dec_in["emb"], e0=dec_in["e0"], e1=dec_in["e1"], e2=dec_in["e2"], e3=dec_in["e3"]
        )
        coefs = self.graphs["df_dec"].push(emb=dec_in["emb"], c0=dec_in["c0"])
        if m is None or coefs is None:
            return None
        return {"m": m["m"], "coefs": coefs["coefs"], "lsnr": enc["lsnr"]}

    @staticmethod
    def _axis(name: str) -> int:
        return 1 if name in ("emb", "lsnr") else 2
