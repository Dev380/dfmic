#!/usr/bin/env python3
"""Convert the upstream DeepFilterNet3_ll ONNX graphs into stateful, static-shape
OpenVINO IR that the NPU plugin can compile.

Why this exists
---------------
The upstream graphs are *stateless*: they carry the GRU recurrence along the
sequence axis and rely on tract's `pulse` transform to hold state between calls.
The NPU has no such transform, so state has to be explicit. Two things in these
graphs are stateful:

  1. the GRUs (one per encoder stage / decoder stage, 6 in total), and
  2. the causal convolutions along the time axis, which need their previous
     kernel-1 frames of input (tract's pulse keeps that history for them).

Both are handled here by pinning the graph to a fixed window of `frames` frames:

  * every GRU is decomposed into `frames` unrolled primitive cells (MatMul/Add/
    Slice/Sigmoid/Tanh) whose hidden state enters and leaves the graph as a
    normal tensor, and
  * the window itself carries the conv history: the caller feeds the last
    `frames` frames and keeps only the final frame's output.

The caller therefore threads (a) each GRU's hidden state and (b) a window of
past frames. See `runtime.py` for that side of the contract.

Usage:
    convert.py --model-tar DeepFilterNet3_ll_onnx.tar.gz --out DIR [--frames 4]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Per-graph static input shapes for a given frame count W. These mirror what
# libDF feeds the pulsed tract models (see libDF/src/tract.rs::process_raw).
INPUT_SHAPES = {
    "enc": lambda L: {"feat_erb": [1, 1, L, 32], "feat_spec": [1, 2, L, 96]},
    "erb_dec": lambda L: {
        "emb": [1, L, 512],
        "e3": [1, 64, L, 8],
        "e2": [1, 64, L, 8],
        "e1": [1, 64, L, 16],
        "e0": [1, 64, L, 32],
    },
    "df_dec": lambda L: {"emb": [1, L, 512], "c0": [1, 64, L, 96]},
}


def _attr(node, name, default):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT:
                return a.i
            if a.type == onnx.AttributeProto.STRING:
                return a.s.decode()
            return a
    return default


class GraphBuilder:
    """Accumulates nodes/initializers for the rewritten graph."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.nodes: list = []
        self.inits: list = []
        self._n = 0

    def name(self, hint: str) -> str:
        self._n += 1
        return f"{self.prefix}.{hint}.{self._n}"

    def const(self, value) -> str:
        nm = self.name("const")
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(np.asarray(value)), nm))
        return nm

    def i64(self, *values: int) -> str:
        return self.const(np.array(values, np.int64))

    def op(self, op_type: str, inputs: list[str], hint: str, **attrs) -> str:
        out = self.name(hint)
        self.nodes.append(helper.make_node(op_type, inputs, [out], name=out, **attrs))
        return out


def _initializer(model: onnx.ModelProto, name: str) -> np.ndarray:
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant" and name in node.output:
            for a in node.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
    raise KeyError(f"cannot resolve initializer {name!r}")


def decompose_gru(model: onnx.ModelProto, node, b: GraphBuilder, frames: int) -> dict:
    """Replace one ONNX GRU node with a single primitive cell plus explicit state.

        z = sigmoid(X @ Wz^T + Wbz + H @ Rz^T + Rbz)
        r = sigmoid(X @ Wr^T + Wbr + H @ Rr^T + Rbr)
        n = tanh(X @ Wh^T + Wbh + r * (H @ Rh^T + Rbh))    # linear_before_reset=1
        H' = (1 - z) * n + z * H

    The GRU advances exactly one frame per inference, no matter how long the
    window is: feeding it a padded window would make it advance over frames that
    do not exist yet (a GRU with non-zero biases does not stay put on zero
    input), which desynchronises it from the audio.

    Its *output* history is threaded as state instead, so that the causal
    convolutions downstream still see `frames` frames of context:

        state in : h[-(frames-1) .. -1]   shape [frames-1, 1, hidden]
        state out: h'                     shape [1, 1, hidden]
        Y        : concat(h[-(frames-1)..-1], h')  shape [frames, 1, 1, hidden]
    """
    hidden = _attr(node, "hidden_size", 0)
    direction = _attr(node, "direction", "forward")
    if direction != "forward":
        raise NotImplementedError(f"{direction} GRU is not supported")
    if _attr(node, "layout", 0) != 0:
        raise NotImplementedError("layout=1 GRU is not supported")

    weights = _initializer(model, node.input[1])  # [1, 3H, IN]
    recur = _initializer(model, node.input[2])  # [1, 3H, H]
    bias = _initializer(model, node.input[3]) if node.input[3] else None  # [1, 6H]
    H = hidden
    n_in = weights.shape[-1]

    weights = weights.reshape(3 * H, n_in)
    recur = recur.reshape(3 * H, H)
    if bias is not None:
        bias = bias.reshape(6 * H)
        wb, rb = bias[: 3 * H], bias[3 * H :]
    else:
        wb, rb = np.zeros(3 * H, np.float32), np.zeros(3 * H, np.float32)

    hist_in = f"{b.prefix}.{node.output[1]}.hist"
    x_all = b.op("Reshape", [node.input[0], b.i64(frames, n_in)], "x_all")
    # Only the newest frame of the window is a real input: earlier frames of the
    # encoder window come from convolutions that lacked their own history.
    x_t = b.op("Slice", [x_all, b.i64(frames - 1), b.i64(frames), b.i64(0)], "x_t")

    h_prev = b.op("Reshape", [b.op("Slice", [hist_in, b.i64(frames - 2), b.i64(frames - 1), b.i64(0)], "h_prev3"), b.i64(1, H)], "h_prev")

    w_t, r_t = b.const(weights.T.copy()), b.const(recur.T.copy())
    wb_c, rb_c = b.const(wb.astype(np.float32)), b.const(rb.astype(np.float32))
    gx = b.op("Add", [b.op("MatMul", [x_t, w_t], "gx_mm"), wb_c], "gx")
    gh = b.op("Add", [b.op("MatMul", [h_prev, r_t], "gh_mm"), rb_c], "gh")

    def gate(src: str, i: int, tag: str) -> str:
        return b.op("Slice", [src, b.i64(i * H), b.i64((i + 1) * H), b.i64(1)], tag)

    gx_z, gx_r, gx_h = (gate(gx, i, g) for i, g in enumerate(("gx_z", "gx_r", "gx_h")))
    gh_z, gh_r, gh_h = (gate(gh, i, g) for i, g in enumerate(("gh_z", "gh_r", "gh_h")))
    z = b.op("Sigmoid", [b.op("Add", [gx_z, gh_z], "pre_z")], "z")
    r = b.op("Sigmoid", [b.op("Add", [gx_r, gh_r], "pre_r")], "r")
    n = b.op("Tanh", [b.op("Add", [gx_h, b.op("Mul", [r, gh_h], "rn")], "pre_n")], "n")
    one = b.const(np.ones((1, H), np.float32))
    h_new = b.op(
        "Add",
        [
            b.op("Mul", [b.op("Sub", [one, z], "omz"), n], "keep"),
            b.op("Mul", [z, h_prev], "carry"),
        ],
        "h_new",
    )

    return {
        "node": node,
        # [frames, 1, 1, H]: the window's GRU outputs, ending with this frame.
        "y_seq": b.op(
            "Concat",
            [b.op("Reshape", [hist_in, b.i64(frames - 1, 1, 1, H)], "hist_seq"),
             b.op("Reshape", [h_new, b.i64(1, 1, 1, H)], "new_seq")],
            "y_seq",
            axis=0,
        ),
        # [1, H]: one frame's state, which the caller stacks into the history.
        "y_h": b.op("Reshape", [h_new, b.i64(1, H)], "y_h"),
        "state": {"gru": node.name, "in": hist_in, "shape": [frames - 1, 1, H]},
    }


# Frames of history each graph needs, set by its causal convolutions along time.
# Measured against the offline reference: the encoder needs 6 frames (e3 only
# converges at P=6), the decoders 3.
GRAPH_FRAMES = {"enc": 6, "erb_dec": 3, "df_dec": 3}


def rewrite(model: onnx.ModelProto, graph_name: str, frames: int) -> list[dict]:
    """Swap every GRU for unrolled primitives; return the state descriptors."""
    graph = model.graph
    builder = GraphBuilder(graph_name)
    states: list[dict] = []
    rename: dict[str, str] = {}
    new_nodes = []

    for node in graph.node:
        if node.op_type != "GRU":
            new_nodes.append(node)
            continue
        info = decompose_gru(model, node, builder, frames)
        rename[node.output[0]] = info["y_seq"]
        rename[node.output[1]] = info["y_h"]
        states.append({**info["state"], "out": info["y_h"]})
        new_nodes.extend(builder.nodes)
        builder.nodes = []

    for node in new_nodes:
        for i, inp in enumerate(node.input):
            if inp in rename:
                node.input[i] = rename[inp]

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(builder.inits)

    for state in states:
        graph.input.append(
            helper.make_tensor_value_info(state["in"], TensorProto.FLOAT, state["shape"])
        )
        graph.output.append(
            helper.make_tensor_value_info(
                state["out"], TensorProto.FLOAT, state["shape"][1:]
            )
        )


    _prune_unused(graph)
    onnx.checker.check_model(model)
    return states


def _prune_unused(graph) -> int:
    """Remove nodes/initializers whose outputs are no longer reachable."""
    needed = {o.name for o in graph.output}
    keep = []
    removed = 0
    for node in reversed(graph.node):
        if any(o in needed for o in node.output):
            keep.append(node)
            needed.update(i for i in node.input if i)
        else:
            removed += 1
    keep.reverse()
    del graph.node[:]
    graph.node.extend(keep)

    inits = [i for i in graph.initializer if i.name in needed]
    removed += len(graph.initializer) - len(inits)
    del graph.initializer[:]
    graph.initializer.extend(inits)
    return removed


def _read_onnx(model: onnx.ModelProto, core):
    """Hand OpenVINO the graph through a scratch file, and no further.

    `read_model` reads from disk, so the graph is written to a temporary directory
    that is removed as soon as it has been parsed. The model it returns is fully
    in memory, so nothing downstream needs the file again.
    """
    with tempfile.TemporaryDirectory() as staging:
        scratch = Path(staging) / "graph.onnx"
        onnx.save(model, str(scratch))
        return core.read_model(str(scratch))


def convert_one(src: Path, out_dir: Path, name: str, frames: int, fp16: bool, core) -> dict:
    import openvino as ov

    model = onnx.load(str(src))
    states = rewrite(model, name, frames)
    # (state output indices are filled in below, after the model is read)

    ov_model = _read_onnx(model, core)

    ov_model.reshape(INPUT_SHAPES[name](frames))

    # OpenVINO folds trivial ops (a Reshape on the state output, for example) and
    # renames the result after its producer, which would break the names the
    # runtime looks up. Re-assert the ONNX graph's output names positionally.
    # OpenVINO preserves output *order* even when it folds ops and renames the
    # results, so state outputs are tracked positionally (see runtime.py).
    onnx_outputs = [o.name for o in model.graph.output]
    for state in states:
        state["out_index"] = onnx_outputs.index(state["out"])
    target = out_dir / f"{name}.xml"
    ov.save_model(ov_model, str(target), compress_to_fp16=fp16)

    return {
        "graph": name,
        "ir": target.name,
        "frames": frames,
        "inputs": {i.get_any_name(): str(i.partial_shape) for i in ov_model.inputs},
        "outputs": {o.get_any_name(): str(o.partial_shape) for o in ov_model.outputs},
        "states": states,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tar", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=None, help="override the per-graph window")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="fp16", action="store_false")
    args = ap.parse_args()

    import openvino as ov

    core = ov.Core()
    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(args.model_tar, "r:gz") as tar:
            tar.extractall(td)
        export = next(Path(td).rglob("config.ini")).parent
        print(f"extracted {args.model_tar.name} -> {export}")

        manifest = {"source": args.model_tar.name, "graphs": []}
        for name in ("enc", "erb_dec", "df_dec"):
            frames = GRAPH_FRAMES[name] if args.frames is None else args.frames
            print(f"  converting {name} (window {frames} frames)")
            manifest["graphs"].append(
                convert_one(export / f"{name}.onnx", args.out, name, frames, args.fp16, core)
            )
        shutil.copy(export / "config.ini", args.out / "config.ini")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out}/manifest.json")


if __name__ == "__main__":
    main()
