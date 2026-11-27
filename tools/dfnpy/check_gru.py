#!/usr/bin/env python3
"""Check the GRU decomposition against the original ONNX GRU node.

`convert.py` replaces every GRU with primitive ops. This compares the two in
isolation — same weights, same input, zero initial state — so a bug in the cell
shows up here rather than as a mysterious audio difference much later.

The converted cell advances exactly one frame per inference and carries the
previous `frames - 1` hidden states as an explicit input, so it is driven the way
`runtime.py` drives it: a left-padded window ending at the current frame, and the
state history rolled forward one entry per call. The ONNX GRU, by contrast, runs
the whole window in one go. Comparing the two means comparing the sequence the
GRU produced with the sequence the cell built up.

Usage:
    check_gru.py --model-tar DeepFilterNet3_ll_onnx.tar.gz [--frames 6]
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import onnx
import openvino as ov
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert import GraphBuilder, decompose_gru, _initializer  # noqa: E402

HIDDEN = 512
FRAMES = 6  # the window convert.py uses for enc; decoders use 3

# The cell is exact to float rounding: every GRU checked measures ~1e-6 against
# the ONNX node, on a state whose entries reach ~1.
TOLERANCE = 2e-4


def build_original(node, x_shape) -> onnx.ModelProto:
    ins = [helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape)]
    outs = [
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("Yh", TensorProto.FLOAT, None),
    ]
    gru = helper.make_node(
        "GRU",
        ["X", node.input[1], node.input[2], node.input[3], "", "zero"],
        ["Y", "Yh"],
        hidden_size=HIDDEN,
        linear_before_reset=1,
    )
    zero = helper.make_tensor(
        "zero", TensorProto.FLOAT, [1, 1, HIDDEN], np.zeros((1, 1, HIDDEN), np.float32)
    )
    graph = helper.make_graph([gru], "orig", ins, outs, [zero])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def build_decomposed(model, node, frames) -> tuple[onnx.ModelProto, dict]:
    # Feed the cell from a graph input instead of the model's internal tensor.
    patched = helper.make_node(
        "GRU",
        ["X", node.input[1], node.input[2], node.input[3], "", ""],
        list(node.output),
        name=node.name,
        hidden_size=HIDDEN,
        linear_before_reset=1,
    )
    b = GraphBuilder("t")
    info = decompose_gru(model, patched, b, frames)
    ins = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [frames, 1, HIDDEN])]
    ins.append(
        helper.make_tensor_value_info(
            info["state"]["in"], TensorProto.FLOAT, info["state"]["shape"]
        )
    )
    outs = [
        helper.make_tensor_value_info("Yseq", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("Yh", TensorProto.FLOAT, None),
    ]
    # Rebuild the graph by hand: feed the decomposed cell the same X and state.
    nodes = list(b.nodes)
    nodes.append(helper.make_node("Identity", [info["y_seq"]], ["Yseq"]))
    nodes.append(helper.make_node("Identity", [info["y_h"]], ["Yh"]))
    graph = helper.make_graph(nodes, "dec", ins, outs, b.inits)
    model2 = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model2, info


def drive_cell(request, outputs, state_name, history, window):
    """One inference of the decomposed cell, returning a name -> array dict.

    `window` is the left-padded frame window as `runtime.py` builds it, and
    `history` the previous `frames - 1` hidden states (oldest first).
    """
    feed = {"X": window, state_name: np.stack(history)}
    raw = request.infer(feed)
    return {outputs[port]: np.array(value, copy=True) for port, value in raw.items()}


def numpy_reference(model, node, x: np.ndarray) -> np.ndarray:
    """The cell rolled forward over the window in numpy, from the model's weights.

    Independent of both graphs, so whichever of them disagrees with this is the
    broken one. This has to roll exactly like the decomposed cell does — one step
    per frame — not evaluate a single step, which would be comparing a frame of a
    recurrence against its sixth iteration.
    """
    w = _initializer(model, node.input[1]).reshape(3 * HIDDEN, HIDDEN).astype(np.float32)
    r_ = _initializer(model, node.input[2]).reshape(3 * HIDDEN, HIDDEN).astype(np.float32)
    b_ = _initializer(model, node.input[3]).reshape(6 * HIDDEN).astype(np.float32)
    h = np.zeros(HIDDEN, np.float32)
    for frame in x:
        gx = frame.reshape(1, HIDDEN) @ w.T + b_[: 3 * HIDDEN]
        gh = h @ r_.T + b_[3 * HIDDEN :]
        z = 1 / (1 + np.exp(-(gx[0, :HIDDEN] + gh[:HIDDEN])))
        r = 1 / (1 + np.exp(-(gx[0, HIDDEN : 2 * HIDDEN] + gh[HIDDEN : 2 * HIDDEN])))
        n = np.tanh(gx[0, 2 * HIDDEN :] + r * gh[2 * HIDDEN :])
        h = (1 - z) * n + z * h
    return h.reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tar", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=FRAMES)
    args = ap.parse_args()

    core = ov.Core()
    rng = np.random.default_rng(0)
    failures = 0
    checked = 0
    skipped = 0
    frames = args.frames

    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(args.model_tar, "r:gz") as tar:
            tar.extractall(td)
        export = next(Path(td).rglob("config.ini")).parent

        for name in ("enc", "erb_dec", "df_dec"):
            model = onnx.load(str(export / f"{name}.onnx"))
            for node in model.graph.node:
                if node.op_type != "GRU":
                    continue

                x = rng.normal(0, 1, [frames, 1, HIDDEN]).astype(np.float32)

                orig = build_original(node, [frames, 1, HIDDEN])
                # The original graph needs the weight initializers from the model.
                wanted = {node.input[1], node.input[2], node.input[3]}
                for init in model.graph.initializer:
                    if init.name in wanted:
                        orig.graph.initializer.append(init)
                for cnode in model.graph.node:
                    if cnode.op_type == "Constant" and cnode.output[0] in wanted:
                        orig.graph.node.append(cnode)
                onnx.save(orig, f"{td}/orig.onnx")
                try:
                    om = core.read_model(f"{td}/orig.onnx")
                    oc = core.compile_model(om, "CPU")
                    onm = {oc.output(i): oc.output(i).get_any_name() for i in range(len(oc.outputs))}
                    oref = {
                        onm[p]: np.array(v, copy=True)
                        for p, v in oc.create_infer_request().infer({"X": x}).items()
                    }
                except Exception as e:  # noqa: BLE001
                    print(f"{name}: cannot build reference for {node.name}: {e}")
                    skipped += 1
                    continue

                dec, info = build_decomposed(model, node, frames)
                onnx.save(dec, f"{td}/dec.onnx")
                dm = core.read_model(f"{td}/dec.onnx")
                dc = core.compile_model(dm, "CPU")
                dnm = {dc.output(i): dc.output(i).get_any_name() for i in range(len(dc.outputs))}
                request = dc.create_infer_request()

                # Drive the cell frame by frame, exactly as runtime.py does.
                history = [np.zeros((1, HIDDEN), np.float32) for _ in range(frames - 1)]
                sequence = np.empty((frames, HIDDEN), np.float32)
                for t in range(frames):
                    window = np.zeros((frames, 1, HIDDEN), np.float32)
                    window[frames - 1 - t :] = x[: t + 1]
                    out = drive_cell(request, dnm, info["state"]["in"], history, window)
                    sequence[t] = out["Yh"].reshape(-1)
                    history = history[1:] + [out["Yh"]]

                # The ONNX GRU's Y is [frames, 1, 1, HIDDEN]: one hidden state per
                # frame, which is exactly what the loop above built up.
                y = oref["Y"].reshape(frames, HIDDEN)
                final = oref["Yh"].reshape(-1)
                numpy_ref = numpy_reference(model, node, x)

                dy = float(np.abs(sequence - y).max())
                dh = float(np.abs(sequence[-1] - final).max())
                d_orig = float(np.abs(final - numpy_ref).max())
                d_dec = float(np.abs(sequence[-1] - numpy_ref).max())
                ok = dy < TOLERANCE and d_dec < TOLERANCE
                print(
                    f"{name}/{node.name}: sequence-vs-onnx {dy:.3e}"
                    f"  final-vs-onnx {dh:.3e}"
                    f"  decomposed-vs-numpy {d_dec:.3e}"
                    f"  onnx-vs-numpy {d_orig:.3e}"
                    f"  {'ok' if ok else 'MISMATCH'}"
                )
                if ok:
                    checked += 1
                else:
                    failures += 1

    # A check that compared nothing must not report success: a broken reference,
    # a renamed weight or an OpenVINO that refuses the GRU node would otherwise
    # leave this exiting 0 with every GRU skipped.
    print(f"checked {checked}, skipped {skipped}, failed {failures}")
    if not checked or failures:
        print("FAIL")
        return 1
    print("OK: GRU decomposition matches the ONNX GRU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
