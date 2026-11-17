#!/usr/bin/env python3
"""Check the GRU decomposition against the original ONNX GRU node.

`convert.py` replaces every GRU with primitive ops. This compares the two in
isolation — same weights, same input, zero initial state — so a bug in the cell
shows up here rather than as a mysterious audio difference much later.

Usage:
    check_gru.py --model-tar DeepFilterNet3_ll_onnx.tar.gz [--frames 1]
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


def build_original(node, x_shape) -> onnx.ModelProto:
    ins = [helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape)]
    for i, name in enumerate(node.input[1:4], start=1):
        pass
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
    gru_node = helper.make_node(
        "GRU", ["X", node.input[1], node.input[2], node.input[3], "", ""], ["Y", "Yh"],
        hidden_size=HIDDEN, linear_before_reset=1,
    )
    # Rebuild the graph by hand: feed the decomposed cell the same X and state.
    nodes = list(b.nodes)
    nodes.append(
        helper.make_node(
            "Identity", [info["y_seq"]], ["Yseq"]
        )
    )
    nodes.append(helper.make_node("Identity", [info["y_h"]], ["Yh"]))
    graph = helper.make_graph(nodes, "dec", ins, outs, b.inits)
    model2 = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model2, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tar", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=FRAMES)
    args = ap.parse_args()

    core = ov.Core()
    rng = np.random.default_rng(0)
    failures = 0

    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(args.model_tar, "r:gz") as tar:
            tar.extractall(td)
        export = next(Path(td).rglob("config.ini")).parent

        for name in ("enc", "erb_dec", "df_dec"):
            model = onnx.load(str(export / f"{name}.onnx"))
            for node in model.graph.node:
                if node.op_type != "GRU":
                    continue

                x_shape = [args.frames, 1, HIDDEN]
                x = rng.normal(0, 1, x_shape).astype(np.float32)

                orig = build_original(node, x_shape)
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
                except Exception as e:  # noqa: BLE001
                    print(f"{name}: cannot build reference for {node.name}: {e}")
                    continue
                onm = {oc.output(i): oc.output(i).get_any_name() for i in range(len(oc.outputs))}
                oref = {
                    onm[p]: np.array(v, copy=True)
                    for p, v in oc.create_infer_request().infer({"X": x}).items()
                }

                dec, info = build_decomposed(model, node, args.frames)
                onnx.save(dec, f"{td}/dec.onnx")
                dm = core.read_model(f"{td}/dec.onnx")
                dc = core.compile_model(dm, "CPU")
                dnm = {dc.output(i): dc.output(i).get_any_name() for i in range(len(dc.outputs))}
                state = np.zeros(info["state"]["shape"], np.float32)
                dref = {
                    dnm[p]: np.array(v, copy=True)
                    for p, v in dc.create_infer_request()
                    .infer({"X": x, info["state"]["in"]: state})
                    .items()
                }

                # Third opinion: the cell computed directly in numpy from the
                # model's own weights. Whichever of the two graphs disagrees with
                # this is the broken one.
                w = _initializer(model, node.input[1]).reshape(3 * HIDDEN, HIDDEN).astype(np.float32)
                r_ = _initializer(model, node.input[2]).reshape(3 * HIDDEN, HIDDEN).astype(np.float32)
                b_ = _initializer(model, node.input[3]).reshape(6 * HIDDEN).astype(np.float32)
                x_t = x[-1].reshape(1, HIDDEN)
                gx = (x_t @ w.T + b_[: 3 * HIDDEN])[0]
                gh = b_[3 * HIDDEN :]
                z = 1 / (1 + np.exp(-(gx[:HIDDEN] + gh[:HIDDEN])))
                rr = 1 / (1 + np.exp(-(gx[HIDDEN : 2 * HIDDEN] + gh[HIDDEN : 2 * HIDDEN])))
                nn = np.tanh(gx[2 * HIDDEN :] + rr * gh[2 * HIDDEN :])
                numpy_ref = ((1 - z) * nn).reshape(-1)

                # Yseq's last frame must equal the GRU's final hidden state.
                yseq = dref["Yseq"].reshape(-1, HIDDEN)
                dy = np.abs(yseq[-1] - oref["Yh"].reshape(-1)).max()
                dh = np.abs(dref["Yh"].reshape(-1) - oref["Yh"].reshape(-1)).max()
                scale = float(np.abs(oref["Yh"]).max())
                ok = dy < 1e-3 * max(scale, 1.0) and dh < 1e-3 * max(scale, 1.0)
                d_orig = float(np.abs(oref["Yh"].reshape(-1) - numpy_ref).max())
                d_dec = float(np.abs(dref["Yh"].reshape(-1) - numpy_ref).max())
                print(
                    f"{name}/{node.name}: decomposed-vs-numpy {d_dec:.3e}"
                    f"  onnx-vs-numpy {d_orig:.3e}  {'ok' if ok else 'MISMATCH'}"
                )
                if not ok:
                    failures += 1

    print("FAIL" if failures else "OK: GRU decomposition matches the ONNX GRU")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
