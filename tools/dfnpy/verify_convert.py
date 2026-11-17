"""Prove the stateful conversion is faithful.

Reference: the original stateless ONNX graphs, run over a whole sequence at once
(OpenVINO CPU, f32) — the same computation tract performs, minus pulse.

Candidate: the converted IR, driven one frame at a time through `Runtime`, which
threads conv history (a window of W frames) and per-frame GRU state.

Usage:
    verify_convert.py --model-tar ... --ir DIR [--device NPU] [--frames 60]
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import openvino as ov

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert import INPUT_SHAPES  # noqa: E402
from runtime import Runtime  # noqa: E402

# Output name -> axis along which the stream (time) runs.
ENC_OUTPUTS = {"emb": 1, "c0": 2, "lsnr": 1, "e0": 2, "e3": 2}
DEC_OUTPUTS = {"m": 2, "coefs": 1}
MIN_SDR_DB = 20.0  # loose: FP16 NPU output vs FP32 CPU reference


def make_inputs(frames: int, rng) -> dict[str, np.ndarray]:
    return {k: rng.normal(-1.0 if k == "feat_erb" else 0.0, 1.0, s).astype(np.float32) for k, s in INPUT_SHAPES["enc"](frames).items()}


def reference(core, path: Path, name: str, frames: int, feed: dict) -> dict[str, np.ndarray]:
    model = core.read_model(str(path))
    model.reshape(INPUT_SHAPES[name](frames))
    compiled = core.compile_model(model, "CPU")
    names = {compiled.output(i): compiled.output(i).get_any_name() for i in range(len(compiled.outputs))}
    raw = compiled.create_infer_request().infer(feed)
    return {names[port]: np.array(v, copy=True) for port, v in raw.items()}


def sdr_db(ref: np.ndarray, test: np.ndarray) -> float:
    noise = float(np.sum((test.astype(np.float64) - ref) ** 2))
    signal = float(np.sum(ref.astype(np.float64) ** 2))
    return float("inf") if noise == 0 else 10.0 * np.log10(signal / noise)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tar", required=True, type=Path)
    ap.add_argument("--ir", required=True, type=Path)
    ap.add_argument("--device", default="NPU")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--settle",
        type=int,
        default=0,
        help="skip this many leading frames (streaming start-up transient)",
    )
    args = ap.parse_args()

    core = ov.Core()
    rng = np.random.default_rng(args.seed)
    feat = make_inputs(args.frames, rng)
    failures = 0

    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(args.model_tar, "r:gz") as tar:
            tar.extractall(td)
        export = next(Path(td).rglob("config.ini")).parent

        enc_ref = reference(core, export / "enc.onnx", "enc", args.frames, feat)

        rt = Runtime(args.ir, device=args.device)
        collected: dict[str, list[np.ndarray]] = {k: [] for k in ENC_OUTPUTS | DEC_OUTPUTS}
        enc_frames: list[int] = []
        dec_frames: list[int] = []

        for t in range(args.frames):
            enc = rt.graphs["enc"].push(
                feat_erb=np.take(feat["feat_erb"], [t], axis=2),
                feat_spec=np.take(feat["feat_spec"], [t], axis=2),
            )
            if enc is None:
                continue
            enc_frames.append(t)
            for name, axis in ENC_OUTPUTS.items():
                collected[name].append(np.take(enc[name], [-1], axis=axis))
            out = rt.push_decoders(enc)
            if out is None:
                continue
            dec_frames.append(t)
            for name, axis in DEC_OUTPUTS.items():
                collected[name].append(np.take(out[name], [-1], axis=axis))

        print(f"frames compared: enc {len(enc_frames)}, decoders {len(dec_frames)} of {args.frames}")

        def compare(name: str, ref: np.ndarray, axis: int, which: list[int]) -> None:
            nonlocal failures
            keep = [i for i, f in enumerate(which) if f >= args.settle]
            got = np.concatenate([collected[name][i] for i in keep], axis=axis)
            aligned = np.take(ref, [which[i] for i in keep], axis=axis)
            db = sdr_db(aligned, got)
            print(f"  {name:6s} SDR {db:7.2f} dB   max|diff| {np.abs(aligned - got).max():.3e}")
            if db < MIN_SDR_DB:
                failures += 1

        for name, axis in ENC_OUTPUTS.items():
            compare(name, enc_ref[name], axis, enc_frames)

        dec_ref = {
            "m": reference(
                core,
                export / "erb_dec.onnx",
                "erb_dec",
                args.frames,
                {k: np.take(v, range(args.frames), axis=(1 if k == "emb" else 2)) for k, v in enc_ref.items() if k in INPUT_SHAPES["erb_dec"](1)},
            )["m"],
            "coefs": reference(
                core,
                export / "df_dec.onnx",
                "df_dec",
                args.frames,
                {k: np.take(v, range(args.frames), axis=(1 if k == "emb" else 2)) for k, v in enc_ref.items() if k in INPUT_SHAPES["df_dec"](1)},
            )["coefs"],
        }
        for name, axis in DEC_OUTPUTS.items():
            compare(name, dec_ref[name], axis, dec_frames)

    print("FAIL" if failures else "OK: streaming conversion reproduces the stateless graphs")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
