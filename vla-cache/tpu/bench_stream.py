"""Run a frame stream through off / dynamic / static VLA-Cache on TPU and report
latency, XLA compilations, and action-token agreement with the full-prefill run.

  python bench_stream.py --frames 30 --bucket 32
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy import VLACachePolicy  # noqa: E402
from check_equivalence import TASK, load  # noqa: E402
from frames import synthetic_stream  # noqa: E402


def compile_stats():
    import torch_xla.debug.metrics as met

    data = met.metric_data("CompileTime")
    return (data[0], data[1] / 1e9) if data else (0, 0.0)  # count, total seconds


def run(policy, frames):
    policy.reset()
    rows = []
    for i, f in enumerate(frames):
        c0, s0 = compile_stats()
        _, tokens, info = policy.step(f, TASK)
        c1, s1 = compile_stats()
        t = info["timing"]
        rows.append(dict(step=i, tokens=tokens.tolist(), rows=info.get("rows"), n_reusable=info.get("n_reusable"),
                         prefill_ms=t["prefill"] * 1e3, decode_ms=t["decode"] * 1e3, host_ms=t["host_plan"] * 1e3,
                         compiles=c1 - c0, compile_s=s1 - s0))
        print(f"  step {i:3d} rows={info.get('rows')} prefill={rows[-1]['prefill_ms']:.1f}ms "
              f"decode={rows[-1]['decode_ms']:.1f}ms compiles={c1 - c0} ({s1 - s0:.1f}s)", flush=True)
    return rows


def summarize(name, rows, ref=None, skip=2):
    steady = [r for r in rows[skip:] if r["compiles"] == 0]
    out = dict(mode=name, steps=len(rows), compiles=sum(r["compiles"] for r in rows),
               compile_s=sum(r["compile_s"] for r in rows),
               steady_steps=len(steady),
               prefill_ms_median=float(np.median([r["prefill_ms"] for r in steady])) if steady else None,
               e2e_ms_median_all=float(np.median([r["prefill_ms"] + r["decode_ms"] for r in rows[skip:]])))
    if ref is not None:
        out["token_match"] = float(np.mean([np.mean(np.array(a["tokens"]) == np.array(b["tokens"]))
                                            for a, b in zip(rows, ref)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/models/openvla-7b-finetuned-libero-spatial"))
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--bucket", type=int, default=32)
    ap.add_argument("--modes", default="off,static,dynamic")
    ap.add_argument("--out", default="results/bench_stream.json")
    args = ap.parse_args()

    import torch_xla.core.xla_model as xm

    model, proc = load(args.ckpt, torch.bfloat16)
    frames = synthetic_stream(args.frames)
    from jikji.openvla_static import StaticOpenVLA

    static_model = StaticOpenVLA(model, xm.xla_device())
    results, ref = {}, None
    for mode in args.modes.split(","):
        print(f"== {mode}", flush=True)
        pol = VLACachePolicy(model, proc, xm.xla_device(), mode=mode, bucket=args.bucket,
                             static_model=static_model)
        rows = run(pol, frames)
        if mode == "off":
            ref = rows
        results[mode] = dict(summary=summarize(mode, rows, ref), rows=rows)
        print(json.dumps(results[mode]["summary"]), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
