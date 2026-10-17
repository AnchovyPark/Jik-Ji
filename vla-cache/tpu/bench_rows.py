"""Prefill latency vs. recomputed row count on one TPU chip.

For each n, compile a prefill graph where layers 0-1 see all P prompt rows and layers 2..31
see n rows (one drop at the first pruning point, identity at the others), and time it.
Also records compile time per shape. Output feeds analyze_buckets.py.

  python bench_rows.py --chip 0 --lo 160 --step 8 --iters 30
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/models/openvla-7b-finetuned-libero-spatial"))
    ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--lo", type=int, default=160)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "bench_rows.json"))
    ap.add_argument("--tuples", default=None, help="json file: list of row tuples to measure with real gathers")
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    from run_libero_tpu import pin_chip

    pin_chip(args.chip)
    import numpy as np
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

    from check_equivalence import TASK, load
    from frames import synthetic_stream
    from openvla_static import NUM_PATCHES, StaticOpenVLA

    dev = xm.xla_device()
    model, proc = load(args.ckpt, torch.bfloat16)
    m = StaticOpenVLA(model, dev)
    P = m.prompt_slots
    ids = proc.tokenizer(f"In: What action should the robot take to {TASK}?\nOut:", return_tensors="pt").input_ids
    ids = torch.cat([ids, torch.tensor([[29871]])], 1)
    plen = ids.shape[1] + NUM_PATCHES
    pixels = proc.image_processor(synthetic_stream(1)[0], return_tensors="pt")["pixel_values"]
    emb, _ = m.build_embeds(ids, pixels)
    xm.mark_step(); xm.wait_device_ops()

    def run(keep, last_row):
        out = m.prefill(emb, plen, keep_idx=keep, last_row=last_row)
        xm.mark_step(); xm.wait_device_ops()
        return out

    def timed(keep, last_row):
        c0 = met.metric_data("CompileTime")
        c0 = (c0[0], c0[1]) if c0 else (0, 0)
        t = time.perf_counter()
        run(keep, last_row)
        first = time.perf_counter() - t
        c1 = met.metric_data("CompileTime")
        compile_s = (c1[1] - c0[1]) / 1e9 if c1 else 0.0
        lat = []
        for _ in range(args.iters):
            t = time.perf_counter()
            run(keep, last_row)
            lat.append((time.perf_counter() - t) * 1e3)
        return dict(first_call_s=first, compile_s=compile_s, median_ms=float(np.median(lat)),
                    p10_ms=float(np.percentile(lat, 10)), p90_ms=float(np.percentile(lat, 90)))

    results = dict(prompt_slots=P, prompt_len=plen, pruning_loc=list(m.pruning_loc), points=[])
    r = timed(None, None)
    results["full"] = r
    print(f"full (no pruning graph): {r}", flush=True)
    vision = list(range(1, 1 + NUM_PATCHES))
    if args.tuples:
        # real VLA-Cache graph structure: one nested gather per pruning point
        results["tuples"] = []
        for rows in json.load(open(args.tuples)):
            alive = np.arange(P)
            keep = []
            for n in rows:
                new_alive = np.array([p for p in range(P) if p not in set(vision[: P - n])])
                keep.append(torch.from_numpy(np.searchsorted(alive, new_alive)).to(dev))
                alive = new_alive
            r = timed(keep, int(np.searchsorted(alive, plen - 1)))
            r["rows"] = list(rows)
            results["tuples"].append(r)
            print(f"tuple {rows}: {r}", flush=True)
        json.dump(results, open(args.out, "w"), indent=1)
        return
    # drop vision tokens from the end of the image block; keep BOS, text and last token
    for n in list(range(args.lo, P, args.step)) + [P]:
        drop = set(vision[: P - n])
        alive = np.array([p for p in range(P) if p not in drop])
        ident = np.arange(n)
        keep = [torch.from_numpy(alive).to(dev)] + [torch.from_numpy(ident).to(dev) for _ in m.pruning_loc[1:]]
        last_row = int(np.searchsorted(alive, plen - 1))
        r = timed(keep, last_row)
        r["rows"] = n
        results["points"].append(r)
        print(f"rows {n}: {r}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
