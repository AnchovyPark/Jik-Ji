"""Trace statistics (step 1) and bucket-policy inefficiency (step 3).

  python analyze_buckets.py trace   --traces results/traces
  python analyze_buckets.py buckets --traces results/traces --bench results/bench_rows.json

Latency model for a row tuple (n_1..n_k at the pruning points): layers before the first point
always see all P rows, segment s has `seg_len[s]` layers at n_s rows, and
  L(n_1..n_k) = L(P) - sum_s seg_len[s] * (L_u(P) - L_u(n_s)) / n_pruned_layers
where L_u(n) is the measured prefill latency with all pruned layers at n rows (bench_rows.py).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vla_cache_host as host  # noqa: E402

PRUNING_LOC = (2, 6, 9, 11)
N_LAYERS = 32
P = 305  # prompt slots (BOS + 256 vision + 48 text slots)


def load_traces(d):
    eps = []
    for f in sorted(glob.glob(os.path.join(d, "task*_ep*.npz"))):
        z = np.load(f, allow_pickle=False)
        eps.append(dict(task_id=int(z["task_id"]), episode=int(z["episode"]), success=bool(z["success"]),
                        plans=json.loads(str(z["plans"])), n_frames=len(z["frames"])))
    return eps


def steps_of(eps):
    for e in eps:
        for i, p in enumerate(e["plans"]):
            if "reusable" in p:
                yield e, i, p


def rows_for(p, bucket):
    props = dict(zip(PRUNING_LOC, p["proportions"]))
    _, _, counts = host.plan_rows(p["reusable"], props, P, p["prompt_len"] - 1, PRUNING_LOC, bucket)
    return tuple(counts)


def trace_stats(eps):
    st = list(steps_of(eps))
    n_static = np.array([p["n_static"] for _, _, p in st])
    n_reuse = np.array([p["n_reusable"] for _, _, p in st])
    exact = [rows_for(p, 1) for _, _, p in st]
    dropped = np.array([P - r[-1] for r in exact])
    q = lambda a: dict(mean=float(a.mean()), p10=float(np.percentile(a, 10)), p50=float(np.median(a)),
                       p90=float(np.percentile(a, 90)), min=int(a.min()), max=int(a.max()))
    return dict(episodes=len(eps), successes=sum(e["success"] for e in eps),
                steps_total=sum(e["n_frames"] for e in eps), steps_with_plan=len(st),
                static_patches=q(n_static), reusable_tokens=q(n_reuse), rows_dropped_last_segment=q(dropped),
                unique_exact_row_tuples=len(set(exact)),
                per_task={t: dict(reusable_mean=float(np.mean([p["n_reusable"] for e, _, p in st if e["task_id"] == t])))
                          for t in sorted({e["task_id"] for e in eps})})


def latency_model(bench):
    pts = sorted(bench["points"], key=lambda r: r["rows"])
    xs = np.array([r["rows"] for r in pts], float)
    ys = np.array([r["median_ms"] for r in pts])
    full = bench["full"]["median_ms"]
    l_p = np.interp(P, xs, ys)
    bounds = list(PRUNING_LOC) + [N_LAYERS]
    seg_len = [bounds[i + 1] - bounds[i] for i in range(len(PRUNING_LOC))]
    n_pruned = N_LAYERS - PRUNING_LOC[0]

    def lat(rows):
        return full - sum(sl * (l_p - np.interp(n, xs, ys)) / n_pruned for sl, n in zip(seg_len, rows))

    # cost of a new shape for an online runtime = first call (dynamo trace + XLA compile + run)
    compile_s = float(np.median([r["first_call_s"] for r in pts]))
    return lat, full, compile_s


def bucket_report(eps, bench, buckets=(1, 4, 8, 16, 32, 64), tuples_bench=None):
    st = list(steps_of(eps))
    model_lat, full, compile_s = latency_model(bench)
    measured, calib = {}, 1.0
    if tuples_bench is not None:
        # measured graphs (real nested gathers) override the model; the rest use the model
        # scaled by the median measured/model saving ratio
        full = tuples_bench["full"]["median_ms"]
        measured = {tuple(r["rows"]): r["median_ms"] for r in tuples_bench["tuples"]}
        ratios = [(full - v) / (full - model_lat(t)) for t, v in measured.items() if full - model_lat(t) > 0.5]
        calib = float(np.median(ratios)) if ratios else 1.0

    def lat(rows):
        rows = tuple(rows)
        if rows in measured:
            return measured[rows]
        return full - calib * (full - model_lat(rows))

    n_steps = len(st)
    exact = [rows_for(p, 1) for _, _, p in st]
    ideal = np.array([lat(r) for r in exact])
    out = dict(full_prefill_ms=full, compile_s_per_graph=compile_s, steps=n_steps, saving_calibration=calib,
               ideal_prefill_ms=float(ideal.mean()), policies=[])
    for b in buckets:
        rows = [rows_for(p, b) for _, _, p in st]
        l = np.array([lat(r) for r in rows])
        graphs = len(set(rows))
        wasted = np.array([sum(sl * (r - e) for sl, r, e in zip((4, 3, 2, 21), rr, ee)) / 30
                           for rr, ee in zip(rows, exact)])
        out["policies"].append(dict(
            bucket=b, graphs=graphs, measured_step_frac=float(np.mean([r in measured for r in rows])), compile_s_total=graphs * compile_s,
            prefill_ms=float(l.mean()), vs_ideal_pct=float(100 * (l.mean() / ideal.mean() - 1)),
            vs_full_pct=float(100 * (l.mean() / full - 1)),
            wasted_rows_per_layer=float(wasted.mean()),
            # an online "dynamic" runtime pays the compile on first sight of each shape
            dynamic_amortized_ms=float(l.mean() + 1e3 * graphs * compile_s / n_steps)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["trace", "buckets"])
    ap.add_argument("--traces", default=os.path.join(HERE, "results", "traces"))
    ap.add_argument("--bench", default=os.path.join(HERE, "results", "bench_rows.json"))
    ap.add_argument("--tuples-bench", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    eps = load_traces(args.traces)
    tb = json.load(open(args.tuples_bench)) if args.tuples_bench else None
    res = trace_stats(eps) if args.stage == "trace" else bucket_report(eps, json.load(open(args.bench)), tuples_bench=tb)
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
