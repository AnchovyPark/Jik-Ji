"""Our implementation (CPU, exact row counts = original semantics) vs. the original VLA-Cache
actions dumped by ref_vla_cache.py on the same frame stream.

  ~/venvs/vlacache-ref/bin/python ref_vla_cache.py --frames 6
  PJRT_DEVICE=CPU ~/venvs/jikji/bin/python compare_ref.py --frames 6
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_equivalence import TASK, load  # noqa: E402
from frames import synthetic_stream  # noqa: E402
from policy import VLACachePolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/models/openvla-7b-finetuned-libero-spatial"))
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--ref", default=os.path.join(HERE, "results", "ref_tokens.json"))
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    ref = json.load(open(args.ref))
    model, proc = load(args.ckpt, dtype)
    pol = VLACachePolicy(model, proc, torch.device("cpu"), mode="dynamic", dtype=dtype)
    for i, f in enumerate(synthetic_stream(args.frames)):
        action, tokens, info = pol.step(f, TASK)
        r = np.asarray(ref[i]["action"])
        print(f"step {i} rows={info.get('rows')} reusable={info.get('n_reusable')} "
              f"{'MATCH' if np.allclose(action, r) else 'DIFF'}  ours={np.round(action, 4).tolist()}  ref={np.round(r, 4).tolist()}")


if __name__ == "__main__":
    main()
