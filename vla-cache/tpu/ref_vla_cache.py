"""Run the ORIGINAL VLA-Cache code path (vla-cache/src/openvla + transformers fork) on CPU
over a frame stream and dump per-step action tokens. Needs the reference venv:

  ~/venvs/vlacache-ref/bin/python ref_vla_cache.py --frames 6 --out results/ref_tokens.json
"""

import argparse
import json
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OPENVLA = os.path.join(HERE, "..", "src", "openvla")
sys.path.insert(0, OPENVLA)
# prismatic/__init__ pulls in training deps (dlimp, RLDS); inference only needs prismatic.extern.hf
_pkg = types.ModuleType("prismatic")
_pkg.__path__ = [os.path.join(OPENVLA, "prismatic")]
sys.modules["prismatic"] = _pkg

from frames import synthetic_stream  # noqa: E402
from check_equivalence import TASK  # noqa: E402
from experiments.robot.openvla_utils import get_vla_action  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor  # noqa: E402


# The fork's LlamaModel.forward times itself with CUDA events; make them no-ops on CPU.
if not torch.cuda.is_available():
    class _Event:
        def __init__(self, *a, **k): pass
        def record(self, *a, **k): pass
        def elapsed_time(self, *a, **k): return 0.0
    torch.cuda.Event = _Event
    torch.cuda.synchronize = lambda *a, **k: None


class _CastingProcessor:
    """get_vla_action casts inputs with .to(DEVICE, dtype=torch.bfloat16); redirect that cast."""

    def __init__(self, proc, dtype):
        self.proc, self.dtype, self.tokenizer = proc, dtype, proc.tokenizer

    def __call__(self, *args, **kwargs):
        feats, dtype = self.proc(*args, **kwargs), self.dtype

        class _Feats(dict):
            def to(self, device, dtype_=None, **kw):
                return feats.to(device, dtype=dtype)

        return _Feats(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/models/openvla-7b-finetuned-libero-spatial"))
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "ref_tokens.json"))
    ap.add_argument("--dtype", default="float32", help="CPU bf16 kernels are near single-threaded here")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    vla = OpenVLAForActionPrediction.from_pretrained(args.ckpt, torch_dtype=dtype).eval()
    vla.language_model.model.all_FLOPs = 0
    vla.norm_stats = json.load(open(os.path.join(args.ckpt, "dataset_statistics.json")))
    from transformers import AutoTokenizer
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor

    proc = PrismaticProcessor(image_processor=PrismaticImageProcessor.from_pretrained(args.ckpt),
                              tokenizer=AutoTokenizer.from_pretrained(args.ckpt))
    if dtype != torch.bfloat16:
        proc = _CastingProcessor(proc, dtype)
    cfg = types.SimpleNamespace(use_vla_cache=True)

    frames = synthetic_stream(args.frames)
    last_caches, prev, out = None, None, []
    for i, f in enumerate(frames):
        obs = {"full_image": np.asarray(f), "prev_image": np.asarray(prev if prev is not None else f)}
        action, last_caches, _ = get_vla_action(cfg, vla, proc, "openvla", obs, TASK, "libero_spatial",
                                                center_crop=False, last_caches=last_caches)
        out.append(dict(step=i, action=np.asarray(action).tolist()))
        print(i, np.asarray(action).round(4).tolist(), flush=True)
        prev = f
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
