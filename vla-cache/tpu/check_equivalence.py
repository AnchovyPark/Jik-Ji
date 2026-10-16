"""Check StaticOpenVLA against HF OpenVLA (full prefill), on CPU fp32 and on TPU bf16.

  python check_equivalence.py --ckpt ~/models/openvla-7b-finetuned-libero-spatial [--tpu]
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy import VLACachePolicy  # noqa: E402
from frames import synthetic_pair  # noqa: E402

TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def load(ckpt, dtype):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(ckpt, torch_dtype=dtype, trust_remote_code=True).eval()
    return model, proc


@torch.no_grad()
def hf_tokens(model, proc, image, unnorm_key="libero_spatial"):
    prompt = f"In: What action should the robot take to {TASK}?\nOut:"
    inputs = proc(prompt, image)
    ids = inputs["input_ids"]
    if ids[0, -1] != 29871:
        ids = torch.cat([ids, torch.tensor([[29871]])], 1)
    out = model.generate(input_ids=ids, pixel_values=inputs["pixel_values"].to(model.dtype),
                         max_new_tokens=model.get_action_dim(unnorm_key), do_sample=False)
    return out[0, -model.get_action_dim(unnorm_key):].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/models/openvla-7b-finetuned-libero-spatial"))
    ap.add_argument("--tpu", action="store_true")
    args = ap.parse_args()

    img, _ = synthetic_pair()
    if not args.tpu:
        model, proc = load(args.ckpt, torch.float32)
        ref = hf_tokens(model, proc, img)
        pol = VLACachePolicy(model, proc, torch.device("cpu"), mode="off", dtype=torch.float32)
        _, mine, _ = pol.step(img, TASK)
        print("HF   cpu fp32:", ref)
        print("ours cpu fp32:", mine, "MATCH" if np.array_equal(ref, mine) else "MISMATCH")
    else:
        import torch_xla.core.xla_model as xm

        model, proc = load(args.ckpt, torch.bfloat16)
        ref = hf_tokens(model, proc, img)
        pol = VLACachePolicy(model, proc, xm.xla_device(), mode="off")
        _, mine, info = pol.step(img, TASK)
        print("HF   cpu bf16:", ref)
        print("ours tpu bf16:", mine, "MATCH" if np.array_equal(ref, mine) else "MISMATCH")


if __name__ == "__main__":
    main()
