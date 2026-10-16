"""Split one prefill step into host tracing time vs device execution time."""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_equivalence import TASK, load  # noqa: E402
from frames import synthetic_stream  # noqa: E402
from openvla_static import NUM_PATCHES, StaticOpenVLA  # noqa: E402


def main():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

    dev = xm.xla_device()
    model, proc = load(os.path.expanduser("~/models/openvla-7b-finetuned-libero-spatial"), torch.bfloat16)
    m = StaticOpenVLA(model, dev)
    img = synthetic_stream(1)[0]
    ids = proc.tokenizer(f"In: What action should the robot take to {TASK}?\nOut:", return_tensors="pt").input_ids
    ids = torch.cat([ids, torch.tensor([[29871]])], 1)
    pixels = proc.image_processor(img, return_tensors="pt")["pixel_values"]
    xm.mark_step(); xm.wait_device_ops()

    for it in range(6):
        met.clear_all()
        t0 = time.perf_counter()
        emb, plen = m.build_embeds(ids, pixels)
        ta = time.perf_counter()
        xm.mark_step()
        tb = time.perf_counter()
        xm.wait_device_ops()
        tc = time.perf_counter()
        out = m.prefill(emb, plen)
        t1 = time.perf_counter()
        print(f"  vision: trace {1e3*(ta-t0):.1f}ms mark_step {1e3*(tb-ta):.1f}ms device {1e3*(tc-tb):.1f}ms | prefill call {1e3*(t1-tc):.1f}ms")
        xm.mark_step()
        t2 = time.perf_counter()
        xm.wait_device_ops()
        t3 = time.perf_counter()
        tok = out.token.cpu()
        t4 = time.perf_counter()
        ex = met.metric_data("ExecuteTime")
        print(f"iter {it}: trace {1e3*(t1-t0):.1f}ms  mark_step {1e3*(t2-t1):.1f}ms  device {1e3*(t3-t2):.1f}ms  "
              f"fetch {1e3*(t4-t3):.1f}ms  ExecuteTime {ex[1]/1e6 if ex else 0:.1f}ms ({ex[0] if ex else 0} graphs) "
              f"compiles {met.metric_data('CompileTime')[0] if met.metric_data('CompileTime') else 0}", flush=True)
    print(met.short_metrics_report())


if __name__ == "__main__":
    main()
