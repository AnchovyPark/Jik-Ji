"""OpenVLA + VLA-Cache policy loop on a static-graph device.

mode:
  "off"     full prefill every step (vendor-runtime baseline)
  "dynamic" VLA-Cache with exact row counts -> a new graph shape almost every step
  "static"  VLA-Cache with row counts rounded to `bucket` -> bounded graph set
"""

import time

import numpy as np
import torch

import vla_cache_host as host
from openvla_static import NUM_PATCHES, VISION_START, StaticOpenVLA

EMPTY_TOKEN = 29871  # OpenVLA appends this after "Out:" (predict_action)


def _sync(device):
    if device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
        xm.wait_device_ops()


class VLACachePolicy:
    def __init__(self, hf_model, processor, device, mode="static", bucket=32, unnorm_key="libero_spatial",
                 static_top_k=130, task_top_k=120, sim_threshold=0.996, dtype=torch.bfloat16, static_model=None,
                 record_plan=False):
        self.hf, self.processor, self.device = hf_model, processor, device
        self.model = static_model or StaticOpenVLA(hf_model, device, dtype=dtype)
        self.mode, self.bucket, self.unnorm_key = mode, bucket, unnorm_key
        self.static_top_k, self.task_top_k, self.sim_threshold = static_top_k, task_top_k, sim_threshold
        self.action_dim = hf_model.get_action_dim(unnorm_key)
        # mode "off" + record_plan: execute full prefill but log the exact VLA-Cache plan (trace collection)
        self.record_plan = record_plan
        self.reset()

    def reset(self):
        self.model.reset_kv()
        self.prev_image, self.prev_stats, self.prev_prompt = None, None, None

    # Per-episode state, so one device model can interleave several concurrent episodes.
    def get_state(self):
        return dict(prev_image=self.prev_image, prev_stats=self.prev_stats, prev_prompt=self.prev_prompt,
                    k_buf=self.model.k_buf, v_buf=self.model.v_buf)

    def set_state(self, state):
        if state is None:
            self.reset()
            return
        self.prev_image, self.prev_stats, self.prev_prompt = state["prev_image"], state["prev_stats"], state["prev_prompt"]
        self.model.k_buf, self.model.v_buf = state["k_buf"], state["v_buf"]

    def _input_ids(self, task_label):
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids
        if ids[0, -1] != EMPTY_TOKEN:
            ids = torch.cat([ids, torch.tensor([[EMPTY_TOKEN]])], 1)
        return ids

    def _plan(self, image, prompt_len):
        """Returns (keep_idx, last_row, info) or None for a full prefill."""
        if (self.mode == "off" and not self.record_plan) or self.prev_stats is None:
            return None
        entropy, t2v = self.prev_stats
        static = host.find_static_patches(image, self.prev_image, top_k=self.static_top_k,
                                          sim_threshold=self.sim_threshold)
        task = host.top_attention_patches(t2v, self.task_top_k)
        reusable = host.reusable_tokens(static, task, VISION_START)
        if not reusable:
            return None
        bucket = self.bucket if self.mode == "static" else 1
        props = host.layer_schedule(entropy)
        keep_idx, last_row, counts = host.plan_rows(reusable, props, self.model.prompt_slots, prompt_len - 1,
                                                    pruning_loc=self.model.pruning_loc, bucket=bucket)
        info = dict(n_static=len(static), n_reusable=len(reusable), rows=counts)
        if self.record_plan:
            info.update(reusable=reusable, proportions=[float(props[l]) for l in self.model.pruning_loc])
        return keep_idx, last_row, info

    @torch.no_grad()
    def step(self, image, task_label):
        """image: PIL RGB 224x224 (already center-cropped if the checkpoint needs it)."""
        t = {}
        t0 = time.perf_counter()
        ids = self._input_ids(task_label)
        if self.prev_prompt is not None and not torch.equal(ids, self.prev_prompt):
            self.reset()
        pixels = self.processor.image_processor(image, return_tensors="pt")["pixel_values"]
        plan = self._plan(image, ids.shape[1] + NUM_PATCHES)
        plan_info = plan[2] if plan is not None else {}
        if self.mode == "off":
            plan = None
        t["host_plan"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        embeds, prompt_len = self.model.build_embeds(ids, pixels)
        if plan is None:
            out = self.model.prefill(embeds, prompt_len)
            info = dict(rows=None, **({"plan": plan_info} if plan_info else {}))
        else:
            keep_idx, last_row, info = plan
            keep = [torch.from_numpy(k).to(self.device) for k in keep_idx]
            out = self.model.prefill(embeds, prompt_len, keep_idx=keep, last_row=last_row)
        _sync(self.device)
        t["prefill"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        tokens = [out.token]
        for j in range(self.action_dim - 1):
            pos = torch.tensor([prompt_len + j], device=self.device)
            tokens.append(self.model.decode(tokens[-1], pos))
            _sync(self.device)
        token_ids = torch.cat(tokens).cpu().numpy()
        entropy, t2v = out.entropy.float().cpu().numpy(), out.text2vision.float().cpu().numpy()
        t["decode"] = time.perf_counter() - t0

        self.prev_image, self.prev_stats, self.prev_prompt = image, (entropy, t2v), ids
        return self._detokenize(token_ids), token_ids, dict(timing=t, prompt_len=prompt_len, **info)

    def _detokenize(self, token_ids):
        bins = self.hf.vocab_size - token_ids
        bins = np.clip(bins - 1, 0, self.hf.bin_centers.shape[0] - 1)
        normed = self.hf.bin_centers[bins]
        stats = self.hf.get_action_stats(self.unnorm_key)
        mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
        hi, lo = np.array(stats["q99"]), np.array(stats["q01"])
        return np.where(mask, 0.5 * (normed + 1) * (hi - lo) + lo, normed)
