"""OpenVLA (Prismatic DINOv2+SigLIP -> Llama-2-7B) as fixed-shape graphs.

Every graph has a shape that is fixed at compile time, so on an AOT/static-graph
backend (TPU via PyTorch/XLA here) each distinct shape is compiled exactly once:

  vision(pixels[1,6,224,224])            -> vision embeds [1, 256, d]
  prefill(embeds[1,P,d], rows=(n_1..n_k)) -> first action token + attention stats
  decode(token, pos)                      -> next token

KV lives in a fixed buffer [n_layers][1, H, S, Dh] (S = prompt slots + decode slots).
Slot index == RoPE position, so a reused vision token's K/V from the previous frame
is valid as-is (VLA-Cache keeps image token positions fixed across frames).

VLA-Cache reuse = at a few layers ("pruning points") the hidden-state rows of reused
tokens are dropped (gather with a fixed-size index), and only the remaining rows
write K/V into the buffer (index_copy). Attention always reads the full buffer with a
causal mask built from positions, so reused slots supply last frame's K/V.
The row count at each pruning point is the only shape that varies; the host rounds it
to a bucket so the number of compiled graphs stays bounded.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

NUM_PATCHES = 256
VISION_START = 1  # slot 0 is BOS
TEXT_START = VISION_START + NUM_PATCHES
# VLA-Cache reads text->vision attention of text positions [257, 257 + 35) (hard-coded in the original)
ORIG_TEXT_LEN = 35
ATTN_LAYER = 15


@dataclass
class PrefillOut:
    token: torch.Tensor  # [1] int, first generated token
    entropy: torch.Tensor  # [n_layers] mean token entropy of head-averaged attention
    text2vision: torch.Tensor  # [256] mean text->vision attention at ATTN_LAYER


class StaticOpenVLA:
    def __init__(self, hf_model, device, dtype=torch.bfloat16, text_slots=48, decode_slots=8):
        cfg = hf_model.language_model.config
        self.device, self.dtype = device, dtype
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = getattr(cfg, "num_key_value_heads", self.n_heads)
        self.head_dim = cfg.hidden_size // self.n_heads
        self.eps = cfg.rms_norm_eps
        self.rope_theta = getattr(cfg, "rope_theta", 10000.0)
        self.prompt_slots = TEXT_START + text_slots
        self.n_slots = self.prompt_slots + decode_slots
        self.pad_token_id = cfg.pad_token_id if cfg.pad_token_id is not None else 32000

        self.vision_backbone = hf_model.vision_backbone.to(device=device, dtype=dtype).eval()
        self.projector = hf_model.projector.to(device=device, dtype=dtype).eval()

        lm = hf_model.language_model.model
        to = lambda t: t.detach().to(device=device, dtype=dtype)
        self.embed = to(lm.embed_tokens.weight)
        self.final_norm = to(lm.norm.weight)
        self.lm_head = to(hf_model.language_model.lm_head.weight)
        self.layers = []
        for layer in lm.layers:
            a, m = layer.self_attn, layer.mlp
            self.layers.append(
                dict(
                    ln1=to(layer.input_layernorm.weight),
                    ln2=to(layer.post_attention_layernorm.weight),
                    wqkv=to(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0)),
                    wo=to(a.o_proj.weight),
                    wgu=to(torch.cat([m.gate_proj.weight, m.up_proj.weight], 0)),
                    wd=to(m.down_proj.weight),
                )
            )
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.inv_freq = inv_freq.to(device)
        self.slot_pos = torch.arange(self.n_slots, device=device)
        self.reset_kv()

    def reset_kv(self):
        shape = (1, self.n_kv_heads, self.n_slots, self.head_dim)
        self.k_buf = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(self.n_layers)]
        self.v_buf = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(self.n_layers)]

    # ---------------------------------------------------------------- building blocks
    def _rms(self, x, w):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return w * xf.to(x.dtype)

    def _rope(self, x, pos):
        # x [1, H, n, Dh], pos [n]; HF Llama rotate_half convention, fp32 cos/sin like HF
        freqs = pos.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], -1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def _layer(self, i, h, pos, want_probs=False):
        """h [1, n, d], pos [n] (also the KV slot of each row). Returns h and head-mean probs [n, S]."""
        p = self.layers[i]
        n = h.shape[1]
        H, Hk, Dh = self.n_heads, self.n_kv_heads, self.head_dim
        x = self._rms(h, p["ln1"])
        q, k, v = (x @ p["wqkv"].t()).split([H * Dh, Hk * Dh, Hk * Dh], -1)
        q = self._rope(q.view(1, n, H, Dh).transpose(1, 2), pos)
        k = self._rope(k.view(1, n, Hk, Dh).transpose(1, 2), pos)
        v = v.view(1, n, Hk, Dh).transpose(1, 2)
        self.k_buf[i] = self.k_buf[i].index_copy(2, pos, k)
        self.v_buf[i] = self.v_buf[i].index_copy(2, pos, v)
        kk, vv = self.k_buf[i], self.v_buf[i]
        if Hk != H:
            kk = kk.repeat_interleave(H // Hk, 1)
            vv = vv.repeat_interleave(H // Hk, 1)
        scores = (q @ kk.transpose(-1, -2)) / math.sqrt(Dh)
        allowed = self.slot_pos[None, :] <= pos[:, None]  # [n, S]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores.float(), dim=-1).to(self.dtype)
        o = (probs @ vv).transpose(1, 2).reshape(1, n, H * Dh)
        h = h + o @ p["wo"].t()
        x = self._rms(h, p["ln2"])
        g, u = (x @ p["wgu"].t()).chunk(2, -1)
        h = h + (F.silu(g) * u) @ p["wd"].t()
        return h, (probs.float().mean(1)[0] if want_probs else None)

    @staticmethod
    def _entropy(attn, row_valid):
        # VLA-Cache get_layer_mask_schedule: renormalize rows, token entropy, mean over rows
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-10)
        ent = -(attn * torch.log(attn + 1e-10)).sum(-1)
        w = row_valid.float()
        return (ent * w).sum() / w.sum()

    def _logits_argmax(self, h_row):
        x = self._rms(h_row, self.final_norm)
        return (x @ self.lm_head.t()).float().argmax(-1)

    # ---------------------------------------------------------------- graphs
    @torch.no_grad()
    def encode_image(self, pixel_values):
        return self.projector(self.vision_backbone(pixel_values.to(self.device, self.dtype)))

    def build_embeds(self, input_ids, vision_embeds):
        """input_ids: [1, T] host tensor starting with BOS. Returns ([1, P, d], prompt_len)."""
        ids = input_ids[0].tolist()
        text = ids[1:]
        assert len(text) <= self.prompt_slots - TEXT_START, f"prompt has {len(text)} text tokens"
        text = text + [self.pad_token_id] * (self.prompt_slots - TEXT_START - len(text))
        bos = torch.tensor([ids[0]], device=self.device)
        txt = torch.tensor(text, device=self.device)
        emb = torch.cat([self.embed[bos][None], vision_embeds, self.embed[txt][None]], 1)
        return emb, len(ids) + NUM_PATCHES

    @torch.no_grad()
    def prefill(self, embeds, prompt_len, keep_idx=None, last_row=None, pruning_loc=(2, 6, 9, 11)):
        """embeds [1, P, d]. keep_idx: None (full) or one device index tensor per pruning point,
        indexing into the rows alive before that point. last_row: row of the last prompt token
        among the rows alive after the final pruning point (device scalar)."""
        h = embeds
        pos = self.slot_pos[: self.prompt_slots]
        prompt_len_t = torch.tensor(prompt_len, device=self.device)
        text_hi = torch.clamp(prompt_len_t, max=TEXT_START + ORIG_TEXT_LEN)
        entropies, t2v = [], None
        ki = 0
        for i in range(self.n_layers):
            if keep_idx is not None and ki < len(pruning_loc) and i == pruning_loc[ki]:
                h = h.index_select(1, keep_idx[ki])
                pos = pos.index_select(0, keep_idx[ki])
                ki += 1
            h, attn = self._layer(i, h, pos, want_probs=True)
            entropies.append(self._entropy(attn, pos < prompt_len_t))
            if i == ATTN_LAYER:
                text_rows = ((pos >= TEXT_START) & (pos < text_hi)).float()
                vis = attn[:, VISION_START:TEXT_START]
                t2v = (vis * text_rows[:, None]).sum(0) / text_rows.sum()
        if last_row is None:
            last_row = prompt_len_t - 1
        token = self._logits_argmax(h[:, last_row])
        return PrefillOut(token=token, entropy=torch.stack(entropies), text2vision=t2v)

    @torch.no_grad()
    def decode(self, token, pos):
        """token [1] device int, pos [1] device int (slot to write)."""
        h = self.embed[token][None]
        for i in range(self.n_layers):
            h, _ = self._layer(i, h, pos)
        return self._logits_argmax(h[:, -1])
