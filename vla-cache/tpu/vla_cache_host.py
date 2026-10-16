"""Host-side VLA-Cache decisions (numpy), ported from vla-cache experiments/robot/vla_cache_utils.py.

Everything that depends on the frame (which tokens to reuse, how many rows each layer
keeps) is decided here. The device only sees fixed-size index tensors, and
`plan_rows` rounds the row counts to a bucket so the set of graph shapes is bounded.
"""

import numpy as np

GRID = 16  # 224 / 14


def _patchify(image, patch_size=14):
    a = np.asarray(image, dtype=np.float32)
    h, w, c = a.shape
    g = h // patch_size
    return a.reshape(g, patch_size, g, patch_size, c).transpose(0, 2, 1, 3, 4).reshape(g * g, -1)


def find_static_patches(img_0, img_1, patch_size=14, top_k=130, sim_threshold=0.996):
    """Patch ids (0..255) whose raw-pixel cosine across frames >= threshold, top_k by similarity."""
    p1, p2 = _patchify(img_0, patch_size), _patchify(img_1, patch_size)
    sim = (p1 * p2).sum(1) / (np.linalg.norm(p1, axis=1) * np.linalg.norm(p2, axis=1) + 1e-8)
    ids = np.nonzero(sim >= sim_threshold)[0]
    ids = ids[np.argsort(-sim[ids], kind="stable")]
    return ids[:top_k].tolist()


def top_attention_patches(text2vision, top_k=120):
    """Task-relevant patch ids: top_k by text->vision attention of the previous step."""
    return np.argsort(-np.asarray(text2vision, dtype=np.float32), kind="stable")[:top_k].tolist()


def layer_schedule(entropies, growth_factor=0.55):
    """Per-layer reuse proportion from previous-step attention entropy (last layer dropped, as original)."""
    e = np.asarray(entropies[:-1], dtype=np.float64)
    reuse = 1.0 - (e - e.min()) / (e.max() - e.min() + 1e-10)
    for i in range(1, len(reuse)):
        delta = reuse[i] - reuse[i - 1]
        if delta > 0:
            reuse[i] = reuse[i - 1] + delta * growth_factor
    return reuse.astype(np.float32)


def reusable_tokens(static_patches, task_patches, vision_start=1):
    """Token positions to reuse: static minus task-relevant, sorted by index (original order)."""
    return sorted(p + vision_start for p in set(static_patches) - set(task_patches))


def plan_rows(reusable, proportions, n_rows, last_pos, pruning_loc=(2, 6, 9, 11), bucket=1):
    """Rows kept at each pruning point.

    Matches the original: at layer l, the first max(1, int(prop[l] * |R|)) tokens of R are
    dropped, and a pruning point only takes effect if it drops more than the previous one.
    With bucket > 1, the kept-row count is rounded up to a multiple of `bucket` by keeping
    (= recomputing) some reusable tokens, which only reuses less.

    Returns (keep_idx list, last_row, kept_counts). keep_idx[s] indexes rows alive before point s.
    """
    alive = np.arange(n_rows)
    keep_idx, counts = [], []
    k_applied = 0
    for loc in pruning_loc:
        k = max(1, int(proportions[loc] * len(reusable)))
        k_applied = max(k_applied, k)
        n = n_rows - k_applied
        n = min(n_rows, -(-n // bucket) * bucket)
        dropped = np.asarray(reusable[: n_rows - n], dtype=np.int64)
        new_alive = alive[~np.isin(alive, dropped)]
        keep_idx.append(np.searchsorted(alive, new_alive))
        counts.append(len(new_alive))
        alive = new_alive
    last_row = int(np.searchsorted(alive, last_pos))
    assert alive[last_row] == last_pos
    return keep_idx, last_row, counts
