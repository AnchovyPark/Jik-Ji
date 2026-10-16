"""Test frames until LIBERO rollouts are available."""

import numpy as np
from PIL import Image


def synthetic_pair(seed=0, size=224, moved=(40, 56, 48, 48), shift=6):
    """A smooth random scene and a second frame where one block moved by `shift` px."""
    rng = np.random.default_rng(seed)
    low = rng.uniform(0, 255, (8, 8, 3)).astype(np.uint8)
    base = np.asarray(Image.fromarray(low).resize((size, size), Image.BICUBIC)).copy()
    y, x, h, w = moved
    nxt = base.copy()
    nxt[y + shift:y + shift + h, x + shift:x + shift + w] = base[y:y + h, x:x + w] // 2 + 60
    base[y:y + h, x:x + w] = base[y:y + h, x:x + w] // 2 + 60
    return Image.fromarray(base), Image.fromarray(nxt)


def synthetic_stream(n, seed=0, size=224, step=3):
    """n frames of a block sliding across a static background (arm-like motion)."""
    rng = np.random.default_rng(seed)
    low = rng.uniform(0, 255, (8, 8, 3)).astype(np.uint8)
    bg = np.asarray(Image.fromarray(low).resize((size, size), Image.BICUBIC))
    frames = []
    for t in range(n):
        f = bg.copy()
        y, x = 60 + t * step // 2, 30 + t * step
        f[y:y + 40, x:x + 40] = (200, 60, 60)
        frames.append(Image.fromarray(f))
    return frames
