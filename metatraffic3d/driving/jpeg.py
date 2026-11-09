"""JPEG helpers for recorded camera frames (no simulator imports, so training needs no MuJoCo)."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def encode(img: np.ndarray, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def decode(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
