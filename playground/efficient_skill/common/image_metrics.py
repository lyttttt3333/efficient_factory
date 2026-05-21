from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def compare_images(reference: Path, candidate: Path) -> dict[str, float]:
    ref = load_rgb(reference)
    cand = load_rgb(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"Image shape mismatch: {reference} {ref.shape} vs {candidate} {cand.shape}")

    diff = cand - ref
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    psnr = float("inf") if mse == 0.0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    return {
        "mse": mse,
        "mae": mae,
        "max_abs": max_abs,
        "psnr": psnr,
    }

