"""NumPy scoring of stored full-color framegen capture code values.

No transfer-function guess or HDR clipping is applied. Per-frame grading and
screen-space temporal residuals retain their distinct nonfinite policies.
Capture admission, sequence ordering, optional LPIPS and reporting belong to
the evaluator; these image-quality metrics do not measure live frame pacing.
"""

import math

import numpy as np


def box_mean(image, radius):
    """Edge-padded box mean using an integral image, with bounded memory."""
    padded = np.pad(image, radius, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64)
    size = 2 * radius + 1
    total = (integral[size:, size:] - integral[:-size, size:]
             - integral[size:, :-size] + integral[:-size, :-size])
    return (total / float(size * size)).astype(np.float32)


def ssim(a, b, radius=3):
    dynamic_range = max(float(np.percentile(b, 99.9) - np.percentile(b, 0.1)), 1e-3)
    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2
    ma, mb = box_mean(a, radius), box_mean(b, radius)
    va = np.maximum(box_mean(a * a, radius) - ma * ma, 0.0)
    vb = np.maximum(box_mean(b * b, radius) - mb * mb, 0.0)
    vab = box_mean(a * b, radius) - ma * mb
    score = ((2.0 * ma * mb + c1) * (2.0 * vab + c2)) / (
        (ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(np.clip(score, -1.0, 1.0).mean())


def luma(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def grade(sample, generated):
    reference = sample["reference"]
    finite = np.isfinite(generated).all(axis=-1) & np.isfinite(reference).all(axis=-1)
    if not finite.all():
        generated = np.where(np.isfinite(generated), generated, 0.0)
        reference = np.where(np.isfinite(reference), reference, 0.0)

    # scRGB can exceed one or go slightly negative. A robust reference scale
    # keeps thresholds meaningful without clipping HDR highlights before grading.
    scale = max(float(np.percentile(np.abs(reference), 99.9)), 1.0)
    delta = np.abs(generated - reference) / scale
    pixel_error = delta.max(axis=-1)
    mse = float(np.mean(((generated - reference) / scale) ** 2))
    psnr = float("inf") if mse == 0.0 else -10.0 * math.log10(mse)

    gen_luma = luma(generated) / scale
    ref_luma = luma(reference) / scale
    grad_g = np.hypot(*np.gradient(gen_luma))
    grad_r = np.hypot(*np.gradient(ref_luma))
    edge = float(np.mean(np.abs(grad_g - grad_r)) / max(float(grad_r.mean()), 1e-4))
    return {
        "mae": float(delta.mean()),
        "bad%": float((pixel_error > 0.05).mean() * 100.0),
        "p95": float(np.percentile(pixel_error, 95.0)),
        "psnr": psnr,
        "ssim": ssim(gen_luma, ref_luma),
        "edge": edge,
        "nonfinite%": float((~finite).mean() * 100.0),
    }


def temporal_residual_change(samples, candidate_index, block_rows=128):
    """Mean screen-space change in prediction error over contiguous probes."""
    total = 0.0
    values = 0
    transitions = 0
    for previous, current in zip(samples, samples[1:]):
        # A successful continuous sequence closes A/B/C, then reuses C as the
        # next A. Do not call a skipped/discontinuous capture temporal data.
        if (current["anchor_id"] != previous["endpoint_id"]
                or current["reference_ns"] <= previous["reference_ns"]):
            continue
        previous_reference = previous["reference"]
        current_reference = current["reference"]
        scale = max(
            float(np.percentile(np.abs(previous_reference), 99.9)),
            float(np.percentile(np.abs(current_reference), 99.9)),
            1.0,
        )
        previous_generated = previous["candidates"][candidate_index][1]
        current_generated = current["candidates"][candidate_index][1]
        for first_row in range(0, current["height"], block_rows):
            rows = slice(first_row, min(first_row + block_rows, current["height"]))
            previous_error = (previous_generated[rows] - previous_reference[rows]) / scale
            current_error = (current_generated[rows] - current_reference[rows]) / scale
            delta = np.abs(current_error - previous_error)
            total += float(delta.sum(dtype=np.float64))
            values += delta.size
        transitions += 1
    return (total / values if values else None), transitions
