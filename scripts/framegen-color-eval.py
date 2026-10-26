#!/usr/bin/env python3
"""Evaluate full-resolution held-out Gamescope frame-generation captures.

GAMESCOPE_FRAMEGEN_RECORD_COLOR records real triplets A/B/C while presenting
all three normally. The framegen GPU predicts B from A and C at B's measured
phase, never presents those predictions, and writes paired occlusion strengths
0/0.5/1, or paired endpoint-trace strengths selected at capture time, beside
the exact real B in one GSCF file. This provides genuine full-colour ground
truth without relying on a repeatable second run or pretending an endpoint is
an intermediate frame.

Higher PSNR/SSIM and lower MAE/bad/edge/residual-change are better. Core metrics
grade the captured output code values; EOTF metadata is reported but no guessed
HDR display transform is applied. Optional LPIPS is used only when explicitly
requested and importable; the core evaluator is numpy-only. Capture file writes
perturb presentation cadence, so these are image-quality measurements, not a
live pacing benchmark.

Usage:
  framegen-color-eval.py --data /path/to/gscf-directory
  framegen-color-eval.py --data /path/to/gscf-directory --per-frame
  framegen-color-eval.py --data /path/to/gscf-directory --lpips
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np


# Preserve direct execution and absolute-path imports from analysis consumers.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from framegen_color_capture import (
    MAGIC, VERSION, SUPPORTED_VERSIONS, HEADER, CANDIDATE_KINDS,
    ARGB8888, XRGB8888, ABGR8888, XBGR8888,
    ARGB2101010, XRGB2101010, ABGR2101010, XBGR2101010,
    ABGR16161616, XBGR16161616, ABGR16161616F, XBGR16161616F,
    ABGR32323232F, FORMAT_BPP, fourcc, fourcc_name, decode_plane, load_sample,
)


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


def lpips_scores(samples, candidate_index):
    try:
        import lpips
        import torch
    except ImportError as exc:
        print(f"LPIPS unavailable ({exc}); install torch + lpips or omit --lpips",
              file=sys.stderr)
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex").to(device).eval()
    scores = []
    with torch.no_grad():
        for sample in samples:
            scale = max(float(np.percentile(np.abs(sample["reference"]), 99.9)), 1.0)
            tensors = []
            for rgb in (sample["candidates"][candidate_index][1], sample["reference"]):
                rgb = np.clip(rgb / scale, 0.0, 1.0)
                tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)[None]).float().to(device) * 2.0 - 1.0)
            scores.append(float(model(tensors[0], tensors[1]).item()))
    return scores


def format_row(name, metrics):
    return (f"{name}: mae={metrics['mae']:.6f} bad={metrics['bad%']:.3f}% "
            f"p95={metrics['p95']:.5f} psnr={metrics['psnr']:.3f}dB "
            f"ssim={metrics['ssim']:.5f} edge={metrics['edge']:.5f} "
            f"nonfinite={metrics['nonfinite%']:.4f}%")


def candidate_label(kind, strength):
    return "configured" if strength is None else f"{kind}={strength:g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="directory containing color_*.gscf")
    parser.add_argument("--per-frame", action="store_true", help="print every sample")
    parser.add_argument("--lpips", action="store_true", help="also run optional torch/lpips")
    args = parser.parse_args()

    files = sorted(Path(args.data).glob("color_*.gscf"))
    if not files:
        sys.exit(f"no color_*.gscf captures under {args.data}")
    samples = []
    for path in files:
        try:
            samples.append(load_sample(path))
        except ValueError as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
    if not samples:
        sys.exit("no valid GSCF samples")

    shape_keys = {(s["width"], s["height"], s["format"], s["eotf"]) for s in samples}
    if len(shape_keys) != 1:
        sys.exit("capture mixes dimensions/formats/EOTFs; evaluate each homogeneous set separately")

    candidate_strengths = [candidate[0] for candidate in samples[0]["candidates"]]
    if any([candidate[0] for candidate in sample["candidates"]] != candidate_strengths
           for sample in samples):
        sys.exit("capture mixes candidate layouts")
    candidate_kinds = {sample["candidate_kind"] for sample in samples}
    if len(candidate_kinds) != 1:
        sys.exit("capture mixes candidate sweep kinds")
    candidate_kind = next(iter(candidate_kinds))
    rows_by_candidate = []
    for candidate_index in range(len(candidate_strengths)):
        rows_by_candidate.append([
            grade(sample, sample["candidates"][candidate_index][1]) for sample in samples
        ])
    if args.per_frame:
        for sample_index, sample in enumerate(samples):
            for candidate_index, strength in enumerate(candidate_strengths):
                label = candidate_label(candidate_kind, strength)
                row = rows_by_candidate[candidate_index][sample_index]
                print(f"{sample['path'].name} phase={sample['phase']:.4f} {label} "
                      + format_row("", row).lstrip(": "))

    width, height, drm_format, eotf = next(iter(shape_keys))
    phases = np.asarray([sample["phase"] for sample in samples], np.float64)
    print(f"loaded {len(samples)} exact held-out pairs at {width}x{height}, "
          f"format={fourcc_name(drm_format)!r}, EOTF={eotf}, "
          f"phase mean/range={phases.mean():.4f}/{phases.min():.4f}-{phases.max():.4f}")
    for candidate_index, strength in enumerate(candidate_strengths):
        rows = rows_by_candidate[candidate_index]
        summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        summary["sample_mae_sigma"] = float(np.std([row["mae"] for row in rows]))
        label = candidate_label(candidate_kind, strength)
        print(format_row(label, summary))
        residual_change, transitions = temporal_residual_change(samples, candidate_index)
        residual_text = "n/a" if residual_change is None else f"{residual_change:.6f}"
        print(f"  sample MAE sigma={summary['sample_mae_sigma']:.6f}; "
              f"temporal residual-change MAE={residual_text} ({transitions} transitions)")
        if candidate_index > 0:
            deltas = np.asarray([
                row["mae"] - baseline["mae"]
                for row, baseline in zip(rows, rows_by_candidate[0])
            ])
            print(f"  paired MAE delta vs first candidate={deltas.mean():+.6f}; "
                  f"wins/ties/losses={(deltas < 0).sum()}/{(deltas == 0).sum()}/{(deltas > 0).sum()}")

    if args.lpips:
        for candidate_index, strength in enumerate(candidate_strengths):
            scores = lpips_scores(samples, candidate_index)
            if scores is not None:
                label = candidate_label(candidate_kind, strength)
                print(f"{label}: lpips={np.mean(scores):.6f} "
                      f"sample_sigma={np.std(scores):.6f}")


if __name__ == "__main__":
    main()
