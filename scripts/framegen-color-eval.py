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


from framegen_color_metrics import (
    box_mean, ssim, luma, grade, temporal_residual_change,
)


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
