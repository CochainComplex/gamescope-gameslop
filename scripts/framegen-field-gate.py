#!/usr/bin/env python3
"""Offline gate for motion-field changes, on captured consecutive frames.

A numpy port of the production chain decides, per content class and before
any runtime code, whether a field arm helps each generation method:

  field   luma pair (on the composite's code values, like production's
          srgbView(), which is the UNORM view; 4 bilinear taps per block),
          2x pyramid to /32, R=4 search at /32, vector-median seeded +/-1
          refine, zero anchor, confidence and sub-texel fit, forward-backward
          check (cs_framegen_motion_{luma_pair,pyramid,match,match_refine,
          fbcheck}.comp)
  output  predictive: warp.comp (second-source agreement; sharp fallback:
          displacement scaled by confidence, as production's default);
          bidir: bidir.comp (phase weights, mixedness x agreement,
          crossfade fallback)

Ground truth from triplets A, B, C of distinct consecutive frames:
predictive predicts C from B along A->B, bidir rebuilds B from A and C.
Per class and method, against the first arm, paired over triplets:

  E_chg   abs error where the frame changed (|truth - fallback| > 4/255)
  sta     abs error on static pixels (|truth - fallback| <= 1/255): drag
  lost    edges in the truth the output lacks (smear)
  false   edges the output invents (tearing, block shear)

Captures come from framegen-capture.sh (NAME-WxH.gray). Arms:
  checked/8, checked/4       production field at /8 or /4 (tolerances in px)
  ...+vm                     3x3 vector median on the final field
  ...+nc                     confidence kept only where a vector agrees with
                             its 3x3 vector median within the FB tolerance
  learned/8                  checked/8 refined by a GSFR net (--net)
  ...@xf                     predictive output with the crossfade fallback
                             (GAMESCOPE_FRAMEGEN_SHARP_FALLBACK=0)

  framegen-field-gate.py --size 1280x720 --reference-model subtexel8-fp32-rtz [--stride 6] [--arms a,b,...] [--domain coded|linear] NAME...

The port's numbers count only while it reproduces the GPU. --parity DIR runs
the checked/8 chain on the lumas of GAMESCOPE_FRAMEGEN_RECORD samples (GSFD)
and compares with both fields the GPU recorded; it fails below 99.9% identical
vectors (max-component error < 0.02), above a mean confidence error of 0.01 in any field, or on non-finite
values. Explicit sampler assumptions and half-float pass storage are empirical;
GSFD v1 does not record numerical provenance. Validate the selected model
against captures from the matching implementation before interpreting rankings.

  framegen-field-gate.py --parity DIR --reference-model subtexel8-fp32-rtz
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Path-based consumers do not automatically put sibling modules on sys.path.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Keep the existing G.* analysis API while sharing one numerical implementation.
from framegen_field_reference import (
    AGREE_HI, AGREE_LO, COARSEST, FB_SLOPE, FB_TOL, REFERENCE_MODEL, REG,
    SUPPRESS_HI, SUPPRESS_LO, Numerics,
    _conf_subpixel, _cur9, _fetch, _match, _neighbours, _refine, _sad,
    base_luma, bilinear, checked_pair, fbcheck, field_from_base, half,
    half_store, interpolate, linear, post, predict, sample_field, smoothstep,
    vector_median,
)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load("framegen_net_train", "framegen-net-train.py")
NE = _load("framegen_net_eval", "framegen-net-eval.py")


def add_numerics_arguments(parser):
    parser.add_argument("--reference-model", choices=(REFERENCE_MODEL,),
                        help="explicit empirical sampler and image-storage assumptions; no device applicability is inferred")


def numerics_from_args(parser, args):
    if args.reference_model is None:
        parser.error("--reference-model is required; GSFD v1 carries no numerical provenance. "
                     "Qualify the selected assumptions on matching captures before interpreting offline comparisons.")
    return Numerics(args.reference_model)


# ---------------------------------------------------------------- arms

def arm_fields(prev_l, cur_l, arm, net, *, numerics):
    """Checked forward (cur-anchored) and reverse fields, and the scale."""
    s = 4 if "/4" in arm else 8
    pb, cb = base_luma(prev_l, s), base_luma(cur_l, s)
    X, Y = checked_pair(pb, cb, s, numerics=numerics)
    tol = FB_TOL * 8 / s
    for suffix in arm.split("+")[1:]:
        X, Y = (post(X, suffix, tol), post(Y, suffix, tol))
    if arm.startswith("learned"):
        Fh, ch = NE.refine(net, X, Y, pb, cb)
        X = np.concatenate([Fh, ch[..., None], X[..., 3:]], -1).astype(np.float32)
    return X, Y, s


# ---------------------------------------------------------------- metrics

def grad(img):
    gy, gx = np.gradient(img)
    return np.hypot(gx, gy)


def score(pred, truth, fallback):
    d = np.abs(truth - fallback)
    chg, sta = d > 4 / 255, d <= 1 / 255
    if chg.sum() < 500:
        return None
    e = np.abs(pred - truth)
    gp, gt = grad(pred), grad(truth)
    return (float(e[chg].mean()), float(e[sta].mean()) if sta.any() else 0.0,
            float(np.maximum(gt - gp, 0)[chg].mean()), float(np.maximum(gp - gt, 0)[chg].mean()))


def triplet(A, B, C, arms, net, domain, *, numerics):
    Al, Bl, Cl = (linear(A), linear(B), linear(C)) if domain == "linear" else (A, B, C)
    rows = {}
    for arm in arms:
        field, _, out = arm.partition("@")
        X, _, s = arm_fields(Al, Bl, field, net, numerics=numerics)
        rp = score(predict(A, B, X, s, sharp=out != "xf"), C, B)
        X, Y, s = arm_fields(Al, Cl, field, net, numerics=numerics)
        rb = score(interpolate(A, C, X, Y, s), B, 0.5 * (A + C))
        if rp is None or rb is None:
            return None
        rows[("predictive", arm)], rows[("bidir", arm)] = rp, rb
    return rows


def distinct_frames(path, w, h):
    f = np.fromfile(path, np.uint8).reshape(-1, h, w)
    keep = [0] + [i for i in range(1, len(f)) if np.any(f[i] != f[i - 1])]
    return f[keep].astype(np.float32) / 255


PARITY_VECTORS, PARITY_CONF_MAE = 0.999, 0.01  # share of identical vectors, mean |conf error|


def parity(rec_dir, every=7, *, numerics):
    """Does the port reproduce the GPU's checked /8 fields (vectors and confidence,
    both directions) from the lumas of GSFD samples?"""
    files = sorted(Path(rec_dir).glob("fg_*.bin"))[::every]
    if not files:
        sys.exit(f"no fg_*.bin samples under {rec_dir}")
    match, mae = [], []
    for f in files:
        lp, lc, gpu_fwd, gpu_rev = T.load_sample(f)
        forward, reverse = checked_pair(lp, lc, 8, numerics=numerics)
        for port, gpu in ((forward, gpu_fwd), (reverse, gpu_rev)):
            gpu = gpu.astype(np.float32)[..., :3]
            if not (np.isfinite(port[..., :3]).all() and np.isfinite(gpu).all()):
                print(f"parity: non-finite field in {f.name}")
                return 1
            match.append(float((np.abs(port[..., :2] - gpu[..., :2]).max(-1) < 0.02).mean()))
            mae.append(float(np.abs(port[..., 2] - gpu[..., 2]).mean()))
    m, e = float(np.mean(match)), float(np.max(mae))
    print(f"{numerics.model_id}: parity over {len(files)} GSFD pairs, both directions: vectors {m * 100:.3f}% identical, "
          f"worst confidence MAE {e:.5f} (limits {PARITY_VECTORS * 100:.1f}%, {PARITY_CONF_MAE}); "
          "capture applicability unverified (GSFD v1 has no numerical provenance)")
    return 0 if m >= PARITY_VECTORS and e <= PARITY_CONF_MAE else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="capture names (NAME-WxH.gray)")
    ap.add_argument("--size", help="WxH")
    ap.add_argument("--parity", type=Path, metavar="DIR", help="check the port against GSFD samples in DIR")
    ap.add_argument("--captures", type=Path, default=Path.home() / ".cache/gamescope-framegen-captures")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--arms", default="checked/8,checked/4")
    ap.add_argument("--net", type=Path, help="GSFR weights or GSFP profile for learned arms")
    ap.add_argument("--net-key", type=NE.parse_key, help="profile set key, when it holds several")
    ap.add_argument("--domain", choices=("coded", "linear"), default="coded",
                    help="matching domain; coded is production for 8/10-bit content")
    add_numerics_arguments(ap)
    args = ap.parse_args()
    numerics = numerics_from_args(ap, args)
    if args.parity:
        return parity(args.parity, numerics=numerics)
    if not args.names or not args.size:
        ap.error("NAME... and --size are required without --parity")
    w, h = (int(v) for v in args.size.split("x"))
    arms = args.arms.split(",")
    net = NE.load_blob(args.net, args.net_key) if args.net else None
    rng = np.random.default_rng(1)
    print(f"assumed numerics: {numerics.model_id}; capture applicability unverified")
    print(f"{args.size} {args.domain}: paired change vs {arms[0]} [95% CI] | sta (drag) | lost edges (smear) | false edges (tearing)")
    for name in args.names:
        f = distinct_frames(args.captures / f"{name}-{args.size}.gray", w, h)
        rows = [r for r in (triplet(f[i - 1], f[i], f[i + 1], arms, net, args.domain, numerics=numerics)
                            for i in range(1, len(f) - 1, args.stride)) if r]
        if not rows:
            print(f"{name}: no triplet with motion")
            continue
        for method in ("predictive", "bidir"):
            ref = np.array([r[(method, arms[0])] for r in rows])
            line = f"{name:22s} {method:10s} n={len(rows):2d}"
            for arm in arms:
                v = np.array([r[(method, arm)] for r in rows])
                d = v[:, 0] / ref[:, 0] - 1
                boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
                lo, hi = np.percentile(boot, [2.5, 97.5]) * 100
                m = v.mean(0)
                line += (f" | {arm}: {d.mean() * 100:+.1f}% [{lo:+.1f},{hi:+.1f}] sta {m[1]:.4f}"
                         f" lost {m[2]:.4f} false {m[3]:.4f}")
            print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
