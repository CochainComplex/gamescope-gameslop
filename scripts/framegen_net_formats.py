"""GSFR/GSFP serialization shared by offline framegen tools.

This preserves the existing Python admission rules, errors and output. They
are distinct from the runtime codec: shape-compatible bare versions warn
without shading-head migration, while profile sets require the current version.
Dataset parsing, training, metrics and CLI argument policy stay with the tools.
"""

import struct
import sys
from pathlib import Path

import numpy as np


MAGIC_BLOB = 0x52465347    # 'GSFR'
BLOB_VERSION = 3           # + zero-neutral causal shading-focus output
LAYERS = ((12, 16), (16, 16), (16, 4))
KSIZE = 3


MAGIC_PROFILE = 0x50465347   # 'GSFP': a gamescope per-game profile
PROFILE_VERSION = 1
PROFILE_MAX_SETS = 8


def _parse_gsfr(raw, off, path, strict):
    """One GSFR blob at raw[off:]: (params, end offset). strict = profile set."""
    magic, version, nlayers = struct.unpack_from("<3I", raw, off)
    if magic != MAGIC_BLOB:
        sys.exit(f"{path}: not a GSFR weights blob")
    if nlayers != len(LAYERS):
        sys.exit(f"{path}: layer count {nlayers} != expected {len(LAYERS)}")
    if strict and version != BLOB_VERSION:
        sys.exit(f"{path}: profile set has GSFR v{version}, want v{BLOB_VERSION}")
    off += 12
    shapes = []
    for i, (exp_ci, exp_co) in enumerate(LAYERS, 1):
        ci, co, ksize = struct.unpack_from("<3I", raw, off)
        off += 12
        if (ci, co, ksize) != (exp_ci, exp_co, KSIZE):
            sys.exit(f"{path}: layer {i} shape mismatch")
        shapes.append((ci, co, ksize))
    params = {}
    for i, (ci, co, ksize) in enumerate(shapes, 1):
        n = co * ksize * ksize * ci
        if off + (n + co) * 4 > len(raw):
            sys.exit(f"{path}: truncated")
        w = np.frombuffer(raw, "<f4", count=n, offset=off).reshape(co, ksize, ksize, ci)
        off += n * 4
        params[f"W{i}"] = np.ascontiguousarray(w.transpose(0, 3, 1, 2))  # -> (co,ci,ky,kx)
        params[f"b{i}"] = np.array(
            np.frombuffer(raw, "<f4", count=co, offset=off), np.float32)
        off += co * 4
    if not all(np.isfinite(v).all() for v in params.values()):
        sys.exit(f"{path}: non-finite weights")
    if not strict and version != BLOB_VERSION:
        print(f"note: blob version {version} != trainer {BLOB_VERSION}; "
              f"shapes match, evaluating anyway", file=sys.stderr)
    return params, off


def load_blob(path, key=None):
    """A bare GSFR blob (the trainer's export), or one set of a GSFP profile.

    A profile keeps one set per input-encoding key; with several sets, `key`
    must name one (see gamescope::framegen::net_input_key)."""
    raw = Path(path).read_bytes()
    if len(raw) < 12:
        sys.exit(f"{path}: too short")
    magic, version, count = struct.unpack_from("<3I", raw, 0)
    if magic != MAGIC_PROFILE:
        params, end = _parse_gsfr(raw, 0, path, strict=False)
        if end != len(raw):
            sys.exit(f"{path}: {len(raw) - end} trailing bytes")
        if key is not None:
            sys.exit(f"{path}: a bare GSFR blob has no keys")
        return params
    if version != PROFILE_VERSION or not 1 <= count <= PROFILE_MAX_SETS:
        sys.exit(f"{path}: unsupported GSFP v{version} with {count} sets")
    sets, off = {}, 12
    for _ in range(count):
        (k,) = struct.unpack_from("<I", raw, off)
        if k in sets:
            sys.exit(f"{path}: duplicate key 0x{k:03x}")
        sets[k], off = _parse_gsfr(raw, off + 4, path, strict=True)
    if off != len(raw):
        sys.exit(f"{path}: {len(raw) - off} trailing bytes")
    if key is None and len(sets) == 1:
        return next(iter(sets.values()))
    if key not in sets:
        have = ", ".join(f"0x{k:03x}" for k in sets)
        sys.exit(f"{path}: pick a set with --net-key ({have})")
    return sets[key]


def export(params, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<3I", MAGIC_BLOB, BLOB_VERSION, len(LAYERS)))
        for ci, co in LAYERS:
            f.write(struct.pack("<3I", ci, co, KSIZE))
        for i in range(1, len(LAYERS) + 1):
            # Shader layout: [c_out][ky][kx][c_in] — input channels contiguous
            # so the kernel reads whole vec4 groups.
            params[f"W{i}"].transpose(0, 2, 3, 1).astype("<f4").tofile(f)
            params[f"b{i}"].astype("<f4").tofile(f)
    total = sum(p.size for p in params.values())
    print(f"exported {total} floats -> {path}")
