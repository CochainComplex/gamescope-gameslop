"""Existing GSFD v1 sample and dataset readers for offline framegen tools.

Admission, dtypes, views and dataset error/stream behavior match the trainer's
reader. This leaf does not import training, numerical-reference or model code.
It deliberately adds no exact-length, sequence, finite-value or shape policy.
"""

import struct
import sys
from pathlib import Path

import numpy as np


MAGIC_SAMPLE = 0x44465347  # 'GSFD'


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def load_sample(path):
    raw = path.read_bytes()
    hdr = struct.unpack_from("<8I", raw, 0)
    if hdr[0] != MAGIC_SAMPLE or hdr[1] != 1:
        raise ValueError(f"{path}: not a GSFD v1 sample")
    w, h, luma_bpp, field_bpp, flags = hdr[2], hdr[3], hdr[4], hdr[5], hdr[6]
    if field_bpp != 8 or not (flags & 1):
        raise ValueError(f"{path}: unsupported layout")
    off = 8 * 4 + 8  # header + seqno

    def luma_plane():
        nonlocal off
        n = w * h * luma_bpp
        plane = np.frombuffer(raw, dtype=np.float16, count=n // 2, offset=off)
        off += n
        if luma_bpp == 8:  # rgba16f luma texture: luma sits in .r
            plane = plane.reshape(h, w, 4)[..., 0]
        return np.ascontiguousarray(plane.reshape(h, w), dtype=np.float32)

    def field_plane():
        nonlocal off
        n = w * h * 8
        plane = np.frombuffer(raw, dtype=np.float16, count=n // 2, offset=off)
        off += n
        return np.ascontiguousarray(plane.reshape(h, w, 4))

    luma_prev, luma_cur = luma_plane(), luma_plane()
    field_fwd, field_rev = field_plane(), field_plane()
    return luma_prev, luma_cur, field_fwd, field_rev


def load_dataset(data_dir):
    files = sorted(Path(data_dir).glob("fg_*.bin"))
    if not files:
        sys.exit(f"no fg_*.bin samples under {data_dir}")
    samples = []
    for f in files:
        try:
            samples.append(load_sample(f))
        except ValueError as e:
            print(f"skipping {e}", file=sys.stderr)
    print(f"loaded {len(samples)} samples "
          f"({samples[0][0].shape[1]}x{samples[0][0].shape[0]} field res)")
    return samples


