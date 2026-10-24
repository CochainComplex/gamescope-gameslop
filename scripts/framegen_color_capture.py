"""GSCF capture admission and code-value RGB decoding for offline tools.

Preserve the existing v1/v2/v3 layout, validation order and metadata. Pixel
values retain their stored range and nonfinite values; this reader applies no
HDR transfer function, alpha composite or quality/scoring policy.
"""

import math
import struct

import numpy as np


MAGIC = 0x46435347  # 'GSCF'
VERSION = 3
SUPPORTED_VERSIONS = (1, 2, VERSION)
HEADER = struct.Struct("<12I8Q4f")
CANDIDATE_KINDS = {
    0: "occlusion",
    1: "endpoint-trace",
}


def fourcc(text):
    return sum(ord(ch) << (8 * i) for i, ch in enumerate(text))


ARGB8888 = fourcc("AR24")
XRGB8888 = fourcc("XR24")
ABGR8888 = fourcc("AB24")
XBGR8888 = fourcc("XB24")
ARGB2101010 = fourcc("AR30")
XRGB2101010 = fourcc("XR30")
ABGR2101010 = fourcc("AB30")
XBGR2101010 = fourcc("XB30")
ABGR16161616 = fourcc("AB48")
XBGR16161616 = fourcc("XB48")
ABGR16161616F = fourcc("AB4H")
XBGR16161616F = fourcc("XB4H")
ABGR32323232F = fourcc("AB8F")

FORMAT_BPP = {
    ARGB8888: 4,
    XRGB8888: 4,
    ABGR8888: 4,
    XBGR8888: 4,
    ARGB2101010: 4,
    XRGB2101010: 4,
    ABGR2101010: 4,
    XBGR2101010: 4,
    ABGR16161616: 8,
    XBGR16161616: 8,
    ABGR16161616F: 8,
    XBGR16161616F: 8,
    ABGR32323232F: 16,
}


def fourcc_name(value):
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


def decode_plane(raw, width, height, drm_format, bpp):
    expected = width * height * bpp
    if len(raw) != expected:
        raise ValueError(f"plane is {len(raw)} bytes, expected {expected}")

    if drm_format in (ABGR8888, XBGR8888):
        rgba = np.frombuffer(raw, np.uint8).reshape(height, width, 4)
        return np.ascontiguousarray(rgba[..., :3], dtype=np.float32) / 255.0
    if drm_format in (ARGB8888, XRGB8888):
        bgra = np.frombuffer(raw, np.uint8).reshape(height, width, 4)
        return np.ascontiguousarray(bgra[..., 2::-1], dtype=np.float32) / 255.0

    if drm_format in (ABGR2101010, XBGR2101010, ARGB2101010, XRGB2101010):
        word = np.frombuffer(raw, "<u4").reshape(height, width)
        lo = (word & 0x3FF).astype(np.float32) / 1023.0
        mid = ((word >> 10) & 0x3FF).astype(np.float32) / 1023.0
        hi = ((word >> 20) & 0x3FF).astype(np.float32) / 1023.0
        if drm_format in (ABGR2101010, XBGR2101010):
            return np.stack((lo, mid, hi), axis=-1)
        return np.stack((hi, mid, lo), axis=-1)

    if drm_format in (ABGR16161616, XBGR16161616):
        rgba = np.frombuffer(raw, "<u2").reshape(height, width, 4)
        return np.ascontiguousarray(rgba[..., :3], dtype=np.float32) / 65535.0
    if drm_format in (ABGR16161616F, XBGR16161616F):
        rgba = np.frombuffer(raw, "<f2").reshape(height, width, 4)
        return np.ascontiguousarray(rgba[..., :3], dtype=np.float32)
    if drm_format == ABGR32323232F:
        rgba = np.frombuffer(raw, "<f4").reshape(height, width, 4)
        return np.ascontiguousarray(rgba[..., :3], dtype=np.float32)

    raise ValueError(f"unsupported DRM format {fourcc_name(drm_format)!r} (0x{drm_format:08x})")


def load_sample(path):
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("truncated GSCF header")
    values = HEADER.unpack_from(raw)
    header = values[:12]
    meta = values[12:20]
    phase = values[20]
    magic, version, width, height, drm_format, bpp, eotf, flags, planes = header[:9]
    if magic != MAGIC or version not in SUPPORTED_VERSIONS:
        raise ValueError("not a supported GSCF v1/v2/v3 capture")
    expected_planes = 2 if version == 1 else 4
    expected_flags = 1 if version == 1 else 3
    if (flags & expected_flags) != expected_flags or planes != expected_planes:
        raise ValueError("capture has no supported exact held-out reference layout")
    if width == 0 or height == 0 or bpp != FORMAT_BPP.get(drm_format):
        raise ValueError("capture has invalid dimensions, format, or bytes-per-pixel")
    if not math.isfinite(phase) or not 0.0 < phase < 1.0:
        raise ValueError("capture has an invalid held-out phase")
    if not (meta[1] < meta[2] < meta[3] and meta[4] < meta[5] < meta[6]):
        raise ValueError("capture has non-monotonic frame IDs or timestamps")
    timestamp_phase = (meta[5] - meta[4]) / (meta[6] - meta[4])
    if not math.isclose(phase, timestamp_phase, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("capture phase disagrees with its timestamps")
    if version >= 2:
        strengths = values[21:24]
        if (not all(math.isfinite(value) and 0.0 <= value <= 1.0
                    for value in strengths)
                or not strengths[0] < strengths[1] < strengths[2]):
            raise ValueError("capture has invalid paired candidate strengths")
    if version == 1:
        candidate_kind = None
    elif version == 2:
        candidate_kind = "occlusion"
    else:
        candidate_kind = CANDIDATE_KINDS.get(header[9])
        if candidate_kind is None:
            raise ValueError(f"capture has unknown paired candidate kind {header[9]}")
    plane_bytes = width * height * bpp
    expected = HEADER.size + planes * plane_bytes
    if len(raw) != expected:
        raise ValueError(f"file is {len(raw)} bytes, expected {expected}")
    decoded = []
    for i in range(planes):
        begin = HEADER.size + i * plane_bytes
        decoded.append(decode_plane(raw[begin:begin + plane_bytes],
                                    width, height, drm_format, bpp))
    if version == 1:
        candidates = [(None, decoded[0])]
    else:
        candidates = list(zip(values[21:24], decoded[:3]))
    reference = decoded[-1]
    return {
        "path": path,
        "width": width,
        "height": height,
        "format": drm_format,
        "eotf": eotf,
        "phase": phase,
        "sequence": meta[0],
        "anchor_id": meta[1],
        "reference_id": meta[2],
        "endpoint_id": meta[3],
        "anchor_ns": meta[4],
        "reference_ns": meta[5],
        "endpoint_ns": meta[6],
        "candidate_kind": candidate_kind,
        "candidates": candidates,
        "reference": reference,
    }


