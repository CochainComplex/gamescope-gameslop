"""Cropped network inputs and evidence policy shared by offline tools.

Preserve the trainer's cast/expression order and canonical bilinear sampler.
The evaluator's whole-frame builder has separate dtype timing and stays there;
this module does not own training loss, optimizer, scoring or GPU inference.
"""

import numpy as np

from framegen_field_reference import bilinear_with_grad as bilinear


# --------------------------------------------------------------------------
# Features (channel-for-channel the shader's phase A)
# --------------------------------------------------------------------------

def evidence_gate(warped, luma_dst, luma_src):
    """The confidence-raise gate of cs_framegen_motion_net.comp: the refined
    vector must predict the pair (normalized residual fades out over
    0.03..0.10) and explain it better than holding (zero motion): the raise
    scales with the share of the zero-motion residual it removes, floored at
    the 0.03 'explained' level, so on static content no vector earns one."""
    def resid(a, b):
        return np.abs(a - b) / np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    rr, r0 = resid(warped, luma_dst), resid(luma_src, luma_dst)
    t = np.clip((rr - 0.03) / 0.07, 0.0, 1.0)
    return (1.0 - t * t * (3.0 - 2.0 * t)) * np.clip((r0 - rr) / np.maximum(r0, 0.03), 0.0, 1.0)


def features(X, Y, luma_src, luma_dst, y0, x0, size, mode=0.0):
    """12-channel feature crop of `size` starting at (y0, x0)."""
    ys, xs = slice(y0, y0 + size), slice(x0, x0 + size)
    Xc = X[ys, xs].astype(np.float32)
    Yc = Y[ys, xs].astype(np.float32)
    ld = luma_dst[ys, xs].astype(np.float32)
    py, px = np.mgrid[y0:y0 + size, x0:x0 + size].astype(np.float32)
    lw, _, _ = bilinear(luma_src, py + 0.5 - Xc[..., 1], px + 0.5 - Xc[..., 0])
    resid = np.abs(ld - lw) / np.maximum(1.0, np.maximum(np.abs(ld), np.abs(lw)))
    f = np.empty((12, size, size), np.float32)
    f[0] = Xc[..., 0] * 0.125
    f[1] = Xc[..., 1] * 0.125
    f[2] = Xc[..., 2]
    f[3] = np.minimum(Xc[..., 3] * 0.5, 1.0)
    f[4] = Yc[..., 0] * 0.125
    f[5] = Yc[..., 1] * 0.125
    f[6] = Yc[..., 2]
    f[7] = np.minimum(Yc[..., 3] * 0.5, 1.0)
    f[8] = ld
    f[9] = lw
    f[10] = resid
    f[11] = mode
    return f


