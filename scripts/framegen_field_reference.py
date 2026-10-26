"""CPU reference operators shared by offline framegen analysis and training.

The empirical numerical model has no inferred device applicability. Keep
expression order, dtypes, half-image boundaries and sampler gradients in sync
with the recorded implementation. This module owns no CLI, files or GPU state.
"""

from dataclasses import dataclass

import numpy as np


def bilinear_with_grad(img, cy, cx):
    """Sample img at continuous texel coords (cy, cx); returns value and the
    analytic gradient (d/dcy, d/dcx). Matches texture() with a normalized
    clamp-to-edge bilinear sampler."""
    h, w = img.shape
    fy = cy - 0.5
    fx = cx - 0.5
    y0 = np.floor(fy).astype(np.int64)
    x0 = np.floor(fx).astype(np.int64)
    wy = (fy - y0).astype(np.float32)
    wx = (fx - x0).astype(np.float32)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    v00 = img[y0c, x0c]
    v10 = img[y0c, x1c]
    v01 = img[y1c, x0c]
    v11 = img[y1c, x1c]
    top = v00 + wx * (v10 - v00)
    bot = v01 + wx * (v11 - v01)
    val = top + wy * (bot - top)
    dvdx = (1.0 - wy) * (v10 - v00) + wy * (v11 - v01)
    dvdy = bot - top
    return val, dvdy, dvdx


REG = 0.02                             # match regulariser per texel
FB_TOL, FB_SLOPE = 0.75, 0.05          # FB tolerance at /8, slope per texel
AGREE_LO, AGREE_HI = 0.12, 0.45        # k_flFramegenAgreeLo/Hi
SUPPRESS_LO, SUPPRESS_HI = 0.08, 0.40  # k_flFramegenSuppressLo/Hi
COARSEST = 32


def bilinear(img, y, x):
    return bilinear_with_grad(img, y, x)[0]


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def linear(coded):
    """sRGB decode, for the --domain linear what-if: production binds srgbView(),
    which despite its name is the UNORM view, so 8/10-bit content is matched on
    code values (float content would be linear already)."""
    return np.where(coded <= 0.04045, coded / 12.92, ((coded + 0.055) / 1.055) ** 2.4).astype(np.float32)


def base_luma(full, s):
    h, w = full.shape
    lh, lw = -(-h // s), -(-w // s)
    cy, cx = np.mgrid[0:lh, 0:lw].astype(np.float32)
    yc, xc = (cy + 0.5) / lh * h, (cx + 0.5) / lw * w
    oy, ox = 0.25 / lh * h, 0.25 / lw * w
    return half_store((sum(bilinear(full, yc + a * oy, xc + b * ox) for a in (-1, 1) for b in (-1, 1)) * 0.25).astype(np.float32))


def half_store(image):
    """Empirical RTZ image transport, preserving non-finites for the finite gate.

    Vulkan permits other adjacent results. This is a selected reference model,
    not a device-independent rounding guarantee. Finite overflow saturates to
    signed max-half; underflow preserves the sign of zero.
    """
    raw = np.asarray(image)
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = raw.astype(np.float16)
        rounded = np.where(np.abs(rounded.astype(np.float32)) > np.abs(raw),
                           np.nextafter(rounded, np.zeros_like(rounded)), rounded)
    return rounded.astype(np.float32)


REFERENCE_MODEL = "subtexel8-fp32-rtz"


@dataclass(frozen=True)
class Numerics:
    model_id: str

    def __post_init__(self):
        if self.model_id != REFERENCE_MODEL:
            raise ValueError("unsupported empirical reference model")

    @property
    def subtexel_bits(self):
        return 8


def half(img, *, numerics):
    h, w = img.shape
    lh, lw = max(1, -(-h // 2)), max(1, -(-w // 2))
    cy, cx = np.mgrid[0:lh, 0:lw].astype(np.float32)
    def snap(coordinate):
        relative = coordinate - 0.5
        whole = np.floor(relative)
        steps = 1 << numerics.subtexel_bits
        return whole + np.rint((relative - whole) * steps) / steps + 0.5

    filtered = bilinear(img, snap((cy + 0.5) / lh * h), snap((cx + 0.5) / lw * w)).astype(np.float32)
    return half_store(filtered)


def _fetch(img, y, x):
    h, w = img.shape
    return img[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]


def _cur9(cur, yy, xx):
    return [_fetch(cur, yy + dy, xx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def _sad(cur9, prev, yy, xx, mvy, mvx):
    s = np.zeros(yy.shape, np.float32)
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            s += np.abs(cur9[k] - _fetch(prev, yy + dy - mvy, xx + dx - mvx))
            k += 1
    return s


def _conf_subpixel(cur9, prev, yy, xx, bmy, bmx, best, zero):
    conf = np.clip(1 - best / 9 * 6, 0, 1) * np.clip((zero - best) * 4, 0, 1)
    sub = [bmx.astype(np.float32), bmy.astype(np.float32)]
    sc = _sad(cur9, prev, yy, xx, bmy, bmx)
    for axis, (m, p) in enumerate((((0, -1), (0, 1)), ((-1, 0), (1, 0)))):
        sm = _sad(cur9, prev, yy, xx, bmy + m[0], bmx + m[1])
        sp = _sad(cur9, prev, yy, xx, bmy + p[0], bmx + p[1])
        den = sm - 2 * sc + sp
        ok = den > 1e-5
        sub[axis] = np.where(ok, sub[axis] + np.clip(0.5 * (sm - sp) / np.where(ok, den, 1), -0.5, 0.5), sub[axis])
    return np.stack([sub[0], sub[1], conf], -1)


def _match(prev, cur, R=4):
    yy, xx = np.mgrid[0:cur.shape[0], 0:cur.shape[1]]
    cur9 = _cur9(cur, yy, xx)
    zero = _sad(cur9, prev, yy, xx, 0, 0)
    best, bmy, bmx = zero.copy(), np.zeros_like(yy), np.zeros_like(xx)
    for my in range(-R, R + 1):
        for mx in range(-R, R + 1):
            if mx or my:
                s = _sad(cur9, prev, yy, xx, my, mx) + REG * (abs(mx) + abs(my))
                b = s < best
                best, bmy, bmx = np.where(b, s, best), np.where(b, my, bmy), np.where(b, mx, bmx)
    return _conf_subpixel(cur9, prev, yy, xx, bmy, bmx, _sad(cur9, prev, yy, xx, bmy, bmx), zero)


def _neighbours(field, shape, coarse):
    """The 3x3 vectors around each texel: of the coarser level (coarse=True,
    doubled grid) or of the field itself."""
    h, w = field.shape[:2]
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    cy, cx = (yy >> 1, xx >> 1) if coarse else (yy, xx)
    return np.stack([field[np.clip(cy + dy, 0, h - 1), np.clip(cx + dx, 0, w - 1), :2]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 0)


def vector_median(nb):
    """Minimum L1-distance-sum vector of the 9 (first wins, as the shader)."""
    dist = np.abs(nb[:, None] - nb[None, :]).sum(-1).sum(1)
    return np.take_along_axis(nb, np.argmin(dist, 0)[None, ..., None], 0)[0]


def _refine(prev, cur, coarse, final):
    seedF = vector_median(_neighbours(coarse, cur.shape, True)) * 2.0
    yy, xx = np.mgrid[0:cur.shape[0], 0:cur.shape[1]]
    sx, sy = np.rint(seedF[..., 0]).astype(np.int64), np.rint(seedF[..., 1]).astype(np.int64)
    cur9 = _cur9(cur, yy, xx)
    bestCost = np.full(cur.shape, 1e30, np.float32)
    bestSad, bmy, bmx = bestCost.copy(), sy.copy(), sx.copy()
    for my in (-1, 0, 1):
        for mx in (-1, 0, 1):
            vy, vx = sy + my, sx + mx
            s = _sad(cur9, prev, yy, xx, vy, vx)
            cost = s + REG * (np.abs(vx - seedF[..., 0]) + np.abs(vy - seedF[..., 1]))
            b = cost < bestCost
            bestCost, bestSad = np.where(b, cost, bestCost), np.where(b, s, bestSad)
            bmy, bmx = np.where(b, vy, bmy), np.where(b, vx, bmx)
    zero = _sad(cur9, prev, yy, xx, 0, 0)
    z = zero + REG * (np.abs(seedF[..., 0]) + np.abs(seedF[..., 1])) < bestCost
    bestSad, bmy, bmx = np.where(z, zero, bestSad), np.where(z, 0, bmy), np.where(z, 0, bmx)
    if not final:
        return np.stack([bmx.astype(np.float32), bmy.astype(np.float32), np.zeros(cur.shape, np.float32)], -1)
    return _conf_subpixel(cur9, prev, yy, xx, bmy, bmx, bestSad, zero)


def field_from_base(prev_b, cur_b, s, *, numerics):
    levels = [(prev_b, cur_b)]
    while s * 2 ** (len(levels) - 1) < COARSEST:
        levels.append((half(levels[-1][0], numerics=numerics), half(levels[-1][1], numerics=numerics)))
    f = half_store(_match(*levels[-1]))
    for i in range(len(levels) - 2, -1, -1):
        f = half_store(_refine(levels[i][0], levels[i][1], f, i == 0))
    return f


def fbcheck(F, R, tol):
    h, w = F.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]

    def factor(py, px):
        py, px = np.clip(py, 0, h - 1), np.clip(px, 0, w - 1)
        f = F[py, px, :2]
        by = np.floor(py + 0.5 - f[..., 1] - 0.5).astype(np.int64)
        bx = np.floor(px + 0.5 - f[..., 0] - 0.5).astype(np.int64)
        fac, err = np.zeros((h, w), np.float32), np.full((h, w), 1e30, np.float32)
        for dy in (0, 1):
            for dx in (0, 1):
                r = R[np.clip(by + dy, 0, h - 1), np.clip(bx + dx, 0, w - 1), :2]
                e = np.abs(f + r).sum(-1)
                t = tol + FB_SLOPE * (np.abs(f).sum(-1) + np.abs(r).sum(-1))
                fac, err = np.maximum(fac, np.clip((2 * t - e) / t, 0, 1)), np.minimum(err, e)
        return fac, err

    tent, acc, wsum, errc = (1.0, 2.0, 1.0), 0.0, 0.0, None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            fac, err = factor(yy + dy, xx + dx)
            acc, wsum = acc + fac * tent[dx + 1] * tent[dy + 1], wsum + tent[dx + 1] * tent[dy + 1]
            errc = err if dx == 0 and dy == 0 else errc
    out = F.copy()
    out[..., 2] = F[..., 2] * (acc / wsum)
    return np.concatenate([out[..., :3], errc[..., None]], -1)


def checked_pair(prev_b, cur_b, s, *, numerics):
    """Compose the original operators at their actual half-float image boundaries."""
    forward = field_from_base(prev_b, cur_b, s, numerics=numerics)
    reverse = field_from_base(cur_b, prev_b, s, numerics=numerics)
    tol = FB_TOL * 8 / s
    return half_store(fbcheck(forward, reverse, tol)), half_store(fbcheck(reverse, forward, tol))


def post(F, how, tol):
    vm = vector_median(_neighbours(F, F.shape[:2], False))
    out = F.copy()
    if how == "vm":
        out[..., :2] = vm
    elif how == "nc":
        dev = np.abs(F[..., :2] - vm).sum(-1)
        out[..., 2] = F[..., 2] * (1 - smoothstep(tol, 2 * tol, dev))
    else:
        raise ValueError(how)
    return out


def sample_field(F, s, h, w):
    py, px = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    return [bilinear(F[..., k], py / s, px / s) for k in range(3)], py, px


def predict(A, B, F, s, strength=1.0, sharp=True):
    """warp.comp. Sharp (the default): where confidence is low, move B only as
    far as the confidence allows; else crossfade with the bounded extrapolation."""
    h, w = B.shape
    (vx, vy, c), py, px = sample_field(F, s, h, w)
    motion = bilinear(B, py - s * vy * strength, px - s * vx * strength)
    second = bilinear(A, py - s * vy * (strength + 1), px - s * vx * (strength + 1))
    c = c * (1 - smoothstep(AGREE_LO, AGREE_HI, np.abs(motion - second) / np.maximum(1.0, np.abs(motion))))
    if sharp:
        return bilinear(B, py - s * vy * strength * c, px - s * vx * strength * c)
    delta = B - A
    extra = B + delta * strength * (1 - smoothstep(SUPPRESS_LO, SUPPRESS_HI, np.abs(delta)))
    pad = np.pad(B, 1, mode="edge")
    cross = [pad[1:-1, 2:], pad[1:-1, :-2], pad[2:, 1:-1], pad[:-2, 1:-1], B]
    extra = np.clip(extra, np.minimum.reduce(cross), np.maximum.reduce(cross))
    return c * motion + (1 - c) * extra


def interpolate(A, C, F, R, s, t=0.5):
    h, w = A.shape
    (fx, fy, cf), py, px = sample_field(F, s, h, w)
    (rx, ry, cr), _, _ = sample_field(R, s, h, w)
    from_cur = bilinear(C, py + s * fy * (1 - t), px + s * fx * (1 - t))
    from_prev = bilinear(A, py + s * ry * t, px + s * rx * t)
    wf, wr = cf * t, cr * (1 - t)
    wsum = wf + wr
    warped = (wf * from_cur + wr * from_prev) / np.maximum(wsum, 1e-6)
    mixed = 4 * wf * wr / np.maximum(wsum * wsum, 1e-9)
    mx = np.maximum(np.abs(from_cur), np.abs(from_prev))
    validity = np.clip(wsum, 0, 1) * (1 - mixed * smoothstep(AGREE_LO, AGREE_HI, np.abs(from_cur - from_prev) / np.maximum(1.0, mx)))
    return validity * warped + (1 - validity) * ((1 - t) * A + t * C)
