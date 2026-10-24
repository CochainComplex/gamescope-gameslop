"""Shared NumPy core for the fixed offline motion-field refiner.

Training and evaluation use the same convolution, cache and gradient operations.
Feature construction, loss, optimizer and dataset/CLI policy stay in the tools.
The layout is the existing GSFR layout; initialization consumes the caller's RNG.
"""

import numpy as np

from framegen_net_formats import LAYERS, KSIZE


# --------------------------------------------------------------------------
# The net: three valid 3x3 convs (matches the shader's fused kernel over a
# padded tile; training crops sit in the interior so no padding is needed)
# --------------------------------------------------------------------------

def conv_fwd(x, W, b):
    win = np.lib.stride_tricks.sliding_window_view(x, (KSIZE, KSIZE), axis=(2, 3))
    return np.einsum("bihwkl,oikl->bohw", win, W, optimize=True) + b[None, :, None, None]


def conv_bwd(x, W, dout):
    win = np.lib.stride_tricks.sliding_window_view(x, (KSIZE, KSIZE), axis=(2, 3))
    dW = np.einsum("bihwkl,bohw->oikl", win, dout, optimize=True)
    db = dout.sum(axis=(0, 2, 3))
    dpad = np.pad(dout, ((0, 0), (0, 0), (2, 2), (2, 2)))
    dwin = np.lib.stride_tricks.sliding_window_view(dpad, (KSIZE, KSIZE), axis=(2, 3))
    dx = np.einsum("bohwkl,oikl->bihw", dwin, W[:, :, ::-1, ::-1], optimize=True)
    return dx, dW, db


def net_fwd(params, x):
    a1 = np.maximum(conv_fwd(x, params["W1"], params["b1"]), 0.0)
    a2 = np.maximum(conv_fwd(a1, params["W2"], params["b2"]), 0.0)
    o = conv_fwd(a2, params["W3"], params["b3"])
    return o, (x, a1, a2)


def net_bwd(params, cache, do):
    x, a1, a2 = cache
    da2, dW3, db3 = conv_bwd(a2, params["W3"], do)
    da2 *= a2 > 0
    da1, dW2, db2 = conv_bwd(a1, params["W2"], da2)
    da1 *= a1 > 0
    _, dW1, db1 = conv_bwd(x, params["W1"], da1)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def init_params(rng):
    p = {}
    for i, (ci, co) in enumerate(LAYERS, 1):
        if i == len(LAYERS):
            # Zero head: the untrained net is exactly Stage B.
            p[f"W{i}"] = np.zeros((co, ci, KSIZE, KSIZE), np.float32)
        else:
            std = np.sqrt(2.0 / (ci * KSIZE * KSIZE))
            p[f"W{i}"] = rng.normal(0.0, std, (co, ci, KSIZE, KSIZE)).astype(np.float32)
        p[f"b{i}"] = np.zeros(co, np.float32)
    return p


