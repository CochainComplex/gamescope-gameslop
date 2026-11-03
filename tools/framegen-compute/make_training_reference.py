#!/usr/bin/env python3
"""Adapt original training arithmetic to the standalone three-SSBO transport.

This checks arithmetic transport parity, not production image/sampler parity.
The training and optimizer bodies are embedded verbatim from the live shaders.
"""

import argparse
from pathlib import Path


COMMON = """#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) readonly buffer Input { float inputData[]; };
layout(std430, binding = 1) buffer Gradients { float gradientData[]; };
layout(std430, binding = 2) buffer State { float stateData[]; };
const int s_samplers[8] = int[8](0, 1, 2, 3, 4, 5, 6, 7);
"""

TRAIN = """
layout(push_constant) uniform TrainPush {
    uint u_seed;
    uint u_sliceBase;
    float u_mode;
    uint u_flags;
    float u_historyTimeScale;
    uint inputWidth;
    uint inputHeight;
    uint inputSceneCut;
};
const int u_grad = 0;
const int u_stats = 1;
ivec2 textureSize(int samplerId, int level) {
    return samplerId == 4 ? ivec2(4644, 4) : ivec2(inputWidth, inputHeight);
}
vec4 texelFetch(int samplerId, ivec2 at, int level) {
    if (samplerId == 4) return vec4(stateData[at.y * 4644 + at.x]);
    int n = int(inputWidth * inputHeight);
    at = clamp(at, ivec2(0), ivec2(inputWidth, inputHeight) - 1);
    int i = at.y * int(inputWidth) + at.x;
    bool reverse = u_sliceBase == 8u;
    if (samplerId == 0 || samplerId == 1 || samplerId == 5) {
        bool current = samplerId == 1;
        if (reverse) current = !current;
        return vec4(inputData[(current ? n : 0) + i]);
    }
    bool opposite = samplerId == 3;
    if (reverse) opposite = !opposite;
    int base = 2 * n + (opposite ? 4 * n : 0) + 4 * i;
    return vec4(inputData[base], inputData[base + 1],
                inputData[base + 2], inputData[base + 3]);
}
// Only the rejected shading-history path uses normalized texture sampling.
vec4 texture(int samplerId, vec2 uv) {
    vec2 p = uv * vec2(textureSize(samplerId, 0)) - 0.5;
    ivec2 a = ivec2(floor(p));
    vec2 f = fract(p);
    return mix(mix(texelFetch(samplerId, a, 0), texelFetch(samplerId, a + ivec2(1, 0), 0), f.x),
               mix(texelFetch(samplerId, a + ivec2(0, 1), 0), texelFetch(samplerId, a + 1, 0), f.x), f.y);
}
uvec4 imageLoad(int imageId, ivec2 at) { return uvec4(inputSceneCut); }
void imageStore(int imageId, ivec2 at, vec4 value) {
    gradientData[at.y * 4644 + at.x] = value.r;
}
"""

OPT = """
layout(push_constant) uniform OptPush {
    float u_lr;
    float u_emaAlpha;
    float u_decay;
    uint u_step;
};
const int u_state = 0;
const int u_served = 1;
ivec2 textureSize(int samplerId, int level) {
    return samplerId == 0 ? ivec2(4644, 16) : ivec2(2048, 3);
}
vec4 texelFetch(int samplerId, ivec2 at, int level) {
    return vec4(samplerId == 0 ? gradientData[at.y * 4644 + at.x]
                              : inputData[inputData.length() - 4644 + at.y * 2048 + at.x]);
}
vec4 imageLoad(int imageId, ivec2 at) {
    return vec4(stateData[at.y * 4644 + at.x]);
}
void imageStore(int imageId, ivec2 at, vec4 value) {
    int i = imageId == u_state ? at.y * 4644 + at.x : 4 * 4644 + at.y * 2048 + at.x;
    stateData[i] = value.r;
}
"""


def body(path, anchor, required):
    source = path.read_text()
    if source.count(anchor) != 1:
        raise ValueError(f"{path}: expected one arithmetic-body anchor {anchor!r}")
    result = source[source.index(anchor):]
    for token in required:
        if token not in result:
            raise ValueError(f"{path}: missing expected arithmetic anchor {token!r}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "optimizer", "evidence", "out_gradient", "out_optimizer"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    train = body(args.train, "const int C_IN  = 12;", (
        "const int N_PARAMS = 4644;", "const int LT    = 4;",
        "float wfast(int idx)", "vec3 bilinearGrad(vec2 c, ivec2 sz)",
        "void main()", "framegenEvidenceGate(", "imageStore(u_grad,"))
    optimizer = body(args.optimizer, "const int N_PARAMS = 4644;", (
        "void main()", "if (u_step == 0u)", "g /= float(nSlices);",
        "imageStore(u_state,", "imageStore(u_served,"))
    evidence = args.evidence.read_text()
    for token in ("float framegenLumaResid(float a, float b)",
                  "float framegenEvidenceGate(float refinedResid, float heldResid)"):
        if evidence.count(token) != 1:
            raise ValueError(f"{args.evidence}: unexpected evidence definitions")
    args.out_gradient.write_text(COMMON + TRAIN + evidence + "\n" + train)
    args.out_optimizer.write_text(COMMON + OPT + optimizer)


if __name__ == "__main__":
    main()
