#ifndef FRAMEGEN_RECONSTRUCT_H
#define FRAMEGEN_RECONSTRUCT_H

float colorError(vec3 a, vec3 b) {
    vec3 d = abs(a - b);
    float scale = max(1.0f, max(
        max(max(abs(a.r), abs(a.g)), abs(a.b)),
        max(max(abs(b.r), abs(b.g)), abs(b.b))));
    return max(max(d.r, d.g), d.b) / scale;
}

// The estimator deliberately works at 1/8 resolution for throughput. Plain
// bilinear upsampling is correct inside a motion layer, but it creates a
// physically impossible average vector at foreground/background boundaries.
// Test the four covering hypotheses against the full-resolution real frames
// and select one only when (a) their flows actually diverge and (b) one has a
// meaningfully better correspondence. Smooth and ambiguous areas retain the
// bilinear field, avoiding hard 8x8 cells and winner flicker.
vec4 reconstructField(vec2 uv, vec3 currentFrame, vec2 invOutSize,
                      float lowResScale, bool enabled, bool skipSmooth) {
    vec4 bilinearField = texture(s_samplers[2], uv);
    if (bilinearField.a < -0.5f || !enabled)
        return bilinearField;

    ivec2 fieldSize = textureSize(s_samplers[2], 0);
    vec2 fieldPos = uv * vec2(fieldSize) - vec2(0.5f);
    ivec2 base = ivec2(floor(fieldPos));

    // Spatial-only specialization: smooth motion has no boundary verdict.
    // Avoid full-resolution color correspondence reads in that case.
    if (skipSmooth) {
        float spread = 0.0f;
        for (int dy = 0; dy <= 1; dy++)
            for (int dx = 0; dx <= 1; dx++) {
                ivec2 q = clamp(base + ivec2(dx, dy), ivec2(0), fieldSize - 1);
                spread = max(spread, length(texelFetch(s_samplers[2], q, 0).rg - bilinearField.rg));
            }
        if (spread <= 0.25f)
            return bilinearField;
    }

    vec4 bestField = bilinearField;
    float bestScore = 1e30f;
    float secondScore = 1e30f;
    float maxFlowSpread = 0.0f;

    for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            ivec2 fieldCoord = clamp(base + ivec2(dx, dy), ivec2(0), fieldSize - 1);
            vec4 candidate = texelFetch(s_samplers[2], fieldCoord, 0);
            vec2 candidateUv = candidate.rg * lowResScale * invOutSize;
            vec3 matchedPrevious = texture(s_samplers[0], uv - candidateUv).rgb;

            vec2 samplePos = vec2(fieldCoord) + vec2(0.5f);
            vec2 spatialDelta = samplePos - uv * vec2(fieldSize);
            float score = colorError(currentFrame, matchedPrevious)
                + 0.0125f * dot(spatialDelta, spatialDelta)
                + 0.025f * (1.0f - clamp(candidate.b, 0.0f, 1.0f));

            maxFlowSpread = max(maxFlowSpread, length(candidate.rg - bilinearField.rg));
            if (score < bestScore) {
                secondScore = bestScore;
                bestScore = score;
                bestField = candidate;
            } else if (score < secondScore) {
                secondScore = score;
            }
        }
    }

    float boundaryTrust = smoothstep(0.25f, 1.0f, maxFlowSpread);
    float winnerTrust = smoothstep(0.0025f, 0.04f, secondScore - bestScore);
    return mix(bilinearField, bestField, boundaryTrust * winnerTrust);
}

#endif
