#pragma once
#include "sceneStructs.h"
#include <cmath>

__host__ __device__ inline float decodeSRGB(float x) {
    return x <= 0.04045f ? x / 12.92f : powf((x + 0.055f) / 1.055f, 2.4f);
}
__host__ __device__ inline float encodeSRGB(float x) {
    return x <= 0.0031308f ? 12.92f * x : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
}
// Display only. Accumulation, texture data and HDR exports stay linear.
__host__ __device__ inline glm::vec3 displayColor(glm::vec3 c, bool enabled, float exposure) {
    if (!enabled) return c;
    c = glm::max(c * exp2f(exposure), glm::vec3(0));
    for (int i = 0; i < 3; ++i) c[i] = encodeSRGB(c[i] / (1.0f + c[i]));
    return c;
}
__host__ __device__ inline int textureWrap(int i, int size) {
    return (i % size + size) % size;
}
// Repeat addressing, bilinear filtering of LINEAR pixels, OBJ bottom-up V.
__host__ __device__ inline glm::vec3 sampleBaseColor(
    const TextureInfo& texture, const glm::vec3* pixels, glm::vec2 uv) {
    const float x = (uv.x - floorf(uv.x)) * texture.width - 0.5f;
    const float y = (1.0f - (uv.y - floorf(uv.y))) * texture.height - 0.5f;
    const int ix = int(floorf(x)), iy = int(floorf(y));
    const float fx = x - floorf(x), fy = y - floorf(y);
    const int x0 = textureWrap(ix, texture.width), x1 = textureWrap(ix + 1, texture.width);
    const int y0 = textureWrap(iy, texture.height), y1 = textureWrap(iy + 1, texture.height);
    const glm::vec3 a = pixels[texture.offset + y0 * texture.width + x0] * (1 - fx)
        + pixels[texture.offset + y0 * texture.width + x1] * fx;
    const glm::vec3 b = pixels[texture.offset + y1 * texture.width + x0] * (1 - fx)
        + pixels[texture.offset + y1 * texture.width + x1] * fx;
    return a * (1 - fy) + b * fy;
}

// Reconstruct attributes only after choosing the closest triangle. All three
// traversal modes keep the same intersection routine and deterministic tie rule.
__host__ __device__ inline glm::vec3 surfaceBarycentrics(const Triangle& t, glm::vec3 p) {
    glm::vec3 e0 = t.v1 - t.v0, e1 = t.v2 - t.v0, q = p - t.v0;
    float scale = fmaxf(glm::length(e0), glm::length(e1));
    if (!(scale > 0)) return glm::vec3(1, 0, 0);
    e0 /= scale; e1 /= scale; q /= scale;
    // Cross products avoid cancellation for thin triangles.
    const glm::vec3 n = glm::cross(e0, e1);
    const float den = glm::dot(n, n);
    if (!(den > 0)) return glm::vec3(1, 0, 0);
    const float b = glm::dot(glm::cross(q, e1), n) / den;
    const float c = glm::dot(glm::cross(e0, q), n) / den;
    glm::vec3 weights = glm::clamp(glm::vec3(1 - b - c, b, c), glm::vec3(0), glm::vec3(1));
    return weights / (weights.x + weights.y + weights.z);
}
