#pragma once

#include "sceneStructs.h"

#include <thrust/random.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

// Shared thin-lens geometry for the production camera kernel and CPU/GPU tests.
namespace cameraSampling {

struct LensFrame {
    glm::vec3 forward = glm::vec3(0.0f);
    glm::vec3 right = glm::vec3(0.0f);
    glm::vec3 up = glm::vec3(0.0f);
};

__host__ __device__ inline bool finite(float value)
{
    return value >= -FLT_MAX && value <= FLT_MAX;
}

__host__ __device__ inline bool finite(const glm::vec3& value)
{
    return finite(value.x) && finite(value.y) && finite(value.z);
}

// Scaling first avoids overflow/underflow when normalizing finite vectors.
__host__ __device__ inline bool unitVector(const glm::vec3& value, glm::vec3& unit)
{
    if (!finite(value)) return false;
    const float scale = fmaxf(fabsf(value.x), fmaxf(fabsf(value.y), fabsf(value.z)));
    if (!(scale > 0.0f)) return false;
    const glm::vec3 scaled = value / scale;
    unit = scaled / sqrtf(glm::dot(scaled, scaled));
    return finite(unit);
}

// Does not mutate Camera::up/right or the existing pinhole projection.
// Reject an undefined roll axis (parallel or nearly parallel view and up).
__host__ __device__ inline bool makeLensFrame(
    const glm::vec3& view, const glm::vec3& upHint, LensFrame& result)
{
    LensFrame frame;
    glm::vec3 up;
    if (!unitVector(view, frame.forward) || !unitVector(upHint, up)) return false;
    const glm::vec3 right = glm::cross(frame.forward, up);
    if (!(glm::dot(right, right) > 1e-12f) || !unitVector(right, frame.right)) return false;
    if (!unitVector(glm::cross(frame.right, frame.forward), frame.up)) return false;
    result = frame;
    return true;
}

// Input coordinates are in [0,1]. This maps a uniform square sample to a disk
// uniformly by area; using u.x instead of sqrt(u.x) would bias the center.
__host__ __device__ inline glm::vec2 sampleUnitDisk(const glm::vec2& u)
{
    const float radius = sqrtf(u.x);
    const float angle = 6.2831853071795864769f * u.y;
    return glm::vec2(radius * cosf(angle), radius * sinf(angle));
}

__host__ __device__ inline uint32_t mixLensSeed(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

// A separate stream from the renderer's AA/depth seeds. Unsigned arithmetic
// avoids signed shifts/overflow and supports the entire 32-bit input range.
__host__ __device__ inline thrust::default_random_engine makeLensRandomEngine(
    uint32_t iteration, uint32_t pixelIndex)
{
    const uint32_t seed = mixLensSeed(0x4c454e53u
        ^ mixLensSeed(iteration ^ 0x63d83595u)
        ^ mixLensSeed(pixelIndex ^ 0xa511e9b3u));
    return thrust::default_random_engine(seed);
}

// The pinhole ray must start at camera.position. Direction need not be unit.
// Disabled/zero-radius calls copy it exactly, without lens math or sampling.
// Invalid active inputs return false and leave that same pinhole ray in result.
// Callers must handle failure; it is not a successful depth-of-field sample.
__host__ __device__ inline bool makeThinLensRay(const Camera& camera,
    const Ray& pinhole, bool enabled, const glm::vec2& lensSample, Ray& result)
{
    result = pinhole;
    if (!enabled || camera.lensRadius == 0.0f) return true;
    if (!finite(camera.lensRadius) || !(camera.lensRadius > 0.0f)
        || !finite(camera.focalDistance) || !(camera.focalDistance > 0.0f)
        || !finite(camera.position) || !finite(pinhole.direction)
        || pinhole.origin.x != camera.position.x
        || pinhole.origin.y != camera.position.y
        || pinhole.origin.z != camera.position.z
        || !(lensSample.x >= 0.0f && lensSample.x <= 1.0f)
        || !(lensSample.y >= 0.0f && lensSample.y <= 1.0f)) return false;

    LensFrame frame;
    if (!makeLensFrame(camera.view, camera.up, frame)) return false;
    glm::vec3 direction;
    if (!unitVector(pinhole.direction, direction)) return false;
    const float cosine = glm::dot(direction, frame.forward);
    if (!(cosine > 0.0f)) return false;

    const glm::vec3 focusOffset = direction * (camera.focalDistance / cosine);
    const glm::vec2 disk = sampleUnitDisk(lensSample);
    const glm::vec3 lensOffset = camera.lensRadius
        * (disk.x * frame.right + disk.y * frame.up);
    Ray ray;
    ray.origin = camera.position + lensOffset;
    // Work with offsets to avoid subtracting two large world-space positions.
    if (!finite(ray.origin) || !unitVector(focusOffset - lensOffset, ray.direction)) return false;
    result = ray;
    return true;
}

} // namespace cameraSampling
