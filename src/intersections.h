#pragma once

#include "sceneStructs.h"
#include "bvh.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>


/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);


// Keep one compiled triangle routine for all traversal modes: separate inlining
// can change floating-point edge acceptance and closest-hit tie resolution.
// Two-sided, flat triangle intersection. The direction is not normalized here:
// return the parameter of the supplied ray, or -1 for a miss.
__host__ __device__ __noinline__ float triangleIntersectionTest(
    const Triangle& triangle,
    const Ray& ray);

// Conservative ray/local-AABB overlap on [0, infinity). Boundary contact and
// rays starting inside are retained. Numerical uncertainty falls back to the
// triangle scan rather than discarding a possible hit.
__host__ __device__ bool aabbIntersectionTest(
    const Ray& ray,
    const glm::vec3& boundsMin,
    const glm::vec3& boundsMax);

// Scan the mesh's range in the shared triangle array. Return world-space
// distance and an inverse-transpose transformed unit face normal, or -1.
// The CPU loader supplies valid ranges and conservative local bounds.
__host__ __device__ float meshIntersectionTest(
    const Geom& mesh,
    const Ray& ray,
    const Triangle* triangles,
    bool enableMeshCulling,
    glm::vec3& normal, int* triangleId = nullptr);


// Conservative overlap with [0, tMax], including touching endpoints. Outputs
// are always initialized; uncertain arithmetic returns the full input interval.
__host__ __device__ bool aabbIntervalIntersectionTest(
    const Ray& ray, const glm::vec3& boundsMin, const glm::vec3& boundsMax,
    float tMax, float& tEnter, float& tExit);

// Flat device arrays use the same absolute indices as the CPU builder.
struct BVHDeviceView
{
    const BVHNode* nodes = nullptr;
    const int* triangleIndices = nullptr;
    int nodeCount = 0;
    int indexCount = 0;
    int triangleCount = 0;
};

// Root depth is zero. DFS can hold one pending sibling per level plus the leaf.
constexpr int BVH_STACK_CAPACITY = BVH_MAX_DEPTH + 1;

// Optional diagnostics; the normal traversal uses a specialization without
// counter updates. Callers zero these counters before collecting one ray.
struct BVHTraversalStats
{
    unsigned long long nodeVisits = 0;
    unsigned long long aabbTests = 0;
    unsigned long long triangleTests = 0;
    unsigned int fallbackCount = 0;
    int maxStack = 0;
};

__host__ __device__ float meshBVHIntersectionTest(
    const Geom& mesh, const Ray& ray, const Triangle* triangles,
    const BVHDeviceView& bvh, glm::vec3& normal,
    BVHTraversalStats* stats = nullptr,
    int stackCapacity = BVH_STACK_CAPACITY, int* triangleId = nullptr);
