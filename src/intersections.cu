#include "intersections.h"

#include <cfloat>
#include <cmath>

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


namespace {
__host__ __device__ float maxAbsComponent(const glm::vec3& v)
{
    return fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z)));
}
}

__host__ __device__ __noinline__ float triangleIntersectionTest(
    const Triangle& triangle,
    const Ray& ray)
{
    glm::vec3 e1 = triangle.v1 - triangle.v0;
    glm::vec3 e2 = triangle.v2 - triangle.v0;

    // Keep determinant tests independent of uniform model/ray scale and avoid
    // squaring tiny/large edge lengths. Undo these scales when returning t.
    const float edgeScale = fmaxf(maxAbsComponent(e1), maxAbsComponent(e2));
    const float directionScale = maxAbsComponent(ray.direction);
    if (!(edgeScale > 0.0f) || !(directionScale > 0.0f)) return -1.0f;
    e1 /= edgeScale;
    e2 /= edgeScale;
    const glm::vec3 direction = ray.direction / directionScale;

    // Moller-Trumbore; absolute determinant deliberately accepts both sides.
    const glm::vec3 p = glm::cross(direction, e2);
    const float determinant = glm::dot(e1, p);
    if (!(fabsf(determinant) > 1e-7f)) return -1.0f;
    const float invDeterminant = 1.0f / determinant;
    const glm::vec3 offset = (ray.origin - triangle.v0) / edgeScale;
    const float u = glm::dot(offset, p) * invDeterminant;
    if (!(u >= 0.0f && u <= 1.0f)) return -1.0f;
    const glm::vec3 q = glm::cross(offset, e1);
    const float v = glm::dot(direction, q) * invDeterminant;
    if (!(v >= 0.0f && u + v <= 1.0f)) return -1.0f;

    const float t = (glm::dot(e2, q) * invDeterminant * edgeScale) / directionScale;
    // meshIntersectionTest preserves world distance; scatterRay already offsets
    // the next ray by 1e-4. Reject zero/near-zero and non-finite intersections.
    return (t > 1e-5f && t < FLT_MAX) ? t : -1.0f;
}

__host__ __device__ bool aabbIntervalIntersectionTest(
    const Ray& ray,
    const glm::vec3& boundsMin,
    const glm::vec3& boundsMax,
    float tMax, float& tEnter, float& tExit)
{
    const float limit = isfinite(tMax) && tMax >= 0.0f ? tMax : FLT_MAX;
    tEnter = 0.0f;
    tExit = limit;
    if (tMax < 0.0f) { tExit = 0.0f; return false; }
    if (isnan(tMax)) return true;
    // Malformed/non-finite data must not introduce a false negative in this
    // optional optimization. The normal triangle path remains the fallback.
    for (int axis = 0; axis < 3; ++axis) {
        if (!isfinite(ray.origin[axis]) || !isfinite(ray.direction[axis])
            || !isfinite(boundsMin[axis]) || !isfinite(boundsMax[axis])
            || boundsMin[axis] > boundsMax[axis]) return true;
    }

    for (int axis = 0; axis < 3; ++axis) {
        const float origin = ray.origin[axis];
        const float direction = ray.direction[axis];
        // Only exact zero is parallel. Nonuniform scaling may legitimately
        // produce very small nonzero components in the unnormalized local ray.
        if (direction == 0.0f) {
            if (origin < boundsMin[axis] || origin > boundsMax[axis]) return false;
            continue;
        }
        // Direct division avoids 0 * infinity on a boundary with tiny direction.
        const float a = (boundsMin[axis] - origin) / direction;
        const float b = (boundsMax[axis] - origin) / direction;
        if (!isfinite(a) || !isfinite(b)) {
            // Earlier axes may already have narrowed the interval. Reset it:
            // using a partial/undefined entry distance could incorrectly prune.
            tEnter = 0.0f;
            tExit = limit;
            return true;
        }
        float nearT = fminf(a, b);
        float farT = fmaxf(a, b);
        // Widen both endpoints outwards to cover subtraction/division rounding.
        // CPU bounds already include a coordinate-scaled margin as well.
        nearT = nextafterf(nearT - 4.0f * FLT_EPSILON * fabsf(nearT), -INFINITY);
        farT = nextafterf(farT + 4.0f * FLT_EPSILON * fabsf(farT), INFINITY);
        tEnter = fmaxf(tEnter, nearT);
        tExit = fminf(tExit, farT);
        if (tEnter > tExit) return false;
    }
    return true;
}

__host__ __device__ bool aabbIntersectionTest(
    const Ray& ray, const glm::vec3& boundsMin, const glm::vec3& boundsMax)
{
    float tEnter, tExit;
    return aabbIntervalIntersectionTest(ray, boundsMin, boundsMax, FLT_MAX, tEnter, tExit);
}

__host__ __device__ float meshIntersectionTest(
    const Geom& mesh,
    const Ray& ray,
    const Triangle* triangles,
    bool enableMeshCulling,
    glm::vec3& normal, int* triangleId)
{
    if (triangleId) *triangleId = -1;
    if (triangles == nullptr || mesh.triangleCount <= 0) return -1.0f;

    Ray localRay;
    localRay.origin = multiplyMV(mesh.inverseTransform, glm::vec4(ray.origin, 1.0f));
    // Normalize in WORLD space only. With an unnormalized local direction, the
    // same t is a world distance, even with nonuniform or negative scaling.
    localRay.direction = multiplyMV(mesh.inverseTransform,
        glm::vec4(glm::normalize(ray.direction), 0.0f));

    if (enableMeshCulling
        && !aabbIntersectionTest(localRay, mesh.boundsMin, mesh.boundsMax)) {
        return -1.0f;
    }

    float closestT = FLT_MAX;
    int hitTriangle = -1;
    for (int i = 0; i < mesh.triangleCount; ++i) {
        const int triangleIndex = mesh.triangleStart + i;
        const float t = triangleIntersectionTest(triangles[triangleIndex], localRay);
        if (t > 0.0f && t < closestT) {
            closestT = t;
            hitTriangle = triangleIndex;
        }
    }
    if (hitTriangle < 0) return -1.0f;

    // Keep the geometric orientation; scatterRay face-forwards it as needed.
    normal = glm::normalize(multiplyMV(mesh.invTranspose,
        glm::vec4(triangles[hitTriangle].normal, 0.0f)));
    if (triangleId) *triangleId = hitTriangle;
    return closestT;
}


namespace {
__host__ __device__ bool validBVHRange(int start, int count, int limit)
{
    return start >= 0 && count >= 0 && start <= limit && count <= limit - start;
}

__host__ __device__ bool intersectBVHNode(
    const BVHNode& node, const Ray& ray, float closestT, float& entry)
{
    float exit;
    return aabbIntervalIntersectionTest(ray,
        glm::vec3(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]),
        glm::vec3(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]),
        closestT, entry, exit);
}

template<bool CollectStats>
__host__ __device__ float fallbackMeshScan(
    const Geom& mesh, const Ray& ray, const Triangle* triangles,
    glm::vec3& normal, BVHTraversalStats* stats, int* triangleId)
{
    if constexpr (CollectStats) {
        ++stats->fallbackCount;
        stats->triangleTests += mesh.triangleCount;
    }
    // Recompute the complete closest hit, including triangles already visited.
    return meshIntersectionTest(mesh, ray, triangles, false, normal, triangleId);
}

template<bool CollectStats>
__host__ __device__ float traverseMeshBVH(
    const Geom& mesh, const Ray& ray, const Triangle* triangles,
    const BVHDeviceView& bvh, glm::vec3& normal,
    BVHTraversalStats* stats, int stackCapacity, int* triangleId)
{
    if (triangleId) *triangleId = -1;
    if (!triangles || mesh.triangleCount <= 0) return -1.0f;
    // Scene validation guarantees the triangle range. Also guard direct callers.
    if (!validBVHRange(mesh.triangleStart, mesh.triangleCount, bvh.triangleCount)) return -1.0f;
    if (!bvh.nodes || !bvh.triangleIndices || mesh.bvhNodeCount <= 0
        || !validBVHRange(mesh.bvhRoot, mesh.bvhNodeCount, bvh.nodeCount)
        || !validBVHRange(mesh.bvhIndexStart, mesh.triangleCount, bvh.indexCount)
        || stackCapacity <= 0 || stackCapacity > BVH_STACK_CAPACITY) {
        return fallbackMeshScan<CollectStats>(mesh, ray, triangles, normal, stats, triangleId);
    }
    Ray localRay;
    localRay.origin = multiplyMV(mesh.inverseTransform, glm::vec4(ray.origin, 1.0f));
    localRay.direction = multiplyMV(mesh.inverseTransform,
        glm::vec4(glm::normalize(ray.direction), 0.0f));

    struct StackEntry { int node; float entry; };
    StackEntry stack[BVH_STACK_CAPACITY];
    int size = 0;
    float closestT = FLT_MAX;
    int hitTriangle = -1;
    float rootEntry;
    if constexpr (CollectStats) ++stats->aabbTests;
    if (!intersectBVHNode(bvh.nodes[mesh.bvhRoot], localRay, closestT, rootEntry)) return -1.0f;
    stack[size++] = {mesh.bvhRoot, rootEntry};
    if constexpr (CollectStats) stats->maxStack = max(stats->maxStack, size);
    const int nodeEnd = mesh.bvhRoot + mesh.bvhNodeCount;
    const int indexEnd = mesh.bvhIndexStart + mesh.triangleCount;
    const int triangleEnd = mesh.triangleStart + mesh.triangleCount;
    int poppedNodes = 0;
    while (size > 0) {
        const StackEntry current = stack[--size];
        if (++poppedNodes > mesh.bvhNodeCount) {
            // Valid CPU trees visit each node at most once. Do not hang on a cycle.
            return fallbackMeshScan<CollectStats>(mesh, ray, triangles, normal, stats, triangleId);
        }
        // Equality must survive so equal-t hits can use the original triangle ID.
        if (current.entry > closestT) continue;
        const BVHNode& node = bvh.nodes[current.node];
        if constexpr (CollectStats) ++stats->nodeVisits;
        if (node.indexCount > 0) {
            if (node.leftChild != -1 || node.rightChild != -1
                || node.firstIndex < mesh.bvhIndexStart
                || !validBVHRange(node.firstIndex, node.indexCount, indexEnd)) {
                return fallbackMeshScan<CollectStats>(mesh, ray, triangles, normal, stats, triangleId);
            }
            for (int i = 0; i < node.indexCount; ++i) {
                const int triangleIndex = bvh.triangleIndices[node.firstIndex + i];
                if (triangleIndex < mesh.triangleStart || triangleIndex >= triangleEnd) {
                    return fallbackMeshScan<CollectStats>(mesh, ray, triangles, normal, stats, triangleId);
                }
                if constexpr (CollectStats) ++stats->triangleTests;
                const float t = triangleIntersectionTest(triangles[triangleIndex], localRay);
                if (t > 0.0f && (t < closestT
                    || (t == closestT && triangleIndex < hitTriangle))) {
                    closestT = t;
                    hitTriangle = triangleIndex;
                }
            }
            continue;
        }
        const int left = node.leftChild, right = node.rightChild;
        if (node.indexCount != 0 || node.firstIndex != -1 || left == right
            || left < mesh.bvhRoot || left >= nodeEnd
            || right < mesh.bvhRoot || right >= nodeEnd
            || left == current.node || right == current.node) {
            return fallbackMeshScan<CollectStats>(mesh, ray, triangles, normal, stats, triangleId);
        }
        float leftEntry, rightEntry;
        if constexpr (CollectStats) stats->aabbTests += 2;
        const bool hitLeft = intersectBVHNode(bvh.nodes[left], localRay, closestT, leftEntry);
        const bool hitRight = intersectBVHNode(bvh.nodes[right], localRay, closestT, rightEntry);
        const int needed = int(hitLeft) + int(hitRight);
        if (size + needed > stackCapacity) {
            return fallbackMeshScan<CollectStats>(mesh, ray, triangles, normal, stats, triangleId);
        }
        if (hitLeft && hitRight) {
            // LIFO: push farther child first; either order is correct for equal entry.
            const bool leftFirst = leftEntry <= rightEntry;
            stack[size++] = {leftFirst ? right : left, leftFirst ? rightEntry : leftEntry};
            stack[size++] = {leftFirst ? left : right, leftFirst ? leftEntry : rightEntry};
        } else if (hitLeft) {
            stack[size++] = {left, leftEntry};
        } else if (hitRight) {
            stack[size++] = {right, rightEntry};
        }
        if constexpr (CollectStats) stats->maxStack = max(stats->maxStack, size);
    }
    if (hitTriangle < 0) return -1.0f;
    normal = glm::normalize(multiplyMV(mesh.invTranspose,
        glm::vec4(triangles[hitTriangle].normal, 0.0f)));
    if (triangleId) *triangleId = hitTriangle;
    return closestT;
}
}

__host__ __device__ float meshBVHIntersectionTest(
    const Geom& mesh, const Ray& ray, const Triangle* triangles,
    const BVHDeviceView& bvh, glm::vec3& normal,
    BVHTraversalStats* stats, int stackCapacity, int* triangleId)
{
    if (triangleId) *triangleId = -1;
    if (stats) return traverseMeshBVH<true>(mesh, ray, triangles, bvh, normal, stats, stackCapacity, triangleId);
    return traverseMeshBVH<false>(mesh, ray, triangles, bvh, normal, nullptr, stackCapacity, triangleId);
}
