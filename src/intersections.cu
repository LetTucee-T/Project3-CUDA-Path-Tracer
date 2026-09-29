#include <climits>
#include <type_traits>
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
// Round both products independently: reversing an edge must negate its value,
// including at a shared edge. Contracting just one product into an FMA breaks
// that symmetry. The projection itself may (and does) use a consistent FMA.
__host__ __device__ double edgeDifference(double a, double b, double c, double d)
{
#ifdef __CUDA_ARCH__
    return __dsub_rn(__dmul_rn(a, b), __dmul_rn(c, d));
#else
    volatile double first = a * b, second = c * d;
    return first - second;
#endif
}
}

__host__ __device__ __noinline__ float triangleIntersectionTest(
    const Triangle& triangle,
    const Ray& ray, glm::vec3* barycentrics)
{
    // Ray-aligned edge functions (Woop/Benthin/Wald; PBRT triangle chapter).
    // Promote BEFORE subtraction: skinny faces far from the origin lose useful
    // bits if only the final determinant is recomputed in double precision.
    if (barycentrics) *barycentrics = glm::vec3(0);
    // Non-finite inputs propagate to the final ordered distance test, which
    // rejects them. Avoid repeating 15 finite checks for every visited triangle.
    const glm::dvec3 d(ray.direction);
    int z = 0;
    if (fabs(d.y) > fabs(d[z])) z = 1;
    if (fabs(d.z) > fabs(d[z])) z = 2;
    if (d[z] == 0.0) return -1.0f;
    const int x = (z + 1) % 3, y = (x + 1) % 3;
    const glm::dvec3 e1 = glm::dvec3(triangle.v1) - glm::dvec3(triangle.v0);
    const glm::dvec3 e2 = glm::dvec3(triangle.v2) - glm::dvec3(triangle.v0);
    if (edgeDifference(e1.y,e2.z,e1.z,e2.y) == 0.0
        && edgeDifference(e1.z,e2.x,e1.x,e2.z) == 0.0
        && edgeDifference(e1.x,e2.y,e1.y,e2.x) == 0.0) return -1.0f;

    const glm::dvec3 a = glm::dvec3(triangle.v0) - glm::dvec3(ray.origin);
    const glm::dvec3 b = glm::dvec3(triangle.v1) - glm::dvec3(ray.origin);
    const glm::dvec3 c = glm::dvec3(triangle.v2) - glm::dvec3(ray.origin);
    const double inverseZ = 1.0 / d[z];
    const double sx = d[x] * inverseZ, sy = d[y] * inverseZ;
    const double ax = fma(-sx,a[z],a[x]), ay = fma(-sy,a[z],a[y]);
    const double bx = fma(-sx,b[z],b[x]), by = fma(-sy,b[z],b[y]);
    const double cx = fma(-sx,c[z],c[x]), cy = fma(-sy,c[z],c[y]);
    const double wa = edgeDifference(bx,cy,by,cx);
    const double wb = edgeDifference(cx,ay,cy,ax);
    const double wc = edgeDifference(ax,by,ay,bx);
    if ((wa < 0 || wb < 0 || wc < 0) && (wa > 0 || wb > 0 || wc > 0)) return -1.0f;
    const double area = wa + wb + wc;
    if (area == 0.0) return -1.0f;
    const double inverseArea = 1.0 / area;
    const double distance = (wa*a[z] + wb*b[z] + wc*c[z]) * inverseArea * inverseZ;
    // Preserve the existing near-origin convention and inclusive float limits.
    const float t = float(distance);
    if (!(t > 1e-5f && t < FLT_MAX)) return -1.0f;
    if (barycentrics) *barycentrics = glm::vec3(float(wa*inverseArea),float(wb*inverseArea),float(wc*inverseArea));
    return t;
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
    glm::vec3& normal, int* triangleId, float maxDistance)
{
    if (triangleId) *triangleId = -1;
    if (triangles == nullptr || mesh.triangleCount <= 0 || maxDistance < 0) return -1.0f;

    Ray localRay;
    localRay.origin = multiplyMV(mesh.inverseTransform, glm::vec4(ray.origin, 1.0f));
    // Normalize in WORLD space only. With an unnormalized local direction, the
    // same t is a world distance, even with nonuniform or negative scaling.
    localRay.direction = multiplyMV(mesh.inverseTransform,
        glm::vec4(glm::normalize(ray.direction), 0.0f));

    float closestT = isfinite(maxDistance) ? maxDistance : FLT_MAX;
    float entry, exit;
    if (enableMeshCulling
        && !aabbIntervalIntersectionTest(localRay, mesh.boundsMin, mesh.boundsMax, closestT, entry, exit)) {
        return -1.0f;
    }

    int hitTriangle = -1;
    for (int i = 0; i < mesh.triangleCount; ++i) {
        const int triangleIndex = mesh.triangleStart + i;
        const float t = triangleIntersectionTest(triangles[triangleIndex], localRay);
        if (t > 0.0f && (t < closestT || (t == closestT && hitTriangle < 0))) {
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

// Accessors preserve the checked traversal's logical node fields in either
// representation. INT_MIN is not a valid leaf encoding (avoid signed overflow
// for corrupted direct callers); validated arrays cannot contain it.
template<bool Validated>
__host__ __device__ int leafCount(const BVHNode& node) { return node.indexCount; }
template<bool Validated>
__host__ __device__ int leafCount(const CompactBVHNode& node) {
    if constexpr (!Validated) { if (node.countOrRight == INT_MIN) return -1; }
    return node.countOrRight < 0 ? -node.countOrRight : 0;
}
__host__ __device__ int firstIndex(const BVHNode& node) { return node.firstIndex; }
__host__ __device__ int firstIndex(const CompactBVHNode& node) {
    return node.countOrRight < 0 ? node.firstOrLeft : -1;
}
__host__ __device__ int leftChild(const BVHNode& node) { return node.leftChild; }
__host__ __device__ int leftChild(const CompactBVHNode& node) {
    return node.countOrRight < 0 ? -1 : node.firstOrLeft;
}
__host__ __device__ int rightChild(const BVHNode& node) { return node.rightChild; }
__host__ __device__ int rightChild(const CompactBVHNode& node) {
    return node.countOrRight < 0 ? -1 : node.countOrRight;
}

template<class Node>
__host__ __device__ bool intersectBVHNode(
    const Node& node, const Ray& ray, float closestT, float& entry)
{
    float exit;
    return aabbIntervalIntersectionTest(ray,
        glm::vec3(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]),
        glm::vec3(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]),
        closestT, entry, exit);
}

struct CachedBVHRay {
    glm::vec3 inverseDirection;
    bool usable;
    __host__ __device__ explicit CachedBVHRay(const Ray& ray) : usable(true) {
        for(int axis=0;axis<3;++axis) {
            const float d=ray.direction[axis];
            inverseDirection[axis]=1.0f/d;
            usable = usable && isfinite(ray.origin[axis]) && isfinite(d)
                && d!=0.0f && isfinite(inverseDirection[axis]);
        }
    }
};

template<bool Validated, class Node>
__host__ __device__ bool intersectCachedBVHNode(const Node& node,const Ray& ray,
    const CachedBVHRay& cached,float limit,float& entry)
{
    // Exact-zero/tiny/invalid directions keep the conservative reference path.
    if(!cached.usable) return intersectBVHNode(node,ray,limit,entry);
    entry=0;float exit=limit;
    if constexpr(!Validated) {
        for(int axis=0;axis<3;++axis) {
            if(!isfinite(node.boundsMin[axis]) || !isfinite(node.boundsMax[axis])
                || node.boundsMin[axis]>node.boundsMax[axis]) return true;
        }
    }
    for(int axis=0;axis<3;++axis) {
        const float a=(node.boundsMin[axis]-ray.origin[axis])*cached.inverseDirection[axis];
        const float b=(node.boundsMax[axis]-ray.origin[axis])*cached.inverseDirection[axis];
        if(!isfinite(a) || !isfinite(b)) { entry=0;return true; }
        float nearT=fminf(a,b),farT=fmaxf(a,b);
        // Subtraction, reciprocal and multiplication each round. Widen more
        // than the division-based reference; never use fast-math reciprocals.
        nearT=nextafterf(nearT-6.0f*FLT_EPSILON*fabsf(nearT),-INFINITY);
        farT=nextafterf(farT+6.0f*FLT_EPSILON*fabsf(farT),INFINITY);
        entry=fmaxf(entry,nearT);exit=fminf(exit,farT);
        if(entry>exit)return false;
    }
    return true;
}

template<bool CollectStats>
__host__ __device__ float fallbackMeshScan(
    const Geom& mesh, const Ray& ray, const Triangle* triangles,
    glm::vec3& normal, BVHTraversalStats* stats, int* triangleId, float maxDistance)
{
    if constexpr (CollectStats) {
        ++stats->fallbackCount;
        stats->triangleTests += mesh.triangleCount;
    }
    // Recompute the complete closest hit, including triangles already visited.
    return meshIntersectionTest(mesh, ray, triangles, false, normal, triangleId, maxDistance);
}

template<class Node, bool CollectStats, bool CacheBounds, bool Validated = false>
__host__ __device__ float traverseMeshBVH(
    const Geom& mesh, const Ray& ray, const Triangle* triangles,
    const BVHDeviceView& bvh, glm::vec3& normal,
    BVHTraversalStats* stats, int stackCapacity, int* triangleId, float maxDistance)
{
    const Node* nodes;
    if constexpr (std::is_same<Node, CompactBVHNode>::value) nodes = bvh.compactNodes;
    else nodes = bvh.nodes;
    if (triangleId) *triangleId = -1;
    if (!triangles || mesh.triangleCount <= 0 || maxDistance < 0) return -1.0f;
    // Scene validation guarantees the triangle range. Also guard direct callers.
    if constexpr(!Validated) {
        if (!validBVHRange(mesh.triangleStart, mesh.triangleCount, bvh.triangleCount)) return -1.0f;
        if (!nodes || !bvh.triangleIndices || mesh.bvhNodeCount <= 0
            || !validBVHRange(mesh.bvhRoot, mesh.bvhNodeCount, bvh.nodeCount)
            || !validBVHRange(mesh.bvhIndexStart, mesh.triangleCount, bvh.indexCount)) {
            return fallbackMeshScan<CollectStats>(mesh, ray, triangles, normal, stats, triangleId, maxDistance);
        }
    }
    if (stackCapacity <= 0 || stackCapacity > BVH_STACK_CAPACITY)
        return fallbackMeshScan<CollectStats>(mesh, ray, triangles, normal, stats, triangleId, maxDistance);
    Ray localRay;
    localRay.origin = multiplyMV(mesh.inverseTransform, glm::vec4(ray.origin, 1.0f));
    localRay.direction = multiplyMV(mesh.inverseTransform,
        glm::vec4(glm::normalize(ray.direction), 0.0f));
    // The unused constructor is eliminated from the reference specialization.
    const CachedBVHRay cached(localRay);
    const auto intersectNode = [&](const Node& node,float limit,float& entry) {
        if constexpr(CacheBounds) return intersectCachedBVHNode<Validated>(node,localRay,cached,limit,entry);
        else return intersectBVHNode(node,localRay,limit,entry);
    };

    struct StackEntry { int node; float entry; };
    // Cached traversal keeps the active near child in registers and stores
    // only deferred siblings. The logical capacity still includes that child.
    StackEntry stack[CacheBounds ? BVH_STACK_CAPACITY - 1 : BVH_STACK_CAPACITY];
    int size = 0;
    float closestT = isfinite(maxDistance) ? maxDistance : FLT_MAX;
    int hitTriangle = -1;
    float rootEntry;
    if constexpr (CollectStats) ++stats->aabbTests;
    if (!intersectNode(nodes[mesh.bvhRoot], closestT, rootEntry)) return -1.0f;
    StackEntry current{mesh.bvhRoot,rootEntry};
    stack[0] = current;
    if constexpr(!CacheBounds) size = 1;
    const auto advance = [&]() {
        if constexpr(CacheBounds) {
            if(size>0) current=stack[--size];
            else current.node=-1;
        }
    };
    if constexpr (CollectStats) stats->maxStack = max(stats->maxStack, 1);
    [[maybe_unused]] const int nodeEnd = mesh.bvhRoot + mesh.bvhNodeCount;
    [[maybe_unused]] const int indexEnd = mesh.bvhIndexStart + mesh.triangleCount;
    [[maybe_unused]] const int triangleEnd = mesh.triangleStart + mesh.triangleCount;
    [[maybe_unused]] int poppedNodes = 0;
    while (CacheBounds ? current.node>=0 : size>0) {
        if constexpr(!CacheBounds) current = stack[--size];
        if constexpr(!Validated) {
            if (++poppedNodes > mesh.bvhNodeCount) {
                // Direct callers may supply a cycle; CPU-validated trees cannot.
                return fallbackMeshScan<CollectStats>(mesh, ray, triangles, normal, stats, triangleId, maxDistance);
            }
        }
        // Equality must survive so equal-t hits can use the original triangle ID.
        if (current.entry > closestT) { advance();continue; }
        const Node& node = nodes[current.node];
        if constexpr (CollectStats) ++stats->nodeVisits;
        const int count = leafCount<Validated>(node);
        if (count > 0) {
            if constexpr(!Validated) {
                if (leftChild(node) != -1 || rightChild(node) != -1
                    || firstIndex(node) < mesh.bvhIndexStart
                    || !validBVHRange(firstIndex(node), count, indexEnd)) {
                    return fallbackMeshScan<CollectStats>(mesh, ray, triangles, normal, stats, triangleId, maxDistance);
                }
            }
            for (int i = 0; i < count; ++i) {
                const int triangleIndex = bvh.triangleIndices[firstIndex(node) + i];
                if constexpr(!Validated) {
                    if (triangleIndex < mesh.triangleStart || triangleIndex >= triangleEnd) {
                        return fallbackMeshScan<CollectStats>(mesh, ray, triangles, normal, stats, triangleId, maxDistance);
                    }
                }
                if constexpr (CollectStats) ++stats->triangleTests;
                const float t = triangleIntersectionTest(triangles[triangleIndex], localRay);
                if (t > 0.0f && (t < closestT
                    || (t == closestT && (hitTriangle < 0 || triangleIndex < hitTriangle)))) {
                    closestT = t;
                    hitTriangle = triangleIndex;
                }
            }
            advance();continue;
        }
        const int left = leftChild(node), right = rightChild(node);
        if constexpr(!Validated) {
            if (count != 0 || firstIndex(node) != -1 || left == right
                || left < mesh.bvhRoot || left >= nodeEnd
                || right < mesh.bvhRoot || right >= nodeEnd
                || left == current.node || right == current.node) {
                return fallbackMeshScan<CollectStats>(mesh, ray, triangles, normal, stats, triangleId, maxDistance);
            }
        }
        float leftEntry, rightEntry;
        if constexpr (CollectStats) stats->aabbTests += 2;
        const bool hitLeft = intersectNode(nodes[left], closestT, leftEntry);
        const bool hitRight = intersectNode(nodes[right], closestT, rightEntry);
        const int needed = int(hitLeft) + int(hitRight);
        if (size + needed > stackCapacity) {
            return fallbackMeshScan<CollectStats>(mesh, ray, triangles, normal, stats, triangleId, maxDistance);
        }
        if constexpr(CacheBounds) {
            if(hitLeft && hitRight) {
                const bool leftFirst=leftEntry<=rightEntry;
                stack[size++]={leftFirst?right:left,leftFirst?rightEntry:leftEntry};
                current={leftFirst?left:right,leftFirst?leftEntry:rightEntry};
            }else if(hitLeft)current={left,leftEntry};
            else if(hitRight)current={right,rightEntry};
            else advance();
        } else if (hitLeft && hitRight) {
            // LIFO: push farther child first; either order is correct for equal entry.
            const bool leftFirst = leftEntry <= rightEntry;
            stack[size++] = {leftFirst ? right : left, leftFirst ? rightEntry : leftEntry};
            stack[size++] = {leftFirst ? left : right, leftFirst ? leftEntry : rightEntry};
        } else if (hitLeft) {
            stack[size++] = {left, leftEntry};
        } else if (hitRight) {
            stack[size++] = {right, rightEntry};
        }
        if constexpr (CollectStats) stats->maxStack = max(stats->maxStack,
            size + int(CacheBounds && current.node>=0));
    }
    if (hitTriangle < 0) return -1.0f;
    normal = glm::normalize(multiplyMV(mesh.invTranspose,
        glm::vec4(triangles[hitTriangle].normal, 0.0f)));
    if (triangleId) *triangleId = hitTriangle;
    return closestT;
}
}

template<class Node>
__host__ __device__ float dispatchMeshBVH(
    const Geom& mesh, const Ray& ray, const Triangle* triangles,
    const BVHDeviceView& bvh, glm::vec3& normal,
    BVHTraversalStats* stats, int stackCapacity, int* triangleId, float maxDistance)
{
    if (triangleId) *triangleId = -1;
    if(bvh.cachedBounds) {
        if(bvh.validated) {
            if(stats) return traverseMeshBVH<Node,true,true,true>(mesh, ray, triangles, bvh, normal, stats, stackCapacity, triangleId, maxDistance);
            return traverseMeshBVH<Node,false,true,true>(mesh, ray, triangles, bvh, normal, nullptr, stackCapacity, triangleId, maxDistance);
        }
        if(stats) return traverseMeshBVH<Node,true,true>(mesh, ray, triangles, bvh, normal, stats, stackCapacity, triangleId, maxDistance);
        return traverseMeshBVH<Node,false,true>(mesh, ray, triangles, bvh, normal, nullptr, stackCapacity, triangleId, maxDistance);
    }
    if (stats) return traverseMeshBVH<Node,true,false>(mesh, ray, triangles, bvh, normal, stats, stackCapacity, triangleId, maxDistance);
    return traverseMeshBVH<Node,false,false>(mesh, ray, triangles, bvh, normal, nullptr, stackCapacity, triangleId, maxDistance);
}

__host__ __device__ float meshBVHIntersectionTest(
    const Geom& mesh, const Ray& ray, const Triangle* triangles,
    const BVHDeviceView& bvh, glm::vec3& normal,
    BVHTraversalStats* stats, int stackCapacity, int* triangleId, float maxDistance)
{
    if (bvh.compactNodes)
        return dispatchMeshBVH<CompactBVHNode>(mesh, ray, triangles, bvh, normal,
            stats, stackCapacity, triangleId, maxDistance);
    return dispatchMeshBVH<BVHNode>(mesh, ray, triangles, bvh, normal,
        stats, stackCapacity, triangleId, maxDistance);
}
