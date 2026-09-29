#pragma once

#include "intersections.h"
#include <vector>

// Work counts from a separate replay of sampled incoming paths, not timings.
// No counters or diagnostic buffers are used by the normal intersection kernel.
struct MeshIntersectionWork {
    unsigned long long queries = 0, boundedQueries = 0, hits = 0, rootRejects = 0;
    unsigned long long nodeVisits = 0, aabbTests = 0, triangleTests = 0, fallbacks = 0;
    int maxStack = 0;
    void add(const MeshIntersectionWork& r) {
        queries += r.queries; boundedQueries += r.boundedQueries; hits += r.hits;
        rootRejects += r.rootRejects; nodeVisits += r.nodeVisits; aabbTests += r.aabbTests;
        triangleTests += r.triangleTests; fallbacks += r.fallbacks;
        if (r.maxStack > maxStack) maxStack = r.maxStack;
    }
};

struct BounceIntersectionProfile {
    unsigned long long sampledRays = 0, mismatches = 0;
    std::vector<MeshIntersectionWork> perGeometry;
};

namespace intersectionProfile {
void init(int pixelCount, int geometryCount, int pixelStride);
void free();
BounceIntersectionProfile collect(const PathSegment* paths, int count, const Geom* geoms,
    const Triangle* triangles, BVHDeviceView bvh, const ShadeableIntersection* actual,
    bool distancePruning);
}
