#include "intersectionProfile.h"
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
struct Probe { MeshIntersectionWork work; unsigned int mismatch = 0; };
Probe* device = nullptr;
std::vector<Probe> host;
int geometryCount = 0, pixelStride = 1;
void check(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("Intersection profiling: ") + cudaGetErrorString(result));
}

// Replays rays before sorting/shading can change them. Pixel identity makes the
// selected population independent of compaction and material sorting order.
__global__ void replay(const PathSegment* paths, int count, const Geom* geoms, int objects,
    const Triangle* triangles, BVHDeviceView bvh, const ShadeableIntersection* actual,
    bool distancePruning, int stride, Probe* output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || paths[i].remainingBounces <= 0 || paths[i].pixelIndex % stride != 0) return;
    const Ray ray = paths[i].ray;
    Probe* row = output + size_t(paths[i].pixelIndex / stride) * objects;
    float closest = FLT_MAX;
    int winner = -1;
    for (int g = 0; g < objects; ++g) {
        const Geom& geom = geoms[g];
        MeshIntersectionWork work{};
        work.queries = 1;
        const float limit = distancePruning ? closest : FLT_MAX;
        glm::vec3 normal, point;
        bool outside;
        float t = -1;
        if (geom.type == MESH) {
            BVHTraversalStats stats{};
            t = meshBVHIntersectionTest(geom, ray, triangles, bvh, normal, &stats,
                BVH_STACK_CAPACITY, nullptr, limit);
            work.boundedQueries = limit < FLT_MAX;
            work.nodeVisits = stats.nodeVisits; work.aabbTests = stats.aabbTests;
            work.triangleTests = stats.triangleTests; work.fallbacks = stats.fallbackCount;
            work.maxStack = stats.maxStack;
            work.rootRejects = stats.aabbTests == 1 && stats.nodeVisits == 0 && stats.fallbackCount == 0;
        } else if (geom.type == CUBE) t = boxIntersectionTest(geom, ray, point, normal, outside);
        else if (geom.type == SPHERE) t = sphereIntersectionTest(geom, ray, point, normal, outside);
        work.hits = t > 0;
        row[g].work = work;
        if (t > 0 && t < closest) { closest = t; winner = g; }
    }
    const bool same = winner < 0 ? actual[i].t < 0
        : actual[i].t == closest && actual[i].materialId == geoms[winner].materialid;
    row[0].mismatch = !same;
}
}

namespace intersectionProfile {
void init(int pixelCount, int objects, int stride) {
    if (pixelCount <= 0 || objects < 0 || stride < 1)
        throw std::runtime_error("Invalid intersection profiling dimensions/stride");
    geometryCount = objects; pixelStride = stride;
    const size_t samples = (size_t(pixelCount) + size_t(stride) - 1) / size_t(stride);
    if (size_t(objects) > (std::numeric_limits<size_t>::max)() / sizeof(Probe) / samples)
        throw std::runtime_error("Intersection profiling buffer size overflow");
    host.resize(samples * size_t(objects));
    if (!host.empty()) check(cudaMalloc(&device, host.size() * sizeof(Probe)));
}
void free() {
    cudaFree(device); device = nullptr;
    std::vector<Probe>().swap(host); geometryCount = 0;
}
BounceIntersectionProfile collect(const PathSegment* paths, int count, const Geom* geoms,
    const Triangle* triangles, BVHDeviceView bvh, const ShadeableIntersection* actual,
    bool distancePruning) {
    BounceIntersectionProfile result;
    result.perGeometry.resize(geometryCount);
    if (host.empty() || count == 0) return result;
    check(cudaMemset(device, 0, host.size() * sizeof(Probe)));
    replay<<<(count + 127) / 128, 128>>>(paths, count, geoms, geometryCount, triangles,
        bvh, actual, distancePruning, pixelStride, device);
    check(cudaGetLastError());
    check(cudaMemcpy(host.data(), device, host.size() * sizeof(Probe), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < host.size(); ++i) {
        const auto& probe = host[i];
        const size_t g = i % geometryCount;
        result.perGeometry[g].add(probe.work);
        result.mismatches += probe.mismatch;
        if (g == 0) result.sampledRays += probe.work.queries;
    }
    if (result.mismatches)
        throw std::runtime_error("Intersection profiler replay disagrees with production hits");
    return result;
}
}
