#include "bvh.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

static_assert(std::is_trivially_copyable<BVHNode>::value,
    "BVH nodes must support direct host-to-device copies");

namespace {
using Clock = std::chrono::steady_clock;

double elapsedMilliseconds(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void checkRange(int first, int count, std::size_t size, const std::string& label)
{
    if (first < 0 || count < 0 || std::size_t(first) > size
        || std::size_t(count) > size - std::size_t(first)) {
        throw std::runtime_error(label + " range is out of bounds");
    }
}

struct PrimitiveInfo
{
    int triangleIndex;
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    glm::dvec3 centroid;
};

PrimitiveInfo makePrimitiveInfo(const Triangle& triangle, int index)
{
    PrimitiveInfo info;
    info.triangleIndex = index;
    const float limit = (std::numeric_limits<float>::max)();
    info.boundsMin = glm::vec3(limit);
    info.boundsMax = glm::vec3(-limit);
    double scale = 0.0;
    for (const auto& vertex : {triangle.v0, triangle.v1, triangle.v2}) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(vertex[axis])) {
                throw std::runtime_error("BVH build: non-finite vertex in triangle "
                    + std::to_string(index));
            }
            info.boundsMin[axis] = (std::min)(info.boundsMin[axis], vertex[axis]);
            info.boundsMax[axis] = (std::max)(info.boundsMax[axis], vertex[axis]);
            scale = (std::max)(scale, std::abs(double(vertex[axis])));
        }
    }
    // Match the loader's coordinate-scaled margin, with one further outward ULP
    // for rounded/subnormal endpoints. Clamp before stepping to stay finite.
    const double margin = 8.0 * std::numeric_limits<float>::epsilon() * scale;
    for (int axis = 0; axis < 3; ++axis) {
        float lo = float((std::max)(-double(limit), double(info.boundsMin[axis]) - margin));
        float hi = float((std::min)( double(limit), double(info.boundsMax[axis]) + margin));
        if (lo > -limit) lo = std::nextafter(lo, -std::numeric_limits<float>::infinity());
        if (hi <  limit) hi = std::nextafter(hi,  std::numeric_limits<float>::infinity());
        info.boundsMin[axis] = lo;
        info.boundsMax[axis] = hi;
    }
    // Float vertices are summed in double, including when their sum exceeds FLT_MAX.
    info.centroid = (glm::dvec3(triangle.v0) + glm::dvec3(triangle.v1)
        + glm::dvec3(triangle.v2)) / 3.0;
    return info;
}

class Builder
{
public:
    Builder(std::vector<PrimitiveInfo>& primitives, std::vector<BVHNode>& nodes,
        std::vector<int>& indices, const BVHBuildOptions& options)
        : primitives(primitives), nodes(nodes), indices(indices), options(options) {}

    int build(std::size_t first, std::size_t end, int depth)
    {
        if (nodes.size() >= std::size_t((std::numeric_limits<int>::max)())) {
            throw std::runtime_error("BVH build: node count exceeds int indexing");
        }
        const int nodeIndex = static_cast<int>(nodes.size());
        nodes.emplace_back();
        ++stats.nodeCount;
        stats.maxDepth = (std::max)(stats.maxDepth, depth);
        const std::size_t count = end - first;
        if (count <= std::size_t(options.maxLeafTriangles) || depth >= options.maxDepth) {
            // Canonical leaf order also makes rebuilds independent of nth_element's
            // unspecified ordering within a partition.
            std::sort(primitives.begin() + first, primitives.begin() + end,
                [](const PrimitiveInfo& a, const PrimitiveInfo& b) {
                    return a.triangleIndex < b.triangleIndex;
                });
            BVHNode& node = nodes[nodeIndex];
            node.firstIndex = static_cast<int>(indices.size());
            node.indexCount = static_cast<int>(count);
            for (int axis = 0; axis < 3; ++axis) {
                node.boundsMin[axis] = primitives[first].boundsMin[axis];
                node.boundsMax[axis] = primitives[first].boundsMax[axis];
            }
            for (std::size_t i = first; i < end; ++i) {
                for (int axis = 0; axis < 3; ++axis) {
                    node.boundsMin[axis] = (std::min)(node.boundsMin[axis], primitives[i].boundsMin[axis]);
                    node.boundsMax[axis] = (std::max)(node.boundsMax[axis], primitives[i].boundsMax[axis]);
                }
                indices.push_back(primitives[i].triangleIndex);
            }
            ++stats.leafCount;
            stats.maxLeafTriangles = (std::max)(stats.maxLeafTriangles, node.indexCount);
            return nodeIndex;
        }

        glm::dvec3 centroidMin = primitives[first].centroid;
        glm::dvec3 centroidMax = centroidMin;
        for (std::size_t i = first + 1; i < end; ++i) {
            centroidMin = glm::min(centroidMin, primitives[i].centroid);
            centroidMax = glm::max(centroidMax, primitives[i].centroid);
        }
        const glm::dvec3 extent = centroidMax - centroidMin;
        int axis = 0; // X, then Y, then Z resolves equal extents deterministically.
        if (extent.y > extent[axis]) axis = 1;
        if (extent.z > extent[axis]) axis = 2;
        const std::size_t middle = first + count / 2;
        std::nth_element(primitives.begin() + first, primitives.begin() + middle,
            primitives.begin() + end, [axis](const PrimitiveInfo& a, const PrimitiveInfo& b) {
                if (a.centroid[axis] != b.centroid[axis])
                    return a.centroid[axis] < b.centroid[axis];
                return a.triangleIndex < b.triangleIndex;
            });
        const int left = build(first, middle, depth + 1);
        const int right = build(middle, end, depth + 1);
        // Recursive appends may reallocate nodes: acquire the reference afterwards.
        BVHNode& node = nodes[nodeIndex];
        node.leftChild = left;
        node.rightChild = right;
        for (int axis = 0; axis < 3; ++axis) {
            node.boundsMin[axis] = (std::min)(nodes[left].boundsMin[axis], nodes[right].boundsMin[axis]);
            node.boundsMax[axis] = (std::max)(nodes[left].boundsMax[axis], nodes[right].boundsMax[axis]);
        }
        return nodeIndex;
    }

    BVHStats stats;

private:
    std::vector<PrimitiveInfo>& primitives;
    std::vector<BVHNode>& nodes;
    std::vector<int>& indices;
    const BVHBuildOptions& options;
};
} // namespace

BVHBuildResult buildMeshBVH(const std::vector<Triangle>& triangles,
    int triangleStart, int triangleCount, std::vector<BVHNode>& nodes,
    std::vector<int>& triangleIndices, const BVHBuildOptions& options)
{
    const auto start = Clock::now();
    if (options.maxLeafTriangles < 1 || options.maxDepth < 0 || options.maxDepth > BVH_MAX_DEPTH) {
        throw std::runtime_error("BVH build: invalid leaf size or depth limit");
    }
    checkRange(triangleStart, triangleCount, triangles.size(), "BVH triangle");
    const std::size_t indexLimit = std::size_t((std::numeric_limits<int>::max)());
    if (nodes.size() > indexLimit || triangleIndices.size() > indexLimit
        || std::size_t(triangleStart) + std::size_t(triangleCount) > indexLimit
        || std::size_t(triangleCount) > indexLimit - triangleIndices.size()) {
        throw std::runtime_error("BVH build: array size exceeds int indexing");
    }
    const std::size_t oldNodeCount = nodes.size();
    const std::size_t oldIndexCount = triangleIndices.size();
    BVHBuildResult result;
    result.indexStart = static_cast<int>(oldIndexCount);
    try {
        if (triangleCount > 0) {
            std::vector<PrimitiveInfo> primitives;
            primitives.reserve(std::size_t(triangleCount));
            for (int i = 0; i < triangleCount; ++i) {
                const int index = triangleStart + i;
                primitives.push_back(makePrimitiveInfo(triangles[index], index));
            }
            triangleIndices.reserve(oldIndexCount + std::size_t(triangleCount));
            Builder builder(primitives, nodes, triangleIndices, options);
            result.root = builder.build(0, primitives.size(), 0);
            result.nodeCount = static_cast<int>(nodes.size() - oldNodeCount);
            result.stats = builder.stats;
        }
        result.stats.buildMilliseconds = elapsedMilliseconds(start);
        return result;
    } catch (...) {
        nodes.resize(oldNodeCount);
        triangleIndices.resize(oldIndexCount);
        throw;
    }
}

BVHStats validateMeshBVH(const std::vector<Triangle>& triangles, const Geom& mesh,
    const std::vector<BVHNode>& nodes, const std::vector<int>& triangleIndices, int maxDepth)
{
    const auto start = Clock::now();
    const std::string prefix = "BVH validation (root " + std::to_string(mesh.bvhRoot) + "): ";
    auto fail = [&](const std::string& message) { throw std::runtime_error(prefix + message); };
    if (mesh.type != MESH) fail("expected a mesh");
    if (maxDepth < 0 || maxDepth > BVH_MAX_DEPTH) fail("invalid depth limit");
    checkRange(mesh.triangleStart, mesh.triangleCount, triangles.size(), prefix + "triangle");
    checkRange(mesh.bvhIndexStart, mesh.triangleCount, triangleIndices.size(), prefix + "index");
    BVHStats stats;
    if (mesh.triangleCount == 0) {
        if (mesh.bvhRoot != -1 || mesh.bvhNodeCount != 0) fail("empty mesh has nodes");
        stats.validationMilliseconds = elapsedMilliseconds(start);
        return stats;
    }
    if (mesh.bvhNodeCount <= 0) fail("nonempty mesh has no nodes");
    checkRange(mesh.bvhRoot, mesh.bvhNodeCount, nodes.size(), prefix + "node");
    const std::size_t nodeEnd = std::size_t(mesh.bvhRoot) + std::size_t(mesh.bvhNodeCount);
    const std::size_t indexEnd = std::size_t(mesh.bvhIndexStart) + std::size_t(mesh.triangleCount);
    const std::size_t triangleEnd = std::size_t(mesh.triangleStart) + std::size_t(mesh.triangleCount);
    std::vector<unsigned char> seenNodes(mesh.bvhNodeCount, 0);
    std::vector<unsigned char> seenIndices(mesh.triangleCount, 0);
    std::vector<unsigned char> seenTriangles(mesh.triangleCount, 0);
    std::vector<std::pair<int, int>> pending{{mesh.bvhRoot, 0}};
    auto checkNodeIndex = [&](int id) {
        if (id < mesh.bvhRoot || std::size_t(id) >= nodeEnd)
            fail("node " + std::to_string(id) + " is outside this mesh's node block");
    };
    auto checkBounds = [&](int id) {
        const auto& node = nodes[id];
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(node.boundsMin[axis]) || !std::isfinite(node.boundsMax[axis])
                || node.boundsMin[axis] > node.boundsMax[axis]) {
                fail("node " + std::to_string(id) + " has invalid bounds");
            }
        }
    };
    while (!pending.empty()) {
        const auto [id, depth] = pending.back();
        pending.pop_back();
        checkNodeIndex(id);
        if (seenNodes[id - mesh.bvhRoot]++)
            fail("node " + std::to_string(id) + " has a cycle or repeated reference");
        if (depth > maxDepth) fail("node " + std::to_string(id) + " exceeds depth limit");
        checkBounds(id);
        const BVHNode& node = nodes[id];
        ++stats.nodeCount;
        stats.maxDepth = (std::max)(stats.maxDepth, depth);
        if (node.indexCount < 0) fail("node " + std::to_string(id) + " has negative leaf count");
        if (node.indexCount > 0) {
            if (node.leftChild != -1 || node.rightChild != -1)
                fail("leaf " + std::to_string(id) + " has children");
            if (node.firstIndex < mesh.bvhIndexStart || std::size_t(node.firstIndex) > indexEnd
                || std::size_t(node.indexCount) > indexEnd - std::size_t(node.firstIndex))
                fail("leaf " + std::to_string(id) + " index range is outside this mesh");
            ++stats.leafCount;
            stats.maxLeafTriangles = (std::max)(stats.maxLeafTriangles, node.indexCount);
            for (int i = 0; i < node.indexCount; ++i) {
                const std::size_t slot = std::size_t(node.firstIndex) + std::size_t(i);
                if (seenIndices[slot - std::size_t(mesh.bvhIndexStart)]++)
                    fail("leaf " + std::to_string(id) + " overlaps an index range");
                const int triangleId = triangleIndices[slot];
                if (triangleId < mesh.triangleStart || std::size_t(triangleId) >= triangleEnd)
                    fail("leaf " + std::to_string(id) + " triangle " + std::to_string(triangleId)
                        + " is outside this mesh");
                if (seenTriangles[triangleId - mesh.triangleStart]++)
                    fail("triangle " + std::to_string(triangleId) + " occurs more than once");
                const auto& triangle = triangles[triangleId];
                for (const auto& vertex : {triangle.v0, triangle.v1, triangle.v2}) {
                    for (int axis = 0; axis < 3; ++axis) {
                        if (!std::isfinite(vertex[axis]) || vertex[axis] < node.boundsMin[axis]
                            || vertex[axis] > node.boundsMax[axis])
                            fail("leaf " + std::to_string(id) + " excludes a vertex of triangle "
                                + std::to_string(triangleId));
                    }
                }
            }
        } else {
            if (node.firstIndex != -1) fail("internal node " + std::to_string(id) + " has leaf data");
            for (int child : {node.leftChild, node.rightChild}) {
                checkNodeIndex(child);
                checkBounds(child);
                for (int axis = 0; axis < 3; ++axis) {
                    if (nodes[child].boundsMin[axis] < node.boundsMin[axis]
                        || nodes[child].boundsMax[axis] > node.boundsMax[axis])
                        fail("node " + std::to_string(id) + " does not contain child " + std::to_string(child));
                }
            }
            pending.emplace_back(node.rightChild, depth + 1);
            pending.emplace_back(node.leftChild, depth + 1);
        }
    }
    if (std::find(seenNodes.begin(), seenNodes.end(), 0) != seenNodes.end()) fail("unreachable node");
    if (std::find(seenIndices.begin(), seenIndices.end(), 0) != seenIndices.end()) fail("uncovered index slot");
    if (std::find(seenTriangles.begin(), seenTriangles.end(), 0) != seenTriangles.end()) fail("missing triangle");
    stats.validationMilliseconds = elapsedMilliseconds(start);
    return stats;
}
