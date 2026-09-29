#pragma once

#include "sceneStructs.h"

#include <cstddef>
#include <vector>

// Root depth is zero. Keep this limit explicit for the future GPU traversal stack.
constexpr int BVH_MAX_DEPTH = 32;

enum class BVHSplitMethod { Median, BinnedSAH };

struct BVHBuildOptions
{
    int maxLeafTriangles = 1;
    int maxDepth = BVH_MAX_DEPTH;
    BVHSplitMethod splitMethod = BVHSplitMethod::BinnedSAH;
    int binCount = 16;
};

struct BVHStats
{
    std::size_t nodeCount = 0;
    std::size_t leafCount = 0;
    int maxDepth = 0;
    int maxLeafTriangles = 0;
    double buildMilliseconds = 0.0;
    double validationMilliseconds = 0.0;
};

struct BVHBuildResult
{
    int root = -1;
    int nodeCount = 0;
    int indexStart = 0;
    BVHStats stats;
};

// Append one mesh's contiguous node/index blocks. All stored indices are global:
// child -> nodes, firstIndex -> triangleIndices, triangleIndices[] -> triangles.
// The source triangles are never reordered. On failure, appended data is removed.
// An empty range produces no nodes. The OBJ loader separately rejects empty OBJ files.
BVHBuildResult buildMeshBVH(
    const std::vector<Triangle>& triangles, int triangleStart, int triangleCount,
    std::vector<BVHNode>& nodes, std::vector<int>& triangleIndices,
    const BVHBuildOptions& options = BVHBuildOptions{});

// Independently checks graph reachability, exact triangle/index coverage, bounds,
// ownership of each mesh's ranges, and the depth limit. Throws with node context.
BVHStats validateMeshBVH(
    const std::vector<Triangle>& triangles, const Geom& mesh,
    const std::vector<BVHNode>& nodes, const std::vector<int>& triangleIndices,
    int maxDepth = BVH_MAX_DEPTH);

// Encode an already topology-validated node array without changing any bounds,
// node positions or primitive references. Reject unrepresentable node metadata.
std::vector<CompactBVHNode> packBVHNodes(const std::vector<BVHNode>& nodes);
