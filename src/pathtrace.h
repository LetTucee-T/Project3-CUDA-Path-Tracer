#pragma once

#include "scene.h"
#include "utilities.h"
#include <array>
#include "intersectionProfile.h"

struct PathtraceOptions {
#ifdef NDEBUG
    bool synchronizeEachStage = false;
#else
    bool synchronizeEachStage = true;
#endif
    bool crossMeshPruning = true;
    bool cachedBVHBounds = true;
    bool validatedBVHTraversal = true;
    bool compactBVHNodes = true;
    bool profile = false;
    bool profileIntersections = false;
    int intersectionProfileStride = 64;
};

enum class RenderPhase { Prepare, Camera, Intersection, Sorting, Shading,
    Compaction, Gather, Display, Readback, Count };
struct PhaseTiming {
    double gpuMilliseconds = 0;
    double hostMilliseconds = 0;
    unsigned long long calls = 0;
};
struct PathtraceMetrics {
    std::array<PhaseTiming, static_cast<size_t>(RenderPhase::Count)> phases{};
    unsigned long long samples = 0, imageReadbacks = 0, readbackBytes = 0;
    unsigned long long stageSynchronizations = 0;
    double stageSynchronizationMilliseconds = 0;
    std::vector<BounceIntersectionProfile> intersectionWork;
};

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene, const PathtraceOptions& options = {});
void pathtraceFree();
// Accumulates on the GPU. A null PBO skips display conversion for headless use.
// Read the accumulated image explicitly before accessing RenderState::image.
void pathtrace(uchar4 *pbo, int frame, int iteration);
void pathtraceReadback();
PathtraceMetrics pathtraceGetMetrics();

#ifdef PATHTRACE_TESTING
#include "intersections.h"
// Read-only access to the actual renderer allocations for GPU upload probes.
BVHDeviceView pathtraceBVHForTesting();
const Geom* pathtraceGeomsForTesting();
#endif
