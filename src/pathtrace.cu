#include "pathtrace.h"
#include "cameraSampling.h"
#include "appearance.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <chrono>
#include <vector>
#include <algorithm>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

static PathtraceOptions executionOptions;
static PathtraceMetrics metrics;

static void checkCudaCall(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define CUDA_CHECK(call) checkCudaCall((call), #call)

namespace {
using ProfileClock = std::chrono::steady_clock;
struct ProfileEvents { cudaEvent_t start = nullptr, end = nullptr; RenderPhase phase; };
std::vector<ProfileEvents> profileEvents;
size_t profileUsed = 0;
cudaError_t profileError = cudaSuccess;
// Profile separately from speed benchmarks. Reuse event pairs and resolve once
// per sample. GPU stream intervals can include launch gaps, notably in Thrust.
struct ProfileScope {
    size_t slot = 0;
    ProfileClock::time_point hostStart;
    explicit ProfileScope(RenderPhase phase) {
        if (!executionOptions.profile) return;
        slot = profileUsed++;
        if (slot == profileEvents.size()) {
            profileEvents.push_back({});
            CUDA_CHECK(cudaEventCreate(&profileEvents.back().start));
            CUDA_CHECK(cudaEventCreate(&profileEvents.back().end));
        }
        auto& pair = profileEvents[slot]; pair.phase = phase;
        CUDA_CHECK(cudaEventRecord(pair.start));
        hostStart = ProfileClock::now();
    }
    ~ProfileScope() {
        if (!executionOptions.profile) return;
        auto& pair = profileEvents[slot];
        auto& timing = metrics.phases[static_cast<size_t>(pair.phase)];
        timing.hostMilliseconds += std::chrono::duration<double, std::milli>(ProfileClock::now() - hostStart).count();
        ++timing.calls;
        const auto result = cudaEventRecord(pair.end);
        if (result != cudaSuccess) profileError = result;
    }
};
void resolveProfile() {
    if (!executionOptions.profile || !profileUsed) return;
    checkCudaCall(profileError, "record profiling event");
    CUDA_CHECK(cudaEventSynchronize(profileEvents[profileUsed - 1].end));
    for (size_t i = 0; i < profileUsed; ++i) {
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, profileEvents[i].start, profileEvents[i].end));
        metrics.phases[static_cast<size_t>(profileEvents[i].phase)].gpuMilliseconds += ms;
    }
    profileUsed = 0;
}
}

struct IsTerminated
{
    __host__ __device__
    bool operator()(const PathSegment& path) const
    {
        return path.remainingBounces <= 0;
    }
};

enum MaterialSortKey : int
{
    SORT_DIFFUSE = 0,
    SORT_SPECULAR = 1,
    SORT_EMISSIVE = 2,
    SORT_MISS = 3,
    SORT_TERMINATED = 4,
    SORT_COATED = 5
};

static int* dev_materialKeys = nullptr;

void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
    const auto launchError = cudaGetLastError();
    if (launchError != cudaSuccess) {
        const std::string context = std::string(file) + ":" + std::to_string(line) + " " + msg;
        checkCudaCall(launchError, context.c_str());
    }
    if (executionOptions.synchronizeEachStage) {
        const auto start = ProfileClock::now();
        checkCudaCall(cudaDeviceSynchronize(), msg);
        ++metrics.stageSynchronizations;
        if (executionOptions.profile)
            metrics.stageSynchronizationMilliseconds += std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
    }
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image, bool displayTransform, float exposure)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];
        if (displayTransform) pix = displayColor(pix / float(iter), true, exposure) * float(iter);

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Triangle* dev_triangles = nullptr;
static TriangleSurface* dev_surfaces = nullptr;
static TextureInfo* dev_textures = nullptr;
static glm::vec3* dev_texturePixels = nullptr;
static BVHNode* dev_bvhNodes = nullptr;
static CompactBVHNode* dev_compactBVHNodes = nullptr;
static int* dev_bvhTriangleIndices = nullptr;
static BVHDeviceView dev_bvh;

#ifdef PATHTRACE_TESTING
BVHDeviceView pathtraceBVHForTesting() { return dev_bvh; }
const Geom* pathtraceGeomsForTesting() { return dev_geoms; }
#endif
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static unsigned int* dev_invalidCameraSamples = nullptr;
static glm::vec3* dev_sampleRadiance = nullptr;
static ShadeableIntersection* dev_intersections = NULL;
static unsigned int pendingCameraSamples = 0;
static bool imageDirty = false;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene, const PathtraceOptions& options)
{
    executionOptions = options;
    metrics = {};
    pendingCameraSamples = 0;
    imageDirty = false;
    // Revalidate before allocation: geometry edits must rebuild the matching CPU
    // tree. Camera/flag resets reuse the existing tree without reconstructing it.
    const auto maxCount = size_t(std::numeric_limits<int>::max());
    if (scene->bvhNodes.size() > maxCount || scene->bvhTriangleIndices.size() > maxCount
        || scene->triangles.size() > maxCount)
        throw std::runtime_error("BVH device arrays exceed 32-bit indexing");
    for (const Geom& geom : scene->geoms) {
        if (geom.type == MESH)
            validateMeshBVH(scene->triangles, geom, scene->bvhNodes, scene->bvhTriangleIndices);
    }
    // Packing follows topology validation, before any CUDA allocation. Keep the
    // original CPU nodes for rebuilding, diagnostics and the wide-layout mode.
    const auto compactNodes = options.compactBVHNodes
        ? packBVHNodes(scene->bvhNodes) : std::vector<CompactBVHNode>{};
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    if (options.profileIntersections) {
        if (!scene->state.enableBVH)
            throw std::runtime_error("Intersection profiling requires BVH enabled");
        intersectionProfile::init(pixelcount, static_cast<int>(scene->geoms.size()), options.intersectionProfileStride);
        metrics.intersectionWork.resize(scene->state.traceDepth);
        for (auto& bounce : metrics.intersectionWork) bounce.perGeometry.resize(scene->geoms.size());
    }

    CUDA_CHECK(cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3)));
    CUDA_CHECK(cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3)));
    imageDirty = true;

    CUDA_CHECK(cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment)));
    CUDA_CHECK(cudaMalloc(&dev_invalidCameraSamples, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(dev_invalidCameraSamples, 0, sizeof(unsigned int)));

    CUDA_CHECK(cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom)));
    CUDA_CHECK(cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice));

    // All meshes share one array; each Geom carries its own triangle range.
    // Primitive-only scenes leave dev_triangles null.
    if (!scene->triangles.empty()) {
        const size_t triangleBytes = scene->triangles.size() * sizeof(Triangle);
        CUDA_CHECK(cudaMalloc(&dev_triangles, triangleBytes));
        CUDA_CHECK(cudaMemcpy(dev_triangles, scene->triangles.data(), triangleBytes,
            cudaMemcpyHostToDevice));
    }

    if (!scene->bvhNodes.empty()) {
        if (options.compactBVHNodes) {
            const size_t bytes = compactNodes.size() * sizeof(CompactBVHNode);
            CUDA_CHECK(cudaMalloc(&dev_compactBVHNodes, bytes));
            CUDA_CHECK(cudaMemcpy(dev_compactBVHNodes, compactNodes.data(), bytes, cudaMemcpyHostToDevice));
        } else {
            const size_t bytes = scene->bvhNodes.size() * sizeof(BVHNode);
            CUDA_CHECK(cudaMalloc(&dev_bvhNodes, bytes));
            CUDA_CHECK(cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), bytes, cudaMemcpyHostToDevice));
        }
    }
    if (!scene->bvhTriangleIndices.empty()) {
        const size_t bytes = scene->bvhTriangleIndices.size() * sizeof(int);
        CUDA_CHECK(cudaMalloc(&dev_bvhTriangleIndices, bytes));
        CUDA_CHECK(cudaMemcpy(dev_bvhTriangleIndices, scene->bvhTriangleIndices.data(), bytes, cudaMemcpyHostToDevice));
    }
    dev_bvh = {dev_bvhNodes, dev_bvhTriangleIndices,
        static_cast<int>(scene->bvhNodes.size()),
        static_cast<int>(scene->bvhTriangleIndices.size()),
        static_cast<int>(scene->triangles.size())};
    dev_bvh.cachedBounds = options.cachedBVHBounds;
    dev_bvh.validated = options.validatedBVHTraversal;
    dev_bvh.compactNodes = dev_compactBVHNodes;

    CUDA_CHECK(cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

    if (!scene->surfaces.empty()) {
        CUDA_CHECK(cudaMalloc(&dev_surfaces, scene->surfaces.size()*sizeof(TriangleSurface)));
        CUDA_CHECK(cudaMemcpy(dev_surfaces, scene->surfaces.data(), scene->surfaces.size()*sizeof(TriangleSurface), cudaMemcpyHostToDevice));
    }
    if (!scene->textures.empty()) {
        CUDA_CHECK(cudaMalloc(&dev_textures, scene->textures.size()*sizeof(TextureInfo)));
        CUDA_CHECK(cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size()*sizeof(TextureInfo), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&dev_texturePixels, scene->texturePixels.size()*sizeof(glm::vec3)));
        CUDA_CHECK(cudaMemcpy(dev_texturePixels, scene->texturePixels.data(), scene->texturePixels.size()*sizeof(glm::vec3), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection)));
    CUDA_CHECK(cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection)));

    CUDA_CHECK(cudaMalloc(&dev_sampleRadiance, pixelcount * sizeof(glm::vec3)));
    CUDA_CHECK(cudaMalloc(
        &dev_materialKeys,
        pixelcount * sizeof(int)
    ));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    intersectionProfile::free();
    for (const auto& pair : profileEvents) {
        if (pair.start) cudaEventDestroy(pair.start);
        if (pair.end) cudaEventDestroy(pair.end);
    }
    profileEvents.clear(); profileUsed = 0; profileError = cudaSuccess;
    pendingCameraSamples = 0; imageDirty = false;
    // Null pointers also make cleanup safe before initialization and on reset.
    cudaFree(dev_image);
    dev_image = nullptr;
    cudaFree(dev_paths);
    dev_paths = nullptr;
    cudaFree(dev_invalidCameraSamples);
    dev_invalidCameraSamples = nullptr;
    cudaFree(dev_geoms);
    dev_geoms = nullptr;
    cudaFree(dev_surfaces); dev_surfaces = nullptr;
    cudaFree(dev_textures); dev_textures = nullptr;
    cudaFree(dev_texturePixels); dev_texturePixels = nullptr;
    cudaFree(dev_triangles);
    dev_triangles = nullptr;
    cudaFree(dev_bvhNodes);
    dev_bvhNodes = nullptr;
    cudaFree(dev_compactBVHNodes);
    dev_compactBVHNodes = nullptr;
    cudaFree(dev_bvhTriangleIndices);
    dev_bvhTriangleIndices = nullptr;
    dev_bvh = BVHDeviceView{};
    hst_scene = nullptr;
    cudaFree(dev_materials);
    dev_materials = nullptr;
    cudaFree(dev_intersections);
    dev_intersections = nullptr;
    cudaFree(dev_sampleRadiance);
    dev_sampleRadiance = nullptr;
    cudaFree(dev_materialKeys);
    dev_materialKeys = nullptr;
    checkCUDAError("pathtraceFree");
}

static void validateCameraSamples()
{
    if (!pendingCameraSamples) return;
    unsigned int invalid = 0;
    CUDA_CHECK(cudaMemcpy(&invalid, dev_invalidCameraSamples, sizeof(invalid), cudaMemcpyDeviceToHost));
    if (invalid)
        throw std::runtime_error("Thin-lens camera produced " + std::to_string(invalid)
            + " invalid rays; check camera axes, lens radius and focus distance.");
    CUDA_CHECK(cudaMemset(dev_invalidCameraSamples, 0, sizeof(unsigned int)));
    pendingCameraSamples = 0;
}

void pathtraceReadback()
{
    if (!hst_scene || !dev_image) throw std::runtime_error("Readback requires an initialized renderer");
    if (!imageDirty) return;
    {
        ProfileScope scope(RenderPhase::Readback);
        validateCameraSamples();
        const size_t bytes = hst_scene->state.image.size() * sizeof(glm::vec3);
        CUDA_CHECK(cudaMemcpy(hst_scene->state.image.data(), dev_image, bytes, cudaMemcpyDeviceToHost));
        ++metrics.imageReadbacks; metrics.readbackBytes += bytes;
        imageDirty = false;
    }
    resolveProfile();
}

PathtraceMetrics pathtraceGetMetrics()
{
    resolveProfile();
    return metrics;
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(
    Camera cam,
    int iter,
    int traceDepth,
    bool enableAntialiasing,
    bool enableDepthOfField,
    PathSegment* pathSegments,
    unsigned int* invalidCameraSamples)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cam.resolution.x || y >= cam.resolution.y) {
        return;
    }

    int index = x + y * cam.resolution.x;

    float offsetX = 0.5f;
    float offsetY = 0.5f;

    if (enableAntialiasing) {
        thrust::default_random_engine rng =
            makeSeededRandomEngine(iter, index, 0);

        thrust::uniform_real_distribution<float> u01(
            0.0f, 1.0f
        );

        offsetX = u01(rng);
        offsetY = u01(rng);
    }

    float sampleX = static_cast<float>(x) + offsetX;
    float sampleY = static_cast<float>(y) + offsetY;

    PathSegment& segment = pathSegments[index];

    segment.ray.origin = cam.position;

    segment.ray.direction = glm::normalize(
        cam.view
        - cam.right * cam.pixelLength.x
          * (sampleX - cam.resolution.x * 0.5f)
        - cam.up * cam.pixelLength.y
          * (sampleY - cam.resolution.y * 0.5f)
    );

    segment.color = glm::vec3(1.0f);
    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;

    // Preserve the original AA samples and pinhole arithmetic exactly. Lens
    // samples use a separate stream indexed by the original pixel, before any
    // path sorting/compaction. A zero aperture never consumes that stream.
    if (enableDepthOfField && cam.lensRadius != 0.0f) {
        auto lensRng = cameraSampling::makeLensRandomEngine(
            static_cast<uint32_t>(iter), static_cast<uint32_t>(index));
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        const float u = u01(lensRng);
        const float v = u01(lensRng);
        Ray lensRay;
        if (cameraSampling::makeThinLensRay(cam, segment.ray, true, glm::vec2(u, v), lensRay)) {
            segment.ray = lensRay;
        } else {
            // Do not trace an invalid sample or silently call it a pinhole
            // sample. The host reports an error before gathering this frame.
            segment.color = glm::vec3(0.0f);
            segment.remainingBounces = 0;
            atomicAdd(invalidCameraSamples, 1u);
        }
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    const Triangle* triangles,
    BVHDeviceView bvh,
    bool enableBVH,
    bool enableMeshCulling,
    ShadeableIntersection* intersections
#ifdef PATHTRACE_TESTING
    , BVHTraversalStats* diagnostics = nullptr
#endif
    , const TriangleSurface* surfaces = nullptr
    , bool enableDistancePruning = true
    )
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        if (pathSegment.remainingBounces <= 0) {
            return;
        }

#ifdef PATHTRACE_TESTING
        BVHTraversalStats rayStats{};
#endif
        glm::vec3 normal;
        int nearestTriangle = -1;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            const Geom& geom = geoms[i];
            float t = -1.0f;

            int candidateTriangle = -1;
            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH)
            {
                if (enableBVH) {
                    t = meshBVHIntersectionTest(geom, pathSegment.ray, triangles, bvh, tmp_normal
#ifdef PATHTRACE_TESTING
                        , diagnostics ? &rayStats : nullptr
#else
                        , nullptr
#endif
                        , BVH_STACK_CAPACITY, &candidateTriangle,
                        enableDistancePruning ? t_min : FLT_MAX
                    );
                } else {
                    t = meshIntersectionTest(geom, pathSegment.ray, triangles, enableMeshCulling,
                        tmp_normal, &candidateTriangle, enableDistancePruning ? t_min : FLT_MAX);
                }
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                normal = tmp_normal;
                nearestTriangle = candidateTriangle;
            }
        }

#ifdef PATHTRACE_TESTING
        if (diagnostics) diagnostics[path_index] = rayStats;
#endif
        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            auto& hit = intersections[path_index];
            hit.surfaceNormal = normal;
            hit.geometricNormal = normal;
            hit.useShadingNormal = false;
            hit.hasUV = false;
            hit.uv = glm::vec2(0);
            const Geom& geom = geoms[hit_geom_index];
            if (surfaces && nearestTriangle >= 0 && (geom.smoothNormals || geom.textured)) {
                Ray localRay;
                localRay.origin = multiplyMV(geom.inverseTransform, glm::vec4(pathSegment.ray.origin, 1));
                localRay.direction = multiplyMV(geom.inverseTransform,
                    glm::vec4(glm::normalize(pathSegment.ray.direction), 0));
                glm::vec3 w;
                // Re-evaluate only the winning textured/smooth triangle. Its
                // edge weights avoid reconstructing UVs from a rounded float t.
                triangleIntersectionTest(triangles[nearestTriangle], localRay, &w);
                const auto& surface = surfaces[nearestTriangle];
                if (geom.smoothNormals) {
                    glm::vec3 n = surface.normals[0]*w.x + surface.normals[1]*w.y + surface.normals[2]*w.z;
                    n = multiplyMV(geom.invTranspose, glm::vec4(n, 0));
                    if (glm::dot(n,n) > 1e-20f) {
                        n = glm::normalize(n);
                        if (glm::dot(n, normal) < 0) n = -n;
                        hit.surfaceNormal = n;
                        hit.useShadingNormal = true;
                    }
                }
                if (geom.textured) {
                    hit.uv = surface.uvs[0]*w.x + surface.uvs[1]*w.y + surface.uvs[2]*w.z;
                    hit.hasUV = true;
                }
            }
        }
    }
}

__device__ void finishPath(
    PathSegment& path,
    glm::vec3 radiance,
    glm::vec3* sampleRadiance)
{
    sampleRadiance[path.pixelIndex] = radiance;

    path.color = radiance;
    path.remainingBounces = 0;
}

// Evaluate emission or scatter the next path segment using the surface BSDF.
__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    const ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    const Material* materials,
    glm::vec3* sampleRadiance, const TextureInfo* textures, const glm::vec3* pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_paths) {
        return;
    }

    PathSegment& path = pathSegments[idx];

    if (path.remainingBounces <= 0) {
        return;
    }

    const ShadeableIntersection& intersection =
        shadeableIntersections[idx];

    if (intersection.t <= 0.0f) {
        finishPath(
            path,
            glm::vec3(0.0f),
            sampleRadiance
        );
        return;
    }

    Material material = materials[intersection.materialId];
    if (material.baseColorTexture >= 0 && intersection.hasUV)
        material.color *= sampleBaseColor(textures[material.baseColorTexture], pixels, intersection.uv);

    if (material.emittance > 0.0f) {
        glm::vec3 radiance =
            path.color * (material.color * material.emittance);

        finishPath(path, radiance, sampleRadiance);
        return;
    }

    if (path.remainingBounces <= 1) {
        finishPath(
            path,
            glm::vec3(0.0f),
            sampleRadiance
        );
        return;
    }

    glm::vec3 hitPoint =
        path.ray.origin
        + intersection.t * glm::normalize(path.ray.direction);

    thrust::default_random_engine rng =
        makeSeededRandomEngine(
            iter,
            path.pixelIndex,
            depth + 1
        );

    scatterRay(
        path,
        hitPoint,
        intersection.surfaceNormal,
        material,
        rng,
        intersection.useShadingNormal ? &intersection.geometricNormal : nullptr
    );

    --path.remainingBounces;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(
    int pixelcount,
    glm::vec3* image,
    const glm::vec3* sampleRadiance)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= pixelcount) {
        return;
    }

    image[idx] += sampleRadiance[idx];
}

__global__ void buildMaterialSortKeys(
    int numPaths,
    const PathSegment* paths,
    const ShadeableIntersection* intersections,
    const Material* materials,
    int* keys)
{
    const int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPaths) {
        return;
    }

    const PathSegment& path = paths[idx];

    if (path.remainingBounces <= 0) {
        keys[idx] = SORT_TERMINATED;
        return;
    }

    const ShadeableIntersection& hit =
        intersections[idx];

    if (hit.t <= 0.0f) {
        keys[idx] = SORT_MISS;
        return;
    }

    const Material& material =
        materials[hit.materialId];

    if (material.emittance > 0.0f) {
        keys[idx] = SORT_EMISSIVE;
    }
    else if (material.coatWeight > 0.0f) { keys[idx] = SORT_COATED; }
    else if (material.hasReflective > 0.0f) {
        keys[idx] = SORT_SPECULAR;
    }
    else {
        keys[idx] = SORT_DIFFUSE;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d((cam.resolution.x + 7) / 8, (cam.resolution.y + 7) / 8);
    const int blockSize1d = 128;
    const bool enableCompaction = hst_scene->state.enableStreamCompaction;
    const bool enableSorting = hst_scene->state.enableMaterialSorting;
    const bool useLens = hst_scene->state.enableDepthOfField && cam.lensRadius != 0.0f;
    {
        ProfileScope scope(RenderPhase::Prepare);
        CUDA_CHECK(cudaMemset(dev_sampleRadiance, 0, pixelcount * sizeof(glm::vec3)));
    }
    {
        ProfileScope scope(RenderPhase::Camera);
        generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth,
            hst_scene->state.enableAntialiasing, hst_scene->state.enableDepthOfField,
            dev_paths, dev_invalidCameraSamples);
    }
    checkCUDAError("generate camera rays");
    // Keep invalid-ray detection, but amortize the diagnostic counter readback.
    // Always validate again before exposing the CPU image to saving/tests.
    if (useLens) {
        ++pendingCameraSamples;
        const unsigned int batchLimit = static_cast<unsigned int>(
            (std::min)(size_t(32), size_t(std::numeric_limits<unsigned int>::max()) / size_t(pixelcount)));
        if (executionOptions.synchronizeEachStage || pendingCameraSamples >= batchLimit) {
            ProfileScope scope(RenderPhase::Readback);
            validateCameraSamples();
        }
    }
    int numPaths = pixelcount;
    for (int depth = 0; depth < traceDepth && numPaths > 0; ++depth) {
        const int pathBlocks = (numPaths + blockSize1d - 1) / blockSize1d;
        {
            ProfileScope scope(RenderPhase::Intersection);
            computeIntersections<<<pathBlocks, blockSize1d>>>(depth, numPaths,
                dev_paths, dev_geoms, static_cast<int>(hst_scene->geoms.size()),
                dev_triangles, dev_bvh, hst_scene->state.enableBVH,
                hst_scene->state.enableMeshCulling, dev_intersections,
#ifdef PATHTRACE_TESTING
                nullptr,
#endif
                dev_surfaces, executionOptions.crossMeshPruning);
        }
        checkCUDAError("compute intersections");
        if (executionOptions.profileIntersections) {
            const auto replay = intersectionProfile::collect(dev_paths, numPaths, dev_geoms,
                dev_triangles, dev_bvh, dev_intersections, executionOptions.crossMeshPruning);
            auto& total = metrics.intersectionWork[depth];
            total.sampledRays += replay.sampledRays; total.mismatches += replay.mismatches;
            for (size_t g = 0; g < total.perGeometry.size(); ++g)
                total.perGeometry[g].add(replay.perGeometry[g]);
        }
        if (enableSorting && numPaths > 1) {
            {
                ProfileScope scope(RenderPhase::Sorting);
                buildMaterialSortKeys<<<pathBlocks, blockSize1d>>>(numPaths, dev_paths,
                    dev_intersections, dev_materials, dev_materialKeys);
                checkCUDAError("build material sort keys");
                auto pairedValues = thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections));
                thrust::sort_by_key(thrust::device, dev_materialKeys,
                    dev_materialKeys + numPaths, pairedValues);
            }
            checkCUDAError("sort paths by material");
        }
        {
            ProfileScope scope(RenderPhase::Shading);
            shadeMaterial<<<pathBlocks, blockSize1d>>>(iter, depth, numPaths,
                dev_intersections, dev_paths, dev_materials,
                dev_sampleRadiance, dev_textures, dev_texturePixels);
        }
        checkCUDAError("shade materials");
        if (enableCompaction) {
            {
                ProfileScope scope(RenderPhase::Compaction);
                PathSegment* newEnd = thrust::remove_if(thrust::device,
                    dev_paths, dev_paths + numPaths, IsTerminated{});
                numPaths = static_cast<int>(newEnd - dev_paths);
            }
            checkCUDAError("compact paths");
        }
        if (guiData) guiData->TracedDepth = depth + 1;
    }
    {
        ProfileScope scope(RenderPhase::Gather);
        finalGather<<<(pixelcount + blockSize1d - 1) / blockSize1d, blockSize1d>>>(
            pixelcount, dev_image, dev_sampleRadiance);
    }
    checkCUDAError("final gather");
    imageDirty = true;
    if (pbo) {
        ProfileScope scope(RenderPhase::Display);
        sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter,
            dev_image, hst_scene->state.displayTransform, hst_scene->state.exposure);
        checkCUDAError("display conversion");
    }
    ++metrics.samples;
    resolveProfile();
}
