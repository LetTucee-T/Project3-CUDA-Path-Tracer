#include "pathtrace.h"
#include "cameraSampling.h"
#include "appearance.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <limits>
#include <stdexcept>
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

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

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
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
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
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
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
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_invalidCameraSamples, sizeof(unsigned int));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    // All meshes share one array; each Geom carries its own triangle range.
    // Primitive-only scenes leave dev_triangles null.
    if (!scene->triangles.empty()) {
        const size_t triangleBytes = scene->triangles.size() * sizeof(Triangle);
        cudaMalloc(&dev_triangles, triangleBytes);
        cudaMemcpy(dev_triangles, scene->triangles.data(), triangleBytes,
            cudaMemcpyHostToDevice);
    }

    if (!scene->bvhNodes.empty()) {
        const size_t bytes = scene->bvhNodes.size() * sizeof(BVHNode);
        cudaMalloc(&dev_bvhNodes, bytes);
        cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), bytes, cudaMemcpyHostToDevice);
    }
    if (!scene->bvhTriangleIndices.empty()) {
        const size_t bytes = scene->bvhTriangleIndices.size() * sizeof(int);
        cudaMalloc(&dev_bvhTriangleIndices, bytes);
        cudaMemcpy(dev_bvhTriangleIndices, scene->bvhTriangleIndices.data(), bytes, cudaMemcpyHostToDevice);
    }
    dev_bvh = {dev_bvhNodes, dev_bvhTriangleIndices,
        static_cast<int>(scene->bvhNodes.size()),
        static_cast<int>(scene->bvhTriangleIndices.size()),
        static_cast<int>(scene->triangles.size())};

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    if (!scene->surfaces.empty()) {
        cudaMalloc(&dev_surfaces, scene->surfaces.size()*sizeof(TriangleSurface));
        cudaMemcpy(dev_surfaces, scene->surfaces.data(), scene->surfaces.size()*sizeof(TriangleSurface), cudaMemcpyHostToDevice);
    }
    if (!scene->textures.empty()) {
        cudaMalloc(&dev_textures, scene->textures.size()*sizeof(TextureInfo));
        cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size()*sizeof(TextureInfo), cudaMemcpyHostToDevice);
        cudaMalloc(&dev_texturePixels, scene->texturePixels.size()*sizeof(glm::vec3));
        cudaMemcpy(dev_texturePixels, scene->texturePixels.data(), scene->texturePixels.size()*sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_sampleRadiance, pixelcount * sizeof(glm::vec3));
    cudaMalloc(
        &dev_materialKeys,
        pixelcount * sizeof(int)
    );

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
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
                        , BVH_STACK_CAPACITY, &candidateTriangle
                    );
                } else {
                    t = meshIntersectionTest(geom, pathSegment.ray, triangles, enableMeshCulling, tmp_normal, &candidateTriangle);
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
                const glm::vec3 worldPoint = pathSegment.ray.origin + t_min * glm::normalize(pathSegment.ray.direction);
                const glm::vec3 localPoint = multiplyMV(geom.inverseTransform, glm::vec4(worldPoint, 1));
                const glm::vec3 w = surfaceBarycentrics(triangles[nearestTriangle], localPoint);
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

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
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

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    const bool enableCompaction =
        hst_scene->state.enableStreamCompaction;

    cudaMemset(
        dev_sampleRadiance,
        0,
        pixelcount * sizeof(glm::vec3)
    );

    const bool useLens = hst_scene->state.enableDepthOfField && cam.lensRadius != 0.0f;
    if (useLens) cudaMemset(dev_invalidCameraSamples, 0, sizeof(unsigned int));

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(
        cam,
        iter,
        traceDepth,
        hst_scene->state.enableAntialiasing,
        hst_scene->state.enableDepthOfField,
        dev_paths,
        dev_invalidCameraSamples
    );
    checkCUDAError("generate camera rays");
    if (useLens) {
        unsigned int invalidSamples = 0;
        cudaMemcpy(&invalidSamples, dev_invalidCameraSamples, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        checkCUDAError("validate camera rays");
        if (invalidSamples != 0)
            throw std::runtime_error("Thin-lens camera produced " + std::to_string(invalidSamples)
                + " invalid rays; check camera axes, lens radius and focus distance.");
    }

    int numPaths = pixelcount;
    const bool enableSorting = hst_scene->state.enableMaterialSorting;

    for (int depth = 0;
        depth < traceDepth && numPaths > 0;
        ++depth)
    {
        int pathBlocks =
            (numPaths + blockSize1d - 1) / blockSize1d;

        computeIntersections<<<pathBlocks, blockSize1d>>>(
            depth,
            numPaths,
            dev_paths,
            dev_geoms,
            static_cast<int>(hst_scene->geoms.size()),
            dev_triangles,
            dev_bvh,
            hst_scene->state.enableBVH,
            hst_scene->state.enableMeshCulling,
            dev_intersections,
#ifdef PATHTRACE_TESTING
            nullptr,
#endif
            dev_surfaces
        );
        checkCUDAError("compute intersections");

        if (enableSorting && numPaths > 1) {
            buildMaterialSortKeys<<<pathBlocks, blockSize1d>>>(
                numPaths,
                dev_paths,
                dev_intersections,
                dev_materials,
                dev_materialKeys
            );
            checkCUDAError("build material sort keys");

            auto pairedValues = thrust::make_zip_iterator(
                thrust::make_tuple(
                    dev_paths,
                    dev_intersections
                )
            );

            thrust::sort_by_key(
                thrust::device,
                dev_materialKeys,
                dev_materialKeys + numPaths,
                pairedValues
            );
            checkCUDAError("sort paths by material");
        }

        shadeMaterial<<<pathBlocks, blockSize1d>>>(
            iter,
            depth,
            numPaths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_sampleRadiance, dev_textures, dev_texturePixels
        );
        checkCUDAError("shade materials");

        if (enableCompaction) {
            PathSegment* newEnd = thrust::remove_if(
                thrust::device,
                dev_paths,
                dev_paths + numPaths,
                IsTerminated{}
            );

            numPaths = static_cast<int>(newEnd - dev_paths);
            checkCUDAError("compact paths");
        }

        if (guiData != nullptr) {
            guiData->TracedDepth = depth + 1;
        }
    }

    int imageBlocks =
        (pixelcount + blockSize1d - 1) / blockSize1d;

    finalGather<<<imageBlocks, blockSize1d>>>(
        pixelcount,
        dev_image,
        dev_sampleRadiance
    );
    checkCUDAError("final gather");


    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image,
        hst_scene->state.displayTransform, hst_scene->state.exposure);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
