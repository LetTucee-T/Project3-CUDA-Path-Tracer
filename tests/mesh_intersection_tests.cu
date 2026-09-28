#include "intersections.h"
#include "appearance.h"
#include "interactions.h"
#include "pathtrace.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <algorithm>
#include <cfloat>
#include <limits>
#include <random>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Link and launch the renderer's actual kernel, rather than a test-side copy.
__global__ void computeIntersections(int depth, int num_paths,
    PathSegment* paths, Geom* geoms, int geoms_size,
    const Triangle* triangles, BVHDeviceView bvh, bool enableBVH,
    bool enableMeshCulling, ShadeableIntersection* intersections,
    BVHTraversalStats* diagnostics = nullptr, const TriangleSurface* surfaces = nullptr);

namespace {
void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void cudaCheck(cudaError_t status)
{
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class T> class DeviceArray {
public:
    T* data = nullptr;
    size_t count;
    explicit DeviceArray(size_t size) : count(size)
    {
        if (count) cudaCheck(cudaMalloc(&data, count * sizeof(T)));
    }
    explicit DeviceArray(const std::vector<T>& values) : DeviceArray(values.size())
    {
        if (count) cudaCheck(cudaMemcpy(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    std::vector<T> read() const
    {
        std::vector<T> values(count);
        if (count) cudaCheck(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

void near(float actual, float expected, float tolerance = 1e-4f)
{
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
        "Expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}

void nearVector(const glm::vec3& actual, const glm::vec3& expected, float tolerance = 1e-4f)
{
    for (int axis = 0; axis < 3; ++axis) near(actual[axis], expected[axis], tolerance);
}

Triangle triangle(float z = 0.0f, float size = 1.0f)
{
    Triangle value;
    value.v0 = glm::vec3(-size, -size, z);
    value.v1 = glm::vec3(size, -size, z);
    value.v2 = glm::vec3(0.0f, size, z);
    value.normal = glm::vec3(0.0f, 0.0f, 1.0f);
    return value;
}

Geom geometry(GeomType type, const glm::vec3& position = glm::vec3(0),
    const glm::vec3& rotation = glm::vec3(0), const glm::vec3& scale = glm::vec3(1),
    int material = 0, int start = 0, int count = 1)
{
    Geom value{};
    value.type = type;
    value.materialid = material;
    value.translation = position;
    value.rotation = rotation;
    value.scale = scale;
    value.transform = utilityCore::buildTransformationMatrix(position, rotation, scale);
    value.inverseTransform = glm::inverse(value.transform);
    value.invTranspose = glm::inverseTranspose(value.transform);
    value.triangleStart = start;
    value.triangleCount = count;
    return value;
}

PathSegment path(const glm::vec3& origin = glm::vec3(0, 0, 5),
    const glm::vec3& direction = glm::vec3(0, 0, -1), int bounces = 4)
{
    PathSegment value{};
    value.ray.origin = origin;
    value.ray.direction = direction;
    value.color = glm::vec3(1);
    value.pixelIndex = 37;
    value.remainingBounces = bounces;
    return value;
}

ShadeableIntersection sentinel()
{
    ShadeableIntersection value;
    value.t = -77.0f;
    value.materialId = -91;
    value.surfaceNormal = glm::vec3(9, 8, 7);
    return value;
}

// Only synthetic test fixtures need this; scenes loaded by Scene keep their
// original loader-computed bounds during full-render consistency checks.
void setFixtureBounds(std::vector<Geom>& geoms, const std::vector<Triangle>& triangles)
{
    for (auto& g : geoms) {
        if (g.type != MESH || g.triangleCount <= 0 || triangles.empty()) continue;
        require(g.triangleStart >= 0
            && size_t(g.triangleStart) + size_t(g.triangleCount) <= triangles.size(), "Bad fixture range");
        glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
        for (int j = 0; j < g.triangleCount; ++j) {
            const auto& t = triangles[g.triangleStart + j];
            for (const auto& v : {t.v0, t.v1, t.v2}) { lo = glm::min(lo, v); hi = glm::max(hi, v); }
        }
        float magnitude = 0;
        for (int a = 0; a < 3; ++a) magnitude = std::max(magnitude, std::max(std::abs(lo[a]), std::abs(hi[a])));
        g.boundsMin = lo - glm::vec3(8.0f * FLT_EPSILON * magnitude);
        g.boundsMax = hi + glm::vec3(8.0f * FLT_EPSILON * magnitude);
    }
}

std::vector<ShadeableIntersection> intersect(const std::vector<Geom>& geoms,
    const std::vector<Triangle>& triangles, const std::vector<PathSegment>& paths)
{
    auto boundedGeoms = geoms;
    setFixtureBounds(boundedGeoms, triangles);
    std::vector<BVHNode> nodes;
    std::vector<int> indices;
    for (Geom& g : boundedGeoms) {
        if (g.type != MESH || g.triangleCount <= 0 || triangles.empty()) continue;
        const auto built = buildMeshBVH(triangles, g.triangleStart, g.triangleCount, nodes, indices);
        g.bvhRoot = built.root; g.bvhNodeCount = built.nodeCount; g.bvhIndexStart = built.indexStart;
        validateMeshBVH(triangles, g, nodes, indices);
    }
    DeviceArray<BVHNode> gpuNodes(nodes);
    DeviceArray<int> gpuIndices(indices);
    BVHDeviceView bvh{gpuNodes.data, gpuIndices.data, int(nodes.size()), int(indices.size()), int(triangles.size())};
    DeviceArray<BVHTraversalStats> gpuStats(paths.size() + 3);
    DeviceArray<Geom> gpuGeoms(boundedGeoms);
    DeviceArray<Triangle> gpuTriangles(triangles);
    DeviceArray<PathSegment> gpuPaths(paths);
    DeviceArray<ShadeableIntersection> gpuHits(
        std::vector<ShadeableIntersection>(paths.size() + 3, sentinel()));
    const int count = static_cast<int>(paths.size());
    std::vector<ShadeableIntersection> hits, reference;
    const std::vector<ShadeableIntersection> initial(paths.size() + 3, sentinel());
    for (int mode = 0; mode < 4; ++mode) {
        const bool culling = (mode & 1) != 0;
        const bool useBVH = (mode & 2) != 0;
        cudaCheck(cudaMemset(gpuStats.data, 0, gpuStats.count * sizeof(BVHTraversalStats)));
        cudaCheck(cudaMemcpy(gpuHits.data, initial.data(), initial.size() * sizeof(ShadeableIntersection), cudaMemcpyHostToDevice));
        computeIntersections<<<(count + 127) / 128, 128>>>(0, count, gpuPaths.data,
            gpuGeoms.data, static_cast<int>(geoms.size()), gpuTriangles.data, bvh, useBVH,
            culling, gpuHits.data, gpuStats.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        hits = gpuHits.read();
        const auto stats = gpuStats.read();
        bool testedTree = false;
        for (size_t i = 0; i < stats.size(); ++i) {
            require(stats[i].fallbackCount == 0, "Valid tree unexpectedly fell back to brute force");
            if (i >= paths.size() || paths[i].remainingBounces <= 0 || !useBVH)
                require(stats[i].aabbTests == 0, "Inactive or non-BVH path traversed a tree");
            testedTree |= stats[i].aabbTests > 0;
        }
        if (useBVH && !nodes.empty() && std::any_of(paths.begin(), paths.end(),
            [](const PathSegment& p) { return p.remainingBounces > 0; }))
            require(testedTree, "BVH switch did not reach traversal");
        if (mode == 0) reference = hits;
        else for (size_t i = 0; i < hits.size(); ++i) {
            near(hits[i].t, reference[i].t, 0.0f);
            if (hits[i].t > 0) {
                require(hits[i].materialId == reference[i].materialId, "Culling changed material");
                nearVector(hits[i].surfaceNormal, reference[i].surfaceNormal, 0.0f);
            }
        }
    }
    for (size_t i = paths.size(); i < hits.size(); ++i) {
        near(hits[i].t, -77);
        require(hits[i].materialId == -91, "Wrote beyond active path range");
    }
    hits.resize(paths.size());
    return hits;
}

struct BoundsQuery { Ray ray; glm::vec3 lo, hi; };
__global__ void queryBounds(const BoundsQuery* queries, int* output, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = aabbIntersectionTest(queries[i].ray, queries[i].lo, queries[i].hi) ? 1 : 0;
}

struct TriangleQuery { Triangle triangle; Ray ray; };
__global__ void queryTriangles(const TriangleQuery* queries, float* distances, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) distances[i] = triangleIntersectionTest(queries[i].triangle, queries[i].ray);
}

__global__ void inspectUpload(const Triangle* triangles, const Geom* geoms,
    float* fields, int* sizes)
{
    const Triangle& t = triangles[1];
    for (int axis = 0; axis < 3; ++axis) {
        fields[axis] = t.v0[axis];
        fields[3 + axis] = t.v1[axis];
        fields[6 + axis] = t.v2[axis];
        fields[9 + axis] = t.normal[axis];
    }
    fields[12] = geoms[1].inverseTransform[3][0];
    sizes[0] = sizeof(Triangle);
    sizes[1] = sizeof(Geom);
    sizes[2] = geoms[1].triangleStart;
    sizes[3] = geoms[1].triangleCount;
    sizes[4] = geoms[1].materialid;
}

class RenderSession {
public:
    explicit RenderSession(Scene& scene) { pathtraceInit(&scene); }
    ~RenderSession() { pathtraceFree(); }
    RenderSession(const RenderSession&) = delete;
    RenderSession& operator=(const RenderSession&) = delete;
};

std::vector<glm::vec3> render(Scene& scene, int samples)
{
    const auto resolution = scene.state.camera.resolution;
    DeviceArray<uchar4> pbo(size_t(resolution.x) * resolution.y);
    RenderSession session(scene);
    for (int sample = 1; sample <= samples; ++sample) pathtrace(pbo.data, 0, sample);
    cudaCheck(cudaDeviceSynchronize());
    for (const auto& pixel : scene.state.image) {
        require(std::isfinite(pixel.x) && std::isfinite(pixel.y) && std::isfinite(pixel.z),
            "Non-finite radiance");
        require(pixel.x >= 0 && pixel.y >= 0 && pixel.z >= 0, "Negative radiance");
    }
    return scene.state.image;
}
}

#include "bvh_gpu_test_cases.h"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: mesh_intersection_tests SOURCE_ROOT\n";
        return 1;
    }
    int devices = 0;
    const cudaError_t availability = cudaGetDeviceCount(&devices);
    if (availability == cudaErrorNoDevice || availability == cudaErrorInsufficientDriver
        || (availability == cudaSuccess && devices == 0)) {
        std::cout << "SKIP: no available CUDA device\n";
        return 77;
    }
    if (availability != cudaSuccess) {
        std::cerr << cudaGetErrorString(availability) << '\n';
        return 1;
    }
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    std::cout << "GPU: " << properties.name << '\n';
    const auto root = std::filesystem::path(argv[1]);
    const auto sampleScene = root / "scenes/mesh_cpu_validation.json";
    int passed = 0, failed = 0;
    auto test = [&](const std::string& name, const std::function<void()>& fn) {
        try { fn(); ++passed; std::cout << "PASS: " << name << std::endl; }
        catch (const std::exception& error) {
            ++failed;
            std::cerr << "FAIL: " << name << ": " << error.what() << std::endl;
        }
    };

    test("textured smooth hit attributes agree across scan, AABB, BVH and fallback", [&] {
        Triangle t = triangle();
        std::vector<Triangle> ts{t};
        Geom g = geometry(MESH, glm::vec3(0), glm::vec3(0), glm::vec3(2,3,.5f), 0, 0, 1);
        g.smoothNormals = true; g.textured = true;
        std::vector<BVHNode> nodes; std::vector<int> indices;
        buildMeshBVH(ts, 0, 1, nodes, indices);
        g.bvhRoot=0; g.bvhNodeCount=int(nodes.size()); g.bvhIndexStart=0;
        g.boundsMin={-1,-1,-.001f};g.boundsMax={1,1,.001f};
        TriangleSurface surface{};
        surface.normals[0]={0,0,1};surface.normals[1]={.6f,0,.8f};surface.normals[2]={0,.6f,.8f};
        surface.uvs[0]={1,0};surface.uvs[1]={0,0};surface.uvs[2]={0,1};
        PathSegment path{};path.remainingBounces=3;path.ray.origin={0,0,3};path.ray.direction={0,0,-1};
        DeviceArray<Triangle> dt(ts);DeviceArray<TriangleSurface> ds(std::vector<TriangleSurface>{surface});
        DeviceArray<Geom> dg(std::vector<Geom>{g});DeviceArray<PathSegment> dp(std::vector<PathSegment>{path});
        DeviceArray<BVHNode> dn(nodes);DeviceArray<int> di(indices);DeviceArray<ShadeableIntersection> dh(1);
        BVHDeviceView view{dn.data,di.data,int(nodes.size()),int(indices.size()),1};
        for(int mode=0;mode<4;++mode){
            auto v=view;if(mode==3)v.nodes=nullptr;
            computeIntersections<<<1,1>>>(0,1,dp.data,dg.data,1,dt.data,v,mode>=2,mode==1,dh.data,nullptr,ds.data);
            cudaCheck(cudaDeviceSynchronize());const auto hit=dh.read()[0];
            near(hit.t,3);near(hit.uv.x,.25f);near(hit.uv.y,.5f);
            require(hit.hasUV && hit.useShadingNormal,"Missing GPU attributes");
            nearVector(hit.geometricNormal,{0,0,1});
            nearVector(hit.surfaceNormal,glm::normalize(glm::vec3(.075f,.1f,1.7f)));
        }
    });
    test("coated diffuse mixture conserves expected throughput", [&] {
        Material m{};m.color={.2f,.4f,.6f};m.coatWeight=.25f;
        thrust::default_random_engine rng(123);glm::dvec3 sum(0);const int n=100000;
        for(int i=0;i<n;++i){
            PathSegment path{};path.color=glm::vec3(1);path.ray.direction={0,0,-1};
            scatterRay(path,glm::vec3(0),glm::vec3(0,0,1),m,rng);
            require(path.ray.direction.z>0,"Mixture left reflecting hemisphere");sum+=glm::dvec3(path.color);
        }
        nearVector(glm::vec3(sum/double(n)),glm::vec3(.25f)+.75f*m.color,.004f);
    });

    test("GPU reads uploaded triangle fields, mesh ranges and ABI sizes", [&] {
        Triangle t = triangle();
        t.v0 = glm::vec3(1, 2, 3); t.v1 = glm::vec3(4, 5, 6); t.v2 = glm::vec3(7, 8, 9);
        t.normal = glm::vec3(0.2f, 0.3f, 0.4f);
        const Geom g = geometry(MESH, glm::vec3(3, 0, 0), glm::vec3(0), glm::vec3(2), 19, 7, 11);
        DeviceArray<Triangle> triangles(std::vector<Triangle>{triangle(), t});
        DeviceArray<Geom> geoms(std::vector<Geom>{geometry(CUBE), g});
        DeviceArray<float> fields(13); DeviceArray<int> sizes(5);
        inspectUpload<<<1, 1>>>(triangles.data, geoms.data, fields.data, sizes.data);
        cudaCheck(cudaGetLastError()); cudaCheck(cudaDeviceSynchronize());
        const auto f = fields.read(); const auto s = sizes.read();
        for (int i = 0; i < 9; ++i) near(f[i], float(i + 1));
        for (int i = 0; i < 3; ++i) near(f[9 + i], t.normal[i]);
        near(f[12], -1.5f);
        require(s == std::vector<int>{sizeof(Triangle), sizeof(Geom), 7, 11, 19}, "GPU layout/range mismatch");
    });

    struct BoundsCase { std::string name; BoundsQuery q; bool expected; };
    std::vector<BoundsCase> boundsCases;
    auto addBounds = [&](const std::string& name, glm::vec3 origin, glm::vec3 direction,
        bool expected, glm::vec3 lo = glm::vec3(-1), glm::vec3 hi = glm::vec3(1)) {
        BoundsQuery q; q.ray.origin = origin; q.ray.direction = direction; q.lo = lo; q.hi = hi;
        boundsCases.push_back({name, q, expected});
    };
    addBounds("front hit", glm::vec3(0, 0, 5), glm::vec3(0, 0, -1), true);
    addBounds("back hit", glm::vec3(0, 0, -5), glm::vec3(0, 0, 1), true);
    addBounds("box behind ray", glm::vec3(0, 0, 5), glm::vec3(0, 0, 1), false);
    addBounds("parallel outside slab", glm::vec3(2, 0, 5), glm::vec3(0, 0, -1), false);
    addBounds("parallel on min face", glm::vec3(-1, 0, 5), glm::vec3(0, 0, -1), true);
    addBounds("parallel on max face", glm::vec3(1, 0, 5), glm::vec3(-0.0f, 0, -1), true);
    addBounds("inside origin", glm::vec3(0), glm::vec3(1, 2, 3), true);
    addBounds("boundary outward", glm::vec3(1, 0, 0), glm::vec3(1, 0, 0), true);
    addBounds("edge touch", glm::vec3(2, 0, 0), glm::vec3(-1, 1, 0), true);
    addBounds("corner touch", glm::vec3(2), glm::vec3(-1), true);
    addBounds("disjoint axis intervals", glm::vec3(2, 4, 0), glm::vec3(-1, -1, 0), true);
    addBounds("disjoint axis intervals miss", glm::vec3(2, 5, 0), glm::vec3(-1, -1, 0), false);
    addBounds("tiny nonzero direction", glm::vec3(2, 0, 0), glm::vec3(-1e-12f, 0, 0), true);
    addBounds("tiny direction away", glm::vec3(2, 0, 0), glm::vec3(1e-12f, 0, 0), false);
    addBounds("flat box", glm::vec3(0, 0, 5), glm::vec3(0, 0, -1), true, glm::vec3(-1, -1, 0), glm::vec3(1, 1, 0));
    addBounds("parallel within flat box", glm::vec3(-2, 0, 0), glm::vec3(1, 0, 0), true, glm::vec3(-1, -1, 0), glm::vec3(1, 1, 0));
    addBounds("parallel outside flat box", glm::vec3(-2, 0, 0.001f), glm::vec3(1, 0, 0), false, glm::vec3(-1, -1, 0), glm::vec3(1, 1, 0));
    addBounds("small box", glm::vec3(0, 0, 5), glm::vec3(0, 0, -1), true, glm::vec3(-1e-20f), glm::vec3(1e-20f));
    addBounds("large box", glm::vec3(0, 0, 2e20f), glm::vec3(0, 0, -1), true, glm::vec3(-1e20f), glm::vec3(1e20f));
    addBounds("overflow falls back", glm::vec3(2, 0, 0), glm::vec3(-1e-40f, 0, 0), true);
    addBounds("nonfinite falls back", glm::vec3(0, std::numeric_limits<float>::quiet_NaN(), 0), glm::vec3(1), true);
    addBounds("invalid bounds fall back", glm::vec3(5), glm::vec3(1), true, glm::vec3(1), glm::vec3(-1));
    for (const auto& item : boundsCases) {
        test("AABB: " + item.name, [&] {
            DeviceArray<BoundsQuery> input(std::vector<BoundsQuery>{item.q});
            DeviceArray<int> output(1);
            queryBounds<<<1, 64>>>(input.data, output.data, 1);
            cudaCheck(cudaGetLastError()); cudaCheck(cudaDeviceSynchronize());
            require(output.read()[0] == int(item.expected), "Unexpected AABB result");
        });
    }

    test("8193 transformed rays: culling retains every brute-force hit", [&] {
        std::vector<Triangle> triangles = {triangle(), triangle(0.8f, 0.4f)};
        std::vector<Geom> geoms = {geometry(MESH, glm::vec3(1, 2, 3), glm::vec3(25, 47, 13), glm::vec3(-2, 0.5f, 4), 9, 0, 2)};
        std::mt19937 rng(7319);
        std::uniform_real_distribution<float> xy(-1.5f, 1.5f);
        std::vector<PathSegment> paths;
        for (int i = 0; i < 8193; ++i) {
            glm::vec3 target(xy(rng), xy(rng), 0);
            if (i % 4 == 0) target = triangles[0].v0;
            if (i % 4 == 1) target = (triangles[0].v0 + triangles[0].v1) * 0.5f;
            const glm::vec3 localOrigin = target + glm::vec3(0, 0, i % 2 ? 5.0f : -5.0f);
            const glm::vec3 worldTarget = multiplyMV(geoms[0].transform, glm::vec4(target, 1));
            const glm::vec3 worldOrigin = multiplyMV(geoms[0].transform, glm::vec4(localOrigin, 1));
            paths.push_back(path(worldOrigin, worldTarget - worldOrigin));
        }
        const auto hits = intersect(geoms, triangles, paths);
        size_t hitCount = 0;
        for (const auto& h : hits) if (h.t > 0) ++hitCount;
        require(hitCount > 100 && hitCount < hits.size() - 100, "Differential rays must contain hits and misses");
    });

    struct QueryCase { std::string name; TriangleQuery query; float expected; };
    std::vector<QueryCase> cases;
    auto add = [&](std::string name, Triangle t, glm::vec3 origin, glm::vec3 direction, float expected) {
        TriangleQuery q; q.triangle = t; q.ray.origin = origin; q.ray.direction = direction;
        cases.push_back({name, q, expected});
    };
    const glm::vec3 front(0, 0, 2), forward(0, 0, -1);
    add("front side", triangle(), front, forward, 2);
    add("back side", triangle(), -front, -forward, 2);
    Triangle reversed = triangle(); std::swap(reversed.v1, reversed.v2); reversed.normal *= -1.0f;
    add("reversed winding", reversed, front, forward, 2);
    add("outside triangle", triangle(), glm::vec3(2, 0, 2), forward, -1);
    add("outside sloping edge", triangle(), glm::vec3(0.8f, 0.8f, 2), forward, -1);
    add("parallel ray", triangle(), front, glm::vec3(1, 0, 0), -1);
    add("coplanar ray", triangle(), glm::vec3(0), glm::vec3(1, 0, 0), -1);
    add("intersection behind ray", triangle(), front, -forward, -1);
    add("edge hit", triangle(), glm::vec3(0, -1, 2), forward, 2);
    add("vertex hit", triangle(), glm::vec3(-1, -1, 2), forward, 2);
    add("origin on surface", triangle(), glm::vec3(0), forward, -1);
    add("near-zero intersection", triangle(), glm::vec3(0, 0, 1e-6f), forward, -1);
    add("unnormalized direction retains t", triangle(), front, forward * 2.0f, 1);
    add("zero direction", triangle(), front, glm::vec3(0), -1);
    Triangle degenerate = triangle(); degenerate.v2 = degenerate.v1;
    add("degenerate triangle", degenerate, front, forward, -1);
    degenerate.v0 = glm::vec3(-1, 0, 0); degenerate.v1 = glm::vec3(0); degenerate.v2 = glm::vec3(1, 0, 0);
    add("collinear triangle", degenerate, front, forward, -1);
    add("small triangle", triangle(0, 1e-4f), front, forward, 2);
    add("tiny triangle centered hit", triangle(0, 1e-20f), front, forward, 2);
    add("large triangle centered hit", triangle(0, 1e20f), front, forward, 2);
    const glm::vec3 grazing(1, 0, -1e-4f);
    add("grazing hit", triangle(), -2.0f * grazing, grazing, 2);
    for (const auto& item : cases) {
        test("triangle: " + item.name, [&] {
            DeviceArray<TriangleQuery> input(std::vector<TriangleQuery>{item.query});
            DeviceArray<float> output(1);
            queryTriangles<<<1, 64>>>(input.data, output.data, 1);
            cudaCheck(cudaGetLastError()); cudaCheck(cudaDeviceSynchronize());
            near(output.read()[0], item.expected);
        });
    }

    struct TransformCase { std::string name; glm::vec3 translation, rotation, scale; };
    const std::vector<TransformCase> transforms = {
        {"identity", glm::vec3(0), glm::vec3(0), glm::vec3(1)},
        {"translation", glm::vec3(3, -2, 7), glm::vec3(0), glm::vec3(1)},
        {"uniform scale", glm::vec3(0), glm::vec3(0), glm::vec3(3)},
        {"nonuniform scale", glm::vec3(0), glm::vec3(0), glm::vec3(2, 0.5f, 4)},
        {"negative scale", glm::vec3(0), glm::vec3(0), glm::vec3(-2, 1, 3)},
        {"rotation", glm::vec3(0), glm::vec3(25, 47, 13), glm::vec3(1)},
        {"combined transform", glm::vec3(1, 3, -2), glm::vec3(25, 47, 13), glm::vec3(2, 0.5f, 4)},
        {"combined negative transform", glm::vec3(-1, 2, 3), glm::vec3(25, 47, 13), glm::vec3(-2, 0.5f, 4)},
    };
    for (const auto& item : transforms) {
        test("mesh distance and normal: " + item.name, [&] {
            Triangle t = triangle(); t.v2.z = 1;
            t.normal = glm::normalize(glm::cross(t.v1 - t.v0, t.v2 - t.v0));
            const Geom g = geometry(MESH, item.translation, item.rotation, item.scale, 7);
            // Independent reference: intersect at a known transformed centroid;
            // obtain the normal from transformed edges, not inverse-transpose.
            const glm::vec3 a = multiplyMV(g.transform, glm::vec4(t.v0, 1));
            const glm::vec3 b = multiplyMV(g.transform, glm::vec4(t.v1, 1));
            const glm::vec3 c = multiplyMV(g.transform, glm::vec4(t.v2, 1));
            glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
            if (item.scale.x * item.scale.y * item.scale.z < 0) n = -n;
            const glm::vec3 center = (a + b + c) / 3.0f;
            auto hits = intersect({g}, {t}, {path(center + n * 5.0f, -n * 2.0f),
                                          path(center - n * 5.0f, n)});
            for (const auto& hit : hits) {
                near(hit.t, 5); nearVector(hit.surfaceNormal, n);
                require(hit.materialId == 7, "Wrong mesh material");
            }
        });
    }
    for (float zScale : {0.0001f, 10000.0f}) {
        test("mesh local direction scale " + std::to_string(zScale), [&] {
            Geom g = geometry(MESH, glm::vec3(0), glm::vec3(0), glm::vec3(1, 1, zScale));
            near(intersect({g}, {triangle()}, {path()})[0].t, 5);
        });
    }
    test("mesh range and nearest triangle", [&] {
        const Geom g = geometry(MESH, glm::vec3(0), glm::vec3(0), glm::vec3(1), 4, 1, 2);
        auto hit = intersect({g}, {triangle(4.5f), triangle(-1), triangle(2), triangle(4)}, {path()})[0];
        near(hit.t, 3); require(hit.materialId == 4, "Wrong material");
    });
    test("triangle order does not replace closer hit", [&] {
        const Geom g = geometry(MESH, glm::vec3(0), glm::vec3(0), glm::vec3(1), 4, 0, 2);
        near(intersect({g}, {triangle(2), triangle(-1)}, {path()})[0].t, 3);
    });
    for (bool reverse : {false, true}) {
        test("closest of multiple meshes, reverse=" + std::to_string(reverse), [&] {
            std::vector<Geom> geoms = {
                geometry(MESH, glm::vec3(0), glm::vec3(0), glm::vec3(1), 3, 0, 1),
                geometry(MESH, glm::vec3(0), glm::vec3(0), glm::vec3(1), 9, 1, 1)};
            if (reverse) std::reverse(geoms.begin(), geoms.end());
            const auto hit = intersect(geoms, {triangle(-1), triangle(2)}, {path()})[0];
            near(hit.t, 3); require(hit.materialId == 9, "Wrong closest mesh material");
        });
    }
    for (GeomType primitive : {SPHERE, CUBE}) {
        for (bool meshFirst : {false, true}) {
            test("mixed closest, primitive=" + std::to_string(int(primitive))
                + " meshFirst=" + std::to_string(meshFirst), [&] {
                Geom g = geometry(primitive, glm::vec3(0), glm::vec3(0), glm::vec3(2), 3);
                Geom m = geometry(MESH, glm::vec3(0), glm::vec3(0), glm::vec3(1), 9);
                const auto hit = intersect({g, m}, {triangle(meshFirst ? 2.0f : -2.0f)}, {path()})[0];
                near(hit.t, meshFirst ? 3.0f : 4.0f, 1e-3f);
                require(hit.materialId == (meshFirst ? 9 : 3), "Wrong mixed closest material");
            });
        }
    }
    test("primitive-only scene accepts null triangle pointer", [&] {
        near(intersect({geometry(SPHERE)}, {}, {path()})[0].t, 4.5f, 1e-3f);
    });
    test("null triangle pointer and zero triangle count miss", [&] {
        near(intersect({geometry(MESH)}, {}, {path()})[0].t, -1);
        auto g = geometry(MESH); g.triangleCount = 0;
        near(intersect({g}, {triangle()}, {path()})[0].t, -1);
    });
    test("513 paths: hits, misses, terminated lanes and launch bounds", [&] {
        std::vector<PathSegment> paths(513);
        for (int i = 0; i < 513; ++i)
            paths[i] = path(glm::vec3(i % 4 == 0 ? 10.0f : 0.0f, 0, 5),
                glm::vec3(0, 0, -1), i % 7 == 0 ? 0 : 4);
        const auto hits = intersect({geometry(MESH)}, {triangle()}, paths);
        for (int i = 0; i < 513; ++i)
            near(hits[i].t, i % 7 == 0 ? -77.0f : (i % 4 == 0 ? -1.0f : 5.0f));
    });

    test("production upload, nonzero mesh range, empty-array transitions and repeated cleanup", [&] {
        Scene scene(sampleScene.string());
        const auto cube = scene.triangles;
        std::vector<Triangle> twoMeshes = cube;
        for (auto& t : twoMeshes) { t.v0.x += 100; t.v1.x += 100; t.v2.x += 100; }
        twoMeshes.insert(twoMeshes.end(), cube.begin(), cube.end());
        Material light{}; light.emittance = 2; light.color = glm::vec3(0.1f, 0.3f, 0.5f);
        Material wrong = light; wrong.color = glm::vec3(1, 0, 0);
        scene.materials = {wrong, light};
        Camera& camera = scene.state.camera;
        camera.resolution = glm::ivec2(33, 17); camera.position = glm::vec3(0, 0, 5);
        camera.view = glm::vec3(0, 0, -1); camera.up = glm::vec3(0, 1, 0);
        camera.right = glm::vec3(1, 0, 0); camera.pixelLength = glm::vec2(0.004f);
        scene.state.image.resize(33 * 17); scene.state.traceDepth = 4;
        pathtraceFree();
        for (int cycle = 0; cycle < 8; ++cycle) {
            if (cycle % 2 == 0) {
                scene.triangles = twoMeshes;
                scene.geoms = {
                    geometry(MESH, glm::vec3(0), glm::vec3(0), glm::vec3(1), 0, 0, 12),
                    geometry(MESH, glm::vec3(0), glm::vec3(0), glm::vec3(1), 1, 12, 12)};
            } else {
                scene.triangles.clear();
                scene.geoms = {geometry(SPHERE, glm::vec3(0), glm::vec3(0), glm::vec3(2), 1)};
            }
            scene.state.enableBVH = (cycle & 4) != 0;
            scene.state.enableAntialiasing = true;
            scene.state.enableMaterialSorting = (cycle & 2) != 0;
            scene.state.enableStreamCompaction = (cycle & 4) != 0;
            setFixtureBounds(scene.geoms, scene.triangles);
            scene.rebuildMeshBVHs();
            const auto image = render(scene, 3);
            for (const auto& pixel : image) nearVector(pixel, light.color * 6.0f, 1e-5f);
            pathtraceFree(); // RenderSession already freed it; must remain safe.
        }
    });
    for (bool mirror : {false, true}) {
        for (bool backSide : {false, true}) {
            test("analytic mesh scattering, mirror=" + std::to_string(mirror)
                + " backSide=" + std::to_string(backSide), [&] {
                Scene scene(sampleScene.string());
                // An emitting OBJ cube surrounds the camera and a flat triangle.
                // Every scattered ray must reach the enclosure on bounce two.
                scene.triangles.push_back(triangle(0, 4));
                scene.geoms = {
                    geometry(MESH, glm::vec3(0), glm::vec3(0), glm::vec3(10), 1, 0, 12),
                    geometry(MESH, glm::vec3(0), glm::vec3(0), glm::vec3(2, 0.5f, -1), 0, 12, 1)};
                setFixtureBounds(scene.geoms, scene.triangles);
                scene.rebuildMeshBVHs();
                scene.state.enableBVH = true;
                Material surface{}; surface.color = glm::vec3(0.5f, 0.6f, 0.7f);
                surface.hasReflective = mirror ? 1.0f : 0.0f;
                Material light{}; light.color = glm::vec3(0.2f, 0.3f, 0.4f); light.emittance = 2;
                scene.materials = {surface, light};
                Camera& camera = scene.state.camera;
                camera.resolution = glm::ivec2(17, 15);
                camera.position = glm::vec3(0, 0, backSide ? -5.0f : 5.0f);
                camera.view = glm::vec3(0, 0, backSide ? 1.0f : -1.0f);
                camera.up = glm::vec3(0, 1, 0); camera.right = glm::vec3(1, 0, 0);
                camera.pixelLength = glm::vec2(0.004f);
                scene.state.image.resize(17 * 15);
                scene.state.enableAntialiasing = true;
                scene.state.enableStreamCompaction = true;
                scene.state.enableMaterialSorting = true;
                scene.state.traceDepth = 1;
                for (const auto& pixel : render(scene, 2)) nearVector(pixel, glm::vec3(0), 0);
                scene.state.traceDepth = 2;
                for (const auto& pixel : render(scene, 4))
                    nearVector(pixel, surface.color * light.color * 8.0f, 1e-5f);
            });
        }
    }
    for (bool aa : {false, true}) {
        std::vector<glm::vec3> reference;
        for (int mode = 0; mode < 16; ++mode) {
            test("full mesh path tracing, AA=" + std::to_string(aa) + " mode=" + std::to_string(mode), [&] {
                Scene scene(sampleScene.string());
                scene.state.enableAntialiasing = aa;
                scene.state.enableStreamCompaction = (mode & 1) != 0;
                scene.state.enableMaterialSorting = (mode & 2) != 0;
                scene.state.enableMeshCulling = (mode & 4) != 0;
                scene.state.enableBVH = (mode & 8) != 0;
                const auto image = render(scene, 16);
                require(std::any_of(image.begin(), image.end(), [](const glm::vec3& p) {
                    return p.x > 0 || p.y > 0 || p.z > 0;
                }), "Mesh scene produced no radiance");
                if (mode == 0) reference = image;
                else {
                    require(reference.size() == image.size(), "Image size changed");
                    for (size_t i = 0; i < image.size(); ++i) nearVector(image[i], reference[i], 0.0f);
                }
            });
        }
    }
    runBVHGPUTests(test, sampleScene);
    std::cout << "GPU RESULT: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
