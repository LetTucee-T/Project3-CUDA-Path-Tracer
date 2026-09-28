// Included by the CUDA regression executable, after its shared test helpers.
#pragma once

namespace {
struct IntervalQuery { Ray ray; glm::vec3 lo, hi; float limit; };
struct IntervalResult { int hit; float enter, exit; };
__global__ void queryIntervals(const IntervalQuery* queries, IntervalResult* result, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto& q = queries[i];
    result[i].enter = result[i].exit = -123.0f;
    result[i].hit = aabbIntervalIntersectionTest(q.ray, q.lo, q.hi, q.limit,
        result[i].enter, result[i].exit);
}

struct BVHFixture {
    Geom mesh = geometry(MESH);
    std::vector<Triangle> triangles;
    std::vector<BVHNode> nodes;
    std::vector<int> indices;
    void build(const BVHBuildOptions& options = BVHBuildOptions{})
    {
        nodes.clear(); indices.clear();
        const auto result = buildMeshBVH(triangles, mesh.triangleStart, mesh.triangleCount,
            nodes, indices, options);
        mesh.bvhRoot = result.root;
        mesh.bvhNodeCount = result.nodeCount;
        mesh.bvhIndexStart = result.indexStart;
        validateMeshBVH(triangles, mesh, nodes, indices);
    }
};

struct TraversalProbe {
    float bruteT, bvhT, plainT;
    glm::vec3 bruteNormal, bvhNormal, plainNormal;
    BVHTraversalStats stats;
};
__global__ void probeTraversal(Geom mesh, const Triangle* triangles, BVHDeviceView bvh,
    const PathSegment* rays, int count, TraversalProbe* probes, int stackCapacity)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    TraversalProbe result{};
    result.bruteNormal = result.bvhNormal = result.plainNormal = glm::vec3(9, 8, 7);
    result.bruteT = meshIntersectionTest(mesh, rays[i].ray, triangles, false, result.bruteNormal);
    result.bvhT = meshBVHIntersectionTest(mesh, rays[i].ray, triangles, bvh, result.bvhNormal,
        &result.stats, stackCapacity);
    result.plainT = meshBVHIntersectionTest(mesh, rays[i].ray, triangles, bvh, result.plainNormal,
        nullptr, stackCapacity);
    probes[i] = result;
}

std::vector<TraversalProbe> probe(const BVHFixture& fixture, const std::vector<PathSegment>& rays,
    int capacity = BVH_STACK_CAPACITY, int fault = 0)
{
    DeviceArray<Triangle> triangles(fixture.triangles);
    DeviceArray<BVHNode> nodes(fixture.nodes);
    DeviceArray<int> indices(fixture.indices);
    DeviceArray<PathSegment> paths(rays);
    DeviceArray<TraversalProbe> results(rays.size());
    BVHDeviceView view{nodes.data, indices.data, int(nodes.count), int(indices.count), int(triangles.count)};
    if (fault == 1) view.nodes = nullptr;
    if (fault == 2) view.triangleIndices = nullptr;
    if (fault == 3) view.nodeCount = 0;
    if (fault == 4) view.indexCount = 0;
    probeTraversal<<<(int(rays.size()) + 127) / 128, 128>>>(fixture.mesh, triangles.data, view,
        paths.data, int(rays.size()), results.data, capacity);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    auto output = results.read();
    for (const auto& p : output) {
        near(p.bvhT, p.bruteT, 0);
        near(p.plainT, p.bruteT, 0);
        nearVector(p.bvhNormal, p.bruteNormal, 0);
        nearVector(p.plainNormal, p.bruteNormal, 0);
    }
    return output;
}

void setNodeBounds(BVHNode& node, glm::vec3 lo, glm::vec3 hi)
{
    for (int a = 0; a < 3; ++a) { node.boundsMin[a] = lo[a]; node.boundsMax[a] = hi[a]; }
}

BVHFixture twoLeafFixture(bool equalDistance)
{
    BVHFixture f;
    f.triangles = {triangle(), triangle(equalDistance ? 0.0f : -4.0f)};
    f.triangles[1].normal = glm::vec3(0, 0, -1);
    f.mesh.triangleCount = 2; f.mesh.bvhRoot = 0; f.mesh.bvhNodeCount = 3;
    f.nodes.resize(3); f.indices = {1, 0};
    f.nodes[0].leftChild = 1; f.nodes[0].rightChild = 2;
    setNodeBounds(f.nodes[0], glm::vec3(-2, -2, -5), glm::vec3(2, 2, 4));
    setNodeBounds(f.nodes[1], glm::vec3(-2, -2, -5), glm::vec3(2, 2, 4));
    setNodeBounds(f.nodes[2], glm::vec3(-2, -2, -1), glm::vec3(2, 2, 1));
    f.nodes[1].firstIndex = 0; f.nodes[1].indexCount = 1;
    f.nodes[2].firstIndex = 1; f.nodes[2].indexCount = 1;
    validateMeshBVH(f.triangles, f.mesh, f.nodes, f.indices);
    return f;
}

__global__ void inspectBVHUpload(BVHDeviceView view, const Geom* geoms, int geomCount,
    BVHNode* nodes, int* indices, int* metadata, int* sizes)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) { sizes[0] = sizeof(BVHNode); sizes[1] = sizeof(Geom); }
    if (i < view.nodeCount) {
        for (int a = 0; a < 3; ++a) {
            nodes[i].boundsMin[a] = view.nodes[i].boundsMin[a];
            nodes[i].boundsMax[a] = view.nodes[i].boundsMax[a];
        }
        nodes[i].leftChild = view.nodes[i].leftChild;
        nodes[i].rightChild = view.nodes[i].rightChild;
        nodes[i].firstIndex = view.nodes[i].firstIndex;
        nodes[i].indexCount = view.nodes[i].indexCount;
    }
    if (i < view.indexCount) indices[i] = view.triangleIndices[i];
    if (i < geomCount) {
        metadata[i * 4] = geoms[i].bvhRoot;
        metadata[i * 4 + 1] = geoms[i].bvhNodeCount;
        metadata[i * 4 + 2] = geoms[i].bvhIndexStart;
        metadata[i * 4 + 3] = geoms[i].triangleStart;
    }
}

void checkFreedBVH()
{
    const auto view = pathtraceBVHForTesting();
    require(!view.nodes && !view.triangleIndices && !view.nodeCount
        && !view.indexCount && !view.triangleCount && !pathtraceGeomsForTesting(),
        "Device pointers/counts were not cleared after free");
}

template<class Test> void runBVHGPUTests(Test& test, const std::filesystem::path& sampleScene)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    struct IntervalCase { std::string name; IntervalQuery query; bool hit; float enter, exit; };
    const glm::vec3 lo(-1), hi(1);
    const std::vector<IntervalCase> intervals = {
        {"full", {path().ray, lo, hi, 100}, true, 4, 6},
        {"clipped", {path().ray, lo, hi, 5}, true, 4, 5},
        {"ends before box", {path().ray, lo, hi, 3}, false, 0, 0},
        {"exact endpoint", {path().ray, lo, hi, 4}, true, 4, 4},
        {"inside", {path(glm::vec3(0), glm::vec3(1, 0, 0)).ray, lo, hi, 2}, true, 0, 1},
        {"negative limit", {path().ray, lo, hi, -1}, false, 0, 0},
        {"zero limit at boundary", {path(glm::vec3(1, 0, 0), glm::vec3(1, 0, 0)).ray, lo, hi, 0}, true, 0, 0},
        {"parallel outside", {path(glm::vec3(2, 0, 5)).ray, lo, hi, 10}, false, 0, 0},
        {"tiny direction", {path(glm::vec3(0, 0, 5), glm::vec3(0, 0, -1e-12f)).ray, lo, hi, 1e13f}, true, 4e12f, 6e12f},
        {"invalid bounds", {path().ray, hi, lo, 7}, true, 0, 7},
        {"NaN input", {path(glm::vec3(nan, 0, 5)).ray, lo, hi, 7}, true, 0, 7},
        {"NaN limit", {path().ray, lo, hi, nan}, true, 0, FLT_MAX},
        // X narrows the interval first; Y then overflows. Both outputs must reset.
        {"overflow resets interval", {path(glm::vec3(-5, 0, 0), glm::vec3(1, 1e-40f, 0)).ray, lo, hi, 7}, true, 0, 7}
    };
    for (const auto& item : intervals) test("BVH interval: " + item.name, [&] {
        DeviceArray<IntervalQuery> input(std::vector<IntervalQuery>{item.query});
        DeviceArray<IntervalResult> output(1);
        queryIntervals<<<1, 1>>>(input.data, output.data, 1);
        cudaCheck(cudaGetLastError());
        const auto result = output.read().front();
        require(bool(result.hit) == item.hit, "Unexpected overlap result");
        require(std::isfinite(result.enter) && std::isfinite(result.exit), "Undefined interval output");
        if (item.hit) {
            require(result.enter >= 0 && result.enter <= item.enter
                && result.exit >= item.exit, "Interval is not conservative");
            near(result.enter, item.enter, std::max(1e-5f, std::abs(item.enter) * 1e-5f));
            near(result.exit, item.exit, std::max(1e-5f, std::abs(item.exit) * 1e-5f));
        }
    });

    test("BVH actual renderer upload: all node fields, absolute indices, mesh metadata and ABI", [&] {
        Scene scene(sampleScene.string());
        const auto original = scene.triangles;
        const auto first = *std::find_if(scene.geoms.begin(), scene.geoms.end(),
            [](const Geom& g) { return g.type == MESH; });
        Geom second = first;
        second.triangleStart += int(original.size());
        scene.triangles.insert(scene.triangles.end(), original.begin(), original.end());
        scene.geoms = {first, second};
        scene.rebuildMeshBVHs();
        require(scene.geoms[1].bvhRoot > 0 && scene.geoms[1].bvhIndexStart > 0, "Missing offset fixture");
        scene.state.camera.resolution = glm::ivec2(7, 5); scene.state.image.resize(35);
        {
            RenderSession session(scene);
            const auto view = pathtraceBVHForTesting();
            require(view.nodeCount == int(scene.bvhNodes.size())
                && view.indexCount == int(scene.bvhTriangleIndices.size())
                && view.triangleCount == int(scene.triangles.size()), "Uploaded counts differ");
            DeviceArray<BVHNode> nodes(view.nodeCount);
            DeviceArray<int> indices(view.indexCount), metadata(scene.geoms.size() * 4), sizes(2);
            inspectBVHUpload<<<1, 128>>>(view, pathtraceGeomsForTesting(), int(scene.geoms.size()),
                nodes.data, indices.data, metadata.data, sizes.data);
            cudaCheck(cudaGetLastError());
            const auto actual = nodes.read();
            const auto ids = metadata.read();
            const auto layout = sizes.read();
            require(layout[0] == sizeof(BVHNode) && layout[1] == sizeof(Geom), "Host/device ABI mismatch");
            require(indices.read() == scene.bvhTriangleIndices, "Global triangle indices changed");
            for (size_t i = 0; i < actual.size(); ++i) {
                const auto& a = actual[i]; const auto& b = scene.bvhNodes[i];
                for (int axis = 0; axis < 3; ++axis) {
                    near(a.boundsMin[axis], b.boundsMin[axis], 0);
                    near(a.boundsMax[axis], b.boundsMax[axis], 0);
                }
                require(a.leftChild == b.leftChild && a.rightChild == b.rightChild
                    && a.firstIndex == b.firstIndex && a.indexCount == b.indexCount, "Node fields differ");
            }
            for (size_t i = 0; i < scene.geoms.size(); ++i) {
                const auto& g = scene.geoms[i];
                require(ids[i * 4] == g.bvhRoot && ids[i * 4 + 1] == g.bvhNodeCount
                    && ids[i * 4 + 2] == g.bvhIndexStart && ids[i * 4 + 3] == g.triangleStart,
                    "Mesh offsets differ on GPU");
            }
            std::cout << "BVH ABI: node bytes=" << layout[0] << " nodes=" << view.nodeCount
                << " indices=" << view.indexCount << '\n';
        }
        checkFreedBVH();
    });

    test("BVH empty arrays, repeated free and stale CPU tree rejection", [&] {
        Scene scene(sampleScene.string());
        scene.state.camera.resolution = glm::ivec2(7, 5); scene.state.image.resize(35);
        const auto saved = scene.geoms;
        scene.geoms = {geometry(SPHERE)}; scene.triangles.clear(); scene.rebuildMeshBVHs();
        scene.state.enableBVH = true;
        {
            RenderSession session(scene);
            const auto view = pathtraceBVHForTesting();
            require(!view.nodes && !view.triangleIndices && !view.nodeCount && !view.indexCount,
                "Empty scene allocated BVH buffers");
        }
        pathtraceFree(); checkFreedBVH();
        scene.geoms = saved;
        bool rejected = false;
        try { pathtraceInit(&scene); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "Stale CPU tree reached GPU upload");
        checkFreedBVH();
    });


    test("BVH real torus edge rays share exact triangle arithmetic", [&] {
        Scene scene((sampleScene.parent_path() / "mesh_aabb_torus_interior.json").string());
        BVHFixture fixture;
        fixture.mesh = scene.geoms.back();
        fixture.triangles = scene.triangles;
        fixture.nodes = scene.bvhNodes;
        fixture.indices = scene.bvhTriangleIndices;
        require(fixture.mesh.type == MESH && fixture.mesh.triangleCount == 4096, "Wrong edge-ray fixture");
        // Captured from a 256x256, 64-spp full-path differential audit. Separate
        // inlining used to change edge acceptance/tie resolution between paths.
        const auto result = probe(fixture, {
            path(glm::vec3(-0.714171588f, 5.037992f, -4.99489927f),
                 glm::vec3(0.189981207f, -0.0767139047f, 0.978786051f)),
            path(glm::vec3(-2.3014009f, 5.16536856f, -4.99489927f),
                 glm::vec3(0.551600873f, -0.0241520498f, 0.833758473f))});
        for (const auto& ray : result)
            require(ray.bvhT > 0 && ray.stats.fallbackCount == 0, "Invalid edge-ray result");
    });

    test("BVH hierarchy: 8193 rays, 1024 triangles, transforms and production switch", [&] {
        BVHFixture f;
        for (int y = 0; y < 32; ++y) for (int x = 0; x < 32; ++x) {
            auto t = triangle(float((x + y) % 3) * 0.25f);
            const glm::vec3 offset(float(x) * 3, float(y) * 3, 0);
            t.v0 += offset; t.v1 += offset; t.v2 += offset;
            f.triangles.push_back(t);
        }
        f.mesh = geometry(MESH, glm::vec3(3, -2, 9), glm::vec3(25, 47, 13), glm::vec3(-2, .5f, 4),
            7, 0, int(f.triangles.size()));
        f.build();
        std::mt19937 generator(0xB00);
        std::uniform_real_distribution<float> xy(-5, 98);
        std::vector<PathSegment> rays;
        for (int i = 0; i < 8193; ++i) {
            glm::vec3 origin(xy(generator), xy(generator), i % 3 == 0 ? -10.0f : 10.0f);
            glm::vec3 direction(0, 0, origin.z > 0 ? -1.0f : 1.0f);
            if (i % 7 == 0) {
                const auto& tri = f.triangles[i % f.triangles.size()];
                origin.x = tri.v0.x; origin.y = tri.v0.y;
            }
            if (i % 11 == 0) direction = glm::normalize(glm::vec3(1e-7f, .03f, direction.z));
            rays.push_back(path(multiplyMV(f.mesh.transform, glm::vec4(origin, 1)),
                glm::normalize(multiplyMV(f.mesh.transform, glm::vec4(direction, 0)))));
        }
        const auto results = probe(f, rays);
        unsigned long long nodeVisits = 0, triangleTests = 0, hits = 0;
        for (const auto& p : results) {
            require(p.stats.fallbackCount == 0, "Valid hierarchy fell back");
            nodeVisits += p.stats.nodeVisits; triangleTests += p.stats.triangleTests;
            hits += p.bvhT > 0;
        }
        const auto bruteTests = static_cast<unsigned long long>(rays.size()) * f.triangles.size();
        require(hits > 100 && nodeVisits > rays.size(), "Fixture did not exercise hierarchy");
        require(triangleTests < bruteTests / 10, "BVH did not reduce triangle work");
        const auto actual = intersect({f.mesh}, f.triangles, rays);
        for (size_t i = 0; i < actual.size(); ++i) near(actual[i].t, results[i].bruteT, 0);
        std::cout << "BVH WORK: rays=" << rays.size() << " triangles=" << f.triangles.size()
            << " hits=" << hits << " node_visits=" << nodeVisits
            << " triangle_tests=" << triangleTests << " brute_tests=" << bruteTests << '\n';
    });
    for (bool equal : {false, true}) test(equal
        ? "BVH equal-distance hits retain lowest original triangle ID"
        : "BVH overlapping children continue after the first leaf hit", [&] {
        const auto f = twoLeafFixture(equal);
        const auto p = probe(f, {path()}).front();
        near(p.bvhT, 5); nearVector(p.bvhNormal, glm::vec3(0, 0, 1), 0);
        require(p.stats.triangleTests == 2 && p.stats.fallbackCount == 0,
            "Did not test both overlapping children");
    });
    test("BVH near-first traversal prunes the deferred far leaf", [&] {
        auto f = twoLeafFixture(false);
        setNodeBounds(f.nodes[1], glm::vec3(-2, -2, -4.1f), glm::vec3(2, 2, -3.9f));
        validateMeshBVH(f.triangles, f.mesh, f.nodes, f.indices);
        const auto p = probe(f, {path()}).front();
        require(p.stats.triangleTests == 1 && p.stats.nodeVisits == 2 && p.stats.fallbackCount == 0,
            "Near-first pruning was not used");
    });
    test("BVH root leaf may contain more than four triangles at the depth limit", [&] {
        BVHFixture f;
        for (int i = 0; i < 19; ++i) f.triangles.push_back(triangle(-float(i)));
        f.mesh.triangleCount = int(f.triangles.size());
        BVHBuildOptions options; options.maxDepth = 0;
        f.build(options);
        require(f.nodes.size() == 1 && f.nodes[0].indexCount == 19, "Not a large leaf");
        const auto p = probe(f, {path()}).front();
        require(p.stats.triangleTests == 19 && p.stats.fallbackCount == 0, "Leaf was truncated");
    });
    BVHFixture deep;
    deep.triangles.assign(BVH_MAX_DEPTH + 1, triangle());
    deep.mesh.triangleCount = int(deep.triangles.size());
    deep.mesh.bvhRoot = 0; deep.mesh.bvhNodeCount = 2 * BVH_MAX_DEPTH + 1;
    deep.nodes.resize(deep.mesh.bvhNodeCount);
    for (auto& node : deep.nodes) setNodeBounds(node, glm::vec3(-2, -2, -.1f), glm::vec3(2, 2, .1f));
    for (int i = 0; i < BVH_MAX_DEPTH; ++i) {
        deep.nodes[i].leftChild = i + 1;
        deep.nodes[i].rightChild = BVH_MAX_DEPTH + 1 + i;
    }
    for (int i = 0; i <= BVH_MAX_DEPTH; ++i) {
        deep.nodes[BVH_MAX_DEPTH + i].firstIndex = i;
        deep.nodes[BVH_MAX_DEPTH + i].indexCount = 1;
        deep.indices.push_back(i);
    }
    test("BVH maximum validated depth fills all 33 stack entries", [&] {
        const auto stats = validateMeshBVH(deep.triangles, deep.mesh, deep.nodes, deep.indices);
        require(stats.maxDepth == BVH_MAX_DEPTH, "Incorrect deepest fixture");
        const auto p = probe(deep, {path()}).front();
        require(p.stats.maxStack == BVH_STACK_CAPACITY && p.stats.fallbackCount == 0
            && p.stats.triangleTests == deep.triangles.size(), "Maximum stack path failed");
        std::cout << "BVH STACK: validated_depth=" << stats.maxDepth << " high_water=" << p.stats.maxStack << '\n';
    });
    test("BVH stack overflow recomputes a complete brute-force result", [&] {
        const auto p = probe(deep, {path()}, BVH_STACK_CAPACITY - 1).front();
        require(p.stats.fallbackCount == 1, "Overflow did not fall back");
    });
    for (int fault = 1; fault <= 4; ++fault) test("BVH missing device view fallback " + std::to_string(fault), [&] {
        const auto f = twoLeafFixture(false);
        const auto p = probe(f, {path()}, BVH_STACK_CAPACITY, fault).front();
        require(p.stats.fallbackCount == 1, "Missing arrays silently produced a miss");
    });
    for (int fault = 0; fault < 6; ++fault) test("BVH invalid visited index fallback " + std::to_string(fault), [&] {
        auto f = twoLeafFixture(false);
        if (fault == 0) f.mesh.bvhRoot = -1;
        if (fault == 1) f.nodes[0].leftChild = int(f.nodes.size());
        if (fault == 2) f.nodes[1].firstIndex = int(f.indices.size());
        if (fault == 3) f.indices[0] = int(f.triangles.size());
        if (fault == 4) f.nodes[0].leftChild = 0;
        if (fault == 5) f.nodes[1].indexCount = -1;
        const auto p = probe(f, {path()}).front();
        require(p.stats.fallbackCount == 1, "Invalid index did not fall back");
    });
    test("BVH reset/toggle clears accumulation and reuses CPU tree", [&] {
        Scene scene(sampleScene.string());
        const auto originalNodes = scene.bvhNodes;
        const auto originalIndices = scene.bvhTriangleIndices;
        scene.state.camera.resolution = glm::ivec2(33, 31); scene.state.image.resize(33 * 31);
        std::vector<glm::vec3> reference;
        for (int cycle = 0; cycle < 6; ++cycle) {
            scene.state.enableBVH = cycle % 2 != 0;
            scene.state.enableMeshCulling = cycle % 3 == 0;
            const auto image = render(scene, 4);
            if (!cycle) reference = image;
            else for (size_t i = 0; i < image.size(); ++i) nearVector(image[i], reference[i], 0);
            checkFreedBVH();
            require(scene.bvhNodes.size() == originalNodes.size()
                && scene.bvhTriangleIndices == originalIndices, "Reset rebuilt/appended CPU arrays");
            for (size_t i = 0; i < originalNodes.size(); ++i) {
                const auto& a = scene.bvhNodes[i]; const auto& b = originalNodes[i];
                require(a.leftChild == b.leftChild && a.rightChild == b.rightChild
                    && a.firstIndex == b.firstIndex && a.indexCount == b.indexCount, "Reset changed tree");
            }
        }
    });

}
}
