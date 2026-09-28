#include "bvh.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void require(bool ok, const std::string& message)
{
    if (!ok) throw std::runtime_error(message);
}

void expectFailure(const std::function<void()>& fn, const std::string& message)
{
    try { fn(); }
    catch (const std::exception& error) {
        require(std::string(error.what()).find(message) != std::string::npos,
            "Unexpected error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected failure containing: " + message);
}

bool sameVector(const glm::vec3& a, const glm::vec3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool sameVector(const float (&a)[3], const float (&b)[3])
{
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

bool sameNodes(const std::vector<BVHNode>& a, const std::vector<BVHNode>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!sameVector(a[i].boundsMin, b[i].boundsMin)
            || !sameVector(a[i].boundsMax, b[i].boundsMax)
            || a[i].leftChild != b[i].leftChild || a[i].rightChild != b[i].rightChild
            || a[i].firstIndex != b[i].firstIndex || a[i].indexCount != b[i].indexCount) return false;
    }
    return true;
}

std::vector<Triangle> makeTriangles(int count)
{
    std::vector<Triangle> triangles;
    for (int i = 0; i < count; ++i) {
        const glm::vec3 p(float((i * 37) % 101), float(i / 101), float(i % 7));
        triangles.push_back({p, p + glm::vec3(0.5f, 0, 0),
            p + glm::vec3(0, 0.5f, 0), glm::vec3(0, 0, 1)});
    }
    return triangles;
}

struct Fixture
{
    std::vector<Triangle> triangles;
    std::vector<BVHNode> nodes;
    std::vector<int> indices;
    Geom mesh{};
    BVHBuildResult built;

    explicit Fixture(std::vector<Triangle> data, BVHBuildOptions options = {})
        : triangles(std::move(data))
    {
        mesh.type = MESH;
        mesh.triangleCount = static_cast<int>(triangles.size());
        built = buildMeshBVH(triangles, 0, mesh.triangleCount, nodes, indices, options);
        mesh.bvhRoot = built.root;
        mesh.bvhNodeCount = built.nodeCount;
        mesh.bvhIndexStart = built.indexStart;
    }

    BVHStats validate(int maxDepth = BVH_MAX_DEPTH) const
    {
        return validateMeshBVH(triangles, mesh, nodes, indices, maxDepth);
    }
};
} // namespace

int main()
{
    int passed = 0, failed = 0;
    auto test = [&](const std::string& name, const std::function<void()>& fn) {
        try { fn(); ++passed; std::cout << "[PASS] " << name << '\n'; }
        catch (const std::exception& error) {
            ++failed; std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        }
    };

    for (int count : {0, 1, 4, 5, 7, 12, 13, 257, 4096}) {
        test("coverage and leaf threshold: " + std::to_string(count), [=] {
            const auto original = makeTriangles(count);
            Fixture f(original);
            const auto stats = f.validate();
            require(stats.nodeCount == f.nodes.size() && stats.nodeCount == f.built.stats.nodeCount,
                "Node statistics disagree");
            require(stats.leafCount == f.built.stats.leafCount
                && stats.maxDepth == f.built.stats.maxDepth, "Structure statistics disagree");
            require(stats.maxLeafTriangles <= 4, "Default leaf size exceeded");
            auto sorted = f.indices;
            std::sort(sorted.begin(), sorted.end());
            require(sorted.size() == std::size_t(count), "Wrong reference count");
            for (int i = 0; i < count; ++i) {
                require(sorted[i] == i, "Missing or duplicated source index");
                require(sameVector(original[i].v0, f.triangles[i].v0)
                    && sameVector(original[i].v1, f.triangles[i].v1)
                    && sameVector(original[i].v2, f.triangles[i].v2)
                    && sameVector(original[i].normal, f.triangles[i].normal), "Input triangles changed");
            }
            if (count == 0) require(f.mesh.bvhRoot == -1 && f.nodes.empty(), "Empty tree is not empty");
            if (count > 0 && count <= 4) require(stats.nodeCount == 1 && stats.maxDepth == 0, "Expected one leaf");
            if (count == 12) require(stats.nodeCount == 7 && stats.leafCount == 4 && stats.maxDepth == 2, "Wrong 12-triangle tree");
            if (count == 4096) require(stats.nodeCount == 2047 && stats.leafCount == 1024 && stats.maxDepth == 10, "Wrong 4096-triangle tree");
        });
    }
    test("coincident centroids terminate deterministically", [&] {
        std::vector<Triangle> triangles;
        for (int i = 1; i <= 65; ++i) {
            const float s = float(i);
            triangles.push_back({{s, 0, 0}, {0, s, 0}, {-s, -s, 0}, {0, 0, 1}});
        }
        Fixture first(triangles), second(triangles);
        require(first.validate().maxLeafTriangles <= 4, "Coincident centroid leaf is oversized");
        require(first.indices == second.indices && sameNodes(first.nodes, second.nodes), "Rebuild differs");
    });
    test("overlapping duplicate geometry retains distinct primitive IDs", [&] {
        const auto t = makeTriangles(1)[0];
        Fixture f(std::vector<Triangle>(33, t));
        f.validate();
        require(f.indices.size() == 33 && f.built.stats.maxDepth == 4, "Duplicates lost or bad depth");
    });
    test("depth cap permits large leaves without losing triangles", [&] {
        Fixture f(makeTriangles(257), {4, 1});
        const auto stats = f.validate(1);
        require(stats.maxDepth == 1 && stats.nodeCount == 3 && stats.maxLeafTriangles == 129,
            "Depth cap did not terminate splitting");
    });
    test("zero depth creates a single leaf", [&] {
        Fixture f(makeTriangles(13), {4, 0});
        require(f.validate(0).nodeCount == 1 && f.nodes[0].indexCount == 13, "Bad root leaf");
    });
    test("leaf size is configurable", [&] {
        Fixture f(makeTriangles(13), {1, BVH_MAX_DEPTH});
        require(f.validate().nodeCount == 25 && f.built.stats.leafCount == 13, "Bad single-primitive leaves");
    });
    for (float scale : {1e-20f, 1e20f, std::numeric_limits<float>::denorm_min()}) {
        test("finite conservative bounds at scale " + std::to_string(scale), [=] {
            Fixture f({Triangle{{0, 0, 0}, {scale, 0, 0}, {0, scale, 0}, {0, 0, 1}}});
            f.validate();
            require(f.nodes[0].boundsMin[2] < 0 && f.nodes[0].boundsMax[2] > 0, "Planar bounds not widened");
            require(f.nodes[0].boundsMax[0] >= scale && f.nodes[0].boundsMax[0] < scale * 4.0f,
                "Padding excludes geometry or overwhelms a tiny model");
        });
    }
    test("centroid sum beyond FLT_MAX remains finite", [&] {
        const float big = (std::numeric_limits<float>::max)() * 0.75f;
        const float other = std::nextafter(big, 0.0f);
        std::vector<Triangle> ts(9, Triangle{{big, 0, 0}, {other, 1, 0}, {big, 0, 1}, {1, 0, 0}});
        Fixture f(ts); f.validate();
    });
    test("padding near both float limits remains finite", [&] {
        const float m = (std::numeric_limits<float>::max)();
        Fixture f({Triangle{{m, 0, 0}, {m, 1, 0}, {m, 0, 1}, {1, 0, 0}},
            Triangle{{-m, 0, 0}, {-m, 1, 0}, {-m, 0, 1}, {1, 0, 0}}});
        f.validate();
    });
    test("multiple meshes use global offsets and preserve earlier blocks", [&] {
        const auto ts = makeTriangles(25);
        std::vector<BVHNode> nodes;
        std::vector<int> indices;
        const auto a = buildMeshBVH(ts, 2, 7, nodes, indices);
        const auto savedNodes = nodes;
        const auto savedIndices = indices;
        const auto b = buildMeshBVH(ts, 13, 12, nodes, indices);
        require(b.root == int(savedNodes.size()) && b.indexStart == int(savedIndices.size()), "Wrong append offsets");
        require(sameNodes(savedNodes, std::vector<BVHNode>(nodes.begin(), nodes.begin() + savedNodes.size())), "Earlier nodes changed");
        require(std::equal(savedIndices.begin(), savedIndices.end(), indices.begin()), "Earlier references changed");
        for (bool second : {false, true}) {
            const auto& r = second ? b : a;
            Geom g{}; g.type = MESH; g.triangleStart = second ? 13 : 2; g.triangleCount = second ? 12 : 7;
            g.bvhRoot = r.root; g.bvhNodeCount = r.nodeCount; g.bvhIndexStart = r.indexStart;
            validateMeshBVH(ts, g, nodes, indices);
        }
        const auto empty = buildMeshBVH(ts, 25, 0, nodes, indices);
        require(empty.root == -1 && empty.nodeCount == 0 && empty.indexStart == int(indices.size()), "Empty append changed data");
    });
    test("same input gives the same flat tree on repeated builds", [&] {
        const auto ts = makeTriangles(1003);
        Fixture a(ts);
        for (int i = 0; i < 4; ++i) {
            Fixture b(ts); b.validate();
            require(a.indices == b.indices && sameNodes(a.nodes, b.nodes), "Nondeterministic tree");
        }
    });

    for (auto range : {std::pair<int, int>{-1, 1}, {0, -1}, {9, 1}, {1, 9},
                       {(std::numeric_limits<int>::max)(), (std::numeric_limits<int>::max)()}}) {
        test("reject invalid source range " + std::to_string(range.first) + "," + std::to_string(range.second), [=] {
            std::vector<BVHNode> nodes; std::vector<int> ids;
            expectFailure([&] { buildMeshBVH(makeTriangles(8), range.first, range.second, nodes, ids); }, "range");
            require(nodes.empty() && ids.empty(), "Failed build appended data");
        });
    }
    for (auto options : {BVHBuildOptions{0, 32}, {-1, 32}, {4, -1}, {4, BVH_MAX_DEPTH + 1}}) {
        test("reject invalid options " + std::to_string(options.maxLeafTriangles) + "," + std::to_string(options.maxDepth), [=] {
            expectFailure([&] { Fixture f(makeTriangles(5), options); }, "invalid leaf size or depth");
        });
    }
    for (float bad : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        test(std::isnan(bad) ? "reject NaN without corrupting prior mesh" : "reject infinity without corrupting prior mesh", [=] {
            Fixture f(makeTriangles(13));
            const auto nodes = f.nodes; const auto ids = f.indices;
            auto ts = makeTriangles(5); ts.back().v1.x = bad;
            expectFailure([&] { buildMeshBVH(ts, 0, 5, f.nodes, f.indices); }, "non-finite");
            require(sameNodes(nodes, f.nodes) && ids == f.indices, "Previous blocks changed on failure");
        });
    }

    // Mutation tests prove that validation rejects corrupt data, not just that
    // a builder and validator agree on well-formed trees.
    using Mutator = std::function<void(Fixture&)>;
    const std::vector<std::pair<std::string, Mutator>> corruptions = {
        {"negative root", [](Fixture& f) { f.mesh.bvhRoot = -1; }},
        {"root beyond storage", [](Fixture& f) { f.mesh.bvhRoot = int(f.nodes.size()); }},
        {"invalid mesh triangle range", [](Fixture& f) { f.mesh.triangleCount += 1; }},
        {"self cycle", [](Fixture& f) { f.nodes[0].leftChild = 0; }},
        {"shared child", [](Fixture& f) { f.nodes[0].rightChild = f.nodes[0].leftChild; }},
        {"out of range child", [](Fixture& f) { f.nodes[0].leftChild = (std::numeric_limits<int>::max)(); }},
        {"negative child", [](Fixture& f) { f.nodes[0].leftChild = -1; }},
        {"unreachable node", [](Fixture& f) { f.nodes.emplace_back(); ++f.mesh.bvhNodeCount; }},
        {"invalid bounds", [](Fixture& f) { f.nodes[0].boundsMin[0] = f.nodes[0].boundsMax[0] + 1; }},
        {"NaN bounds", [](Fixture& f) { f.nodes[0].boundsMin[0] = std::numeric_limits<float>::quiet_NaN(); }},
        {"parent excludes child", [](Fixture& f) { f.nodes[0].boundsMax[0] = f.nodes[0].boundsMin[0] + 0.01f; }},
        {"internal leaf data", [](Fixture& f) { f.nodes[0].firstIndex = 0; }},
        {"negative leaf count", [](Fixture& f) { f.nodes.back().indexCount = -1; }},
        {"leaf with children", [](Fixture& f) { f.nodes.back().leftChild = 0; }},
        {"leaf beyond index block", [](Fixture& f) { f.nodes.back().firstIndex = int(f.indices.size()); }},
        {"negative leaf offset", [](Fixture& f) { f.nodes.back().firstIndex = -1; }},
        {"missing index slot", [](Fixture& f) { --f.nodes.back().indexCount; }},
        {"overlapping index slots", [](Fixture& f) { f.nodes.back().firstIndex = 0; }},
        {"duplicate triangle", [](Fixture& f) { f.indices.back() = f.indices.front(); }},
        {"negative triangle", [](Fixture& f) { f.indices.back() = -1; }},
        {"triangle beyond mesh", [](Fixture& f) { f.indices.back() = int(f.triangles.size()); }},
        {"leaf excludes vertex", [](Fixture& f) { const int id = f.indices.back(); f.triangles[id].v1.z += 1000; }},
        {"nonfinite vertex", [](Fixture& f) { f.triangles[0].v0.x = std::numeric_limits<float>::infinity(); }},
        {"wrong geometry kind", [](Fixture& f) { f.mesh.type = SPHERE; }}
    };
    for (const auto& [name, mutate] : corruptions) {
        test("validator rejects " + name, [&] {
            Fixture f(makeTriangles(13)); mutate(f);
            expectFailure([&] { f.validate(); }, "BVH validation");
        });
    }
    test("validator rejects a tree beyond traversal depth budget", [&] {
        Fixture f(makeTriangles(13));
        expectFailure([&] { f.validate(1); }, "depth limit");
    });
    test("validator rejects cross-mesh child and triangle references", [&] {
        const auto ts = makeTriangles(26);
        std::vector<BVHNode> nodes; std::vector<int> ids;
        const auto a = buildMeshBVH(ts, 0, 13, nodes, ids);
        const auto b = buildMeshBVH(ts, 13, 13, nodes, ids);
        Geom g{}; g.type = MESH; g.triangleCount = 13;
        g.bvhRoot = a.root; g.bvhNodeCount = a.nodeCount; g.bvhIndexStart = a.indexStart;
        const int oldChild = nodes[a.root].leftChild;
        nodes[a.root].leftChild = b.root;
        expectFailure([&] { validateMeshBVH(ts, g, nodes, ids); }, "outside this mesh");
        nodes[a.root].leftChild = oldChild;
        ids[a.indexStart] = 13;
        expectFailure([&] { validateMeshBVH(ts, g, nodes, ids); }, "outside this mesh");
    });
    std::cout << "RESULT: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}
