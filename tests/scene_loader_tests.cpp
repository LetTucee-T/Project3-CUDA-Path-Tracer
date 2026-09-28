#include "scene.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(float actual, float expected, float tolerance = 1e-5f) {
    return std::abs(actual - expected) <= tolerance;
}

void writeText(const fs::path& path, const std::string& contents) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output << contents;
    require(bool(output), "Could not write fixture: " + path.string());
}

json meshObject(const std::string& filename) {
    return {{"TYPE", "mesh"}, {"FILE", filename}, {"MATERIAL", "white"},
            {"TRANS", {0, 0, 0}}, {"ROTAT", {0, 0, 0}}, {"SCALE", {1, 1, 1}}};
}

json sceneDescription(const std::vector<json>& objects) {
    return {
        {"Materials", {{"white", {{"TYPE", "Diffuse"}, {"RGB", {1, 1, 1}}}},
                       {"mirror", {{"TYPE", "Specular"}, {"RGB", {0.8, 0.8, 0.8}}}}}},
        {"Objects", objects},
        {"Camera", {{"RES", {16, 12}}, {"FOVY", 45}, {"ITERATIONS", 1},
                    {"DEPTH", 4}, {"FILE", "loader_test"}, {"EYE", {0, 2, 6}},
                    {"LOOKAT", {0, 0, 0}}, {"UP", {0, 1, 0}}}}
    };
}

fs::path writeScene(const fs::path& directory, const json& description) {
    const fs::path path = directory / "scene.json";
    writeText(path, description.dump(2));
    return path;
}

void expectFailure(const fs::path& path, const std::string& expected) {
    try {
        Scene scene(path.string());
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(expected) != std::string::npos,
            "Unexpected error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Invalid input was accepted: " + path.string());
}

bool sameNodes(const std::vector<BVHNode>& a, const std::vector<BVHNode>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            if (a[i].boundsMin[axis] != b[i].boundsMin[axis]
                || a[i].boundsMax[axis] != b[i].boundsMax[axis]) return false;
        }
        if (a[i].leftChild != b[i].leftChild || a[i].rightChild != b[i].rightChild
            || a[i].firstIndex != b[i].firstIndex || a[i].indexCount != b[i].indexCount) return false;
    }
    return true;
}

void checkSceneBVH(const Scene& scene) {
    size_t nodeOffset = 0, indexOffset = 0, leafCount = 0;
    int maxDepth = 0, maxLeaf = 0;
    for (const auto& geom : scene.geoms) {
        if (geom.type != MESH) {
            require(geom.bvhRoot == -1 && geom.bvhNodeCount == 0, "Primitive acquired a BVH");
            continue;
        }
        require(geom.bvhIndexStart == int(indexOffset), "Mesh index blocks are not contiguous");
        if (geom.triangleCount > 0) require(geom.bvhRoot == int(nodeOffset), "Mesh node blocks are not contiguous");
        const auto stats = validateMeshBVH(scene.triangles, geom, scene.bvhNodes, scene.bvhTriangleIndices);
        nodeOffset += stats.nodeCount;
        indexOffset += size_t(geom.triangleCount);
        leafCount += stats.leafCount;
        maxDepth = (std::max)(maxDepth, stats.maxDepth);
        maxLeaf = (std::max)(maxLeaf, stats.maxLeafTriangles);
    }
    require(nodeOffset == scene.bvhNodes.size() && indexOffset == scene.bvhTriangleIndices.size(),
        "Unreferenced scene BVH data");
    require(scene.bvhStats.nodeCount == nodeOffset && scene.bvhStats.leafCount == leafCount
        && scene.bvhStats.maxDepth == maxDepth && scene.bvhStats.maxLeafTriangles == maxLeaf,
        "Scene BVH statistics disagree");
}

void checkBoundsAndNormals(const Scene& scene) {
    checkSceneBVH(scene);
    for (const auto& geom : scene.geoms) {
        if (geom.type != MESH) continue;
        require(geom.triangleStart >= 0 && geom.triangleCount > 0, "Invalid mesh range");
        require(size_t(geom.triangleStart) + size_t(geom.triangleCount)
                    <= scene.triangles.size(), "Mesh range is out of bounds");
        for (int i = 0; i < geom.triangleCount; ++i) {
            const auto& triangle = scene.triangles[geom.triangleStart + i];
            require(near(glm::length(triangle.normal), 1.0f), "Face normal is not unit length");
            for (const auto& vertex : {triangle.v0, triangle.v1, triangle.v2}) {
                for (int axis = 0; axis < 3; ++axis) {
                    require(std::isfinite(geom.boundsMin[axis])
                                && std::isfinite(geom.boundsMax[axis]), "Non-finite bounds");
                    require(vertex[axis] >= geom.boundsMin[axis]
                                && vertex[axis] <= geom.boundsMax[axis], "Bounds exclude a vertex");
                }
            }
        }
    }
}

const std::string triangleObj = "v 0 0 0\nv 2 0 0\nv 0 3 0\nf 1 2 3\n";

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: scene_loader_tests SOURCE_ROOT SCRATCH_ROOT\n";
        return 1;
    }
    const fs::path source = fs::absolute(argv[1]);
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path scratch = fs::absolute(argv[2]) / std::to_string(stamp);
    fs::create_directories(scratch);
    int passed = 0;
    int failed = 0;
    auto test = [&](const std::string& name, const std::function<void(const fs::path&)>& body) {
        try {
            const auto directory = scratch / name;
            fs::create_directories(directory);
            body(directory);
            std::cout << "[PASS] " << name << '\n';
            ++passed;
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
            ++failed;
        }
    };

    test("existing_cornell", [&](const fs::path&) {
        Scene scene((source / "scenes/cornell.json").string());
        require(scene.geoms.size() == 7 && scene.materials.size() == 5, "Primitive scene changed");
        require(scene.triangles.empty(), "Primitive scene allocated mesh triangles");
        checkSceneBVH(scene);
        require(near(glm::length(scene.state.camera.right), 1.0f), "Camera right is uninitialized");
        require(near(glm::dot(scene.state.camera.view, scene.state.camera.right), 0.0f),
                "Camera axes are not perpendicular");
    });
    test("existing_sphere", [&](const fs::path&) {
        Scene scene((source / "scenes/sphere.json").string());
        require(scene.geoms.size() == 1 && scene.geoms[0].type == SPHERE, "Sphere type changed");
        require(scene.triangles.empty(), "Unexpected triangles");
        checkSceneBVH(scene);
    });
    test("single_triangle", [&](const fs::path& directory) {
        writeText(directory / "triangle.obj", triangleObj);
        Scene scene(writeScene(directory, sceneDescription({meshObject("triangle.obj")})).string());
        require(scene.triangles.size() == 1 && scene.geoms[0].triangleStart == 0, "Wrong triangle range");
        require(near(scene.triangles[0].v1.x, 2) && near(scene.triangles[0].v2.y, 3), "Wrong positions");
        require(near(scene.triangles[0].normal.z, 1), "Wrong winding or normal");
        require(near(scene.geoms[0].boundsMin.x, 0) && near(scene.geoms[0].boundsMax.y, 3), "Wrong bounds");
        require(scene.geoms[0].boundsMin.z < 0 && scene.geoms[0].boundsMax.z > 0, "Planar bounds not expanded");
        checkBoundsAndNormals(scene);
    });
    test("negative_separate_indices", [&](const fs::path& directory) {
        writeText(directory / "indexed.obj", "v 0 0 0\nv 2 0 0\nv 0 3 0\n"
            "vt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 -1\nf -3/3/1 -2/1/1 -1/2/1\n");
        Scene scene(writeScene(directory, sceneDescription({meshObject("indexed.obj")})).string());
        require(scene.triangles.size() == 1 && near(scene.triangles[0].v1.x, 2), "Negative indices failed");
        require(near(scene.triangles[0].normal.z, 1), "First version must use geometric face normals");
        checkBoundsAndNormals(scene);
    });
    test("groups_quad_and_triangle", [&](const fs::path& directory) {
        writeText(directory / "groups.obj", "v 0 0 0\nv 2 0 0\nv 2 2 0\nv 0 2 0\nv 0 0 1\n"
            "g base\nf 1 2 3 4\ng side\nf 1 2 5\n");
        Scene scene(writeScene(directory, sceneDescription({meshObject("groups.obj")})).string());
        require(scene.triangles.size() == 3, "A group was lost or a quad was not triangulated");
        checkBoundsAndNormals(scene);
    });
    for (bool reverse : {false, true}) {
        test(reverse ? "concave_clockwise" : "concave_counterclockwise", [&, reverse](const fs::path& directory) {
            writeText(directory / "concave.obj", "v 0 0 0\nv 2 0 0\nv 2 1 0\nv 1 1 0\nv 1 2 0\nv 0 2 0\n"
                + std::string(reverse ? "f 6 5 4 3 2 1\n" : "f 1 2 3 4 5 6\n"));
            Scene scene(writeScene(directory, sceneDescription({meshObject("concave.obj")})).string());
            double totalArea = 0;
            for (const auto& triangle : scene.triangles) {
                totalArea += glm::length(glm::cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0)) * 0.5;
                const auto center = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;
                require(!(center.x > 1 && center.y > 1), "Triangle fills the concave cutout");
                require(near(triangle.normal.z, reverse ? -1.0f : 1.0f), "Polygon winding changed");
            }
            require(std::abs(totalArea - 3.0) < 1e-5, "Triangulation changed polygon area");
            checkBoundsAndNormals(scene);
        });
    }
    for (bool reverse : {false, true}) {
        test(reverse ? "concave_quad_clockwise" : "concave_quad_counterclockwise", [&, reverse](const fs::path& directory) {
            writeText(directory / "quad.obj", "v 0 0 0\nv 10 0 0\nv 9 0.5 0\nv 10 1 0\n"
                + std::string(reverse ? "f 4 3 2 1\n" : "f 1 2 3 4\n"));
            Scene scene(writeScene(directory, sceneDescription({meshObject("quad.obj")})).string());
            double area = 0;
            for (const auto& triangle : scene.triangles) {
                area += glm::length(glm::cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0)) * 0.5;
                require(near(triangle.normal.z, reverse ? -1.0f : 1.0f), "Concave quad winding changed");
            }
            require(scene.triangles.size() == 2 && std::abs(area - 4.5) < 1e-5,
                    "Concave quad triangulation changed its area");
            checkBoundsAndNormals(scene);
        });
    }
    test("multiple_mesh_ranges_materials", [&](const fs::path& directory) {
        writeText(directory / "triangle.obj", triangleObj);
        auto second = meshObject("triangle.obj");
        second["MATERIAL"] = "mirror";
        second["TRANS"] = {10, 20, 30};
        second["SCALE"] = {2, 3, -4};
        Scene scene(writeScene(directory, sceneDescription({meshObject("triangle.obj"), second})).string());
        require(scene.triangles.size() == 2 && scene.geoms[1].triangleStart == 1, "Ranges overlap");
        require(scene.geoms[0].materialid != scene.geoms[1].materialid, "Materials were merged");
        require(near(scene.triangles[1].v1.x, 2), "Transform was baked into local vertices");
        const glm::vec3 world = glm::vec3(scene.geoms[1].transform * glm::vec4(scene.triangles[1].v1, 1));
        require(near(world.x, 14) && near(world.y, 20) && near(world.z, 30), "Transform was lost");
        const glm::vec3 restored = glm::vec3(scene.geoms[1].inverseTransform * glm::vec4(world, 1));
        require(near(restored.x, 2) && near(restored.y, 0), "Inverse transform is wrong");
        checkBoundsAndNormals(scene);
    });
    test("relative_path_different_cwd", [&](const fs::path& directory) {
        writeText(directory / "models with spaces/triangle.obj", triangleObj);
        const auto path = writeScene(directory, sceneDescription({meshObject("models with spaces/triangle.obj")}));
        const auto previous = fs::current_path();
        struct Restore { fs::path path; ~Restore() { fs::current_path(path); } } restore{previous};
        fs::current_path(source);
        Scene scene(path.string());
        require(scene.triangles.size() == 1, "OBJ path was resolved from the process working directory");
    });
    test("utf8_json_mesh_filename", [&](const fs::path& directory) {
        const std::string filename = u8"\u6a21\u578b.obj";
        writeText(directory / fs::u8path(filename), triangleObj);
        Scene scene(writeScene(directory, sceneDescription({meshObject(filename)})).string());
        require(scene.triangles.size() == 1, "UTF-8 mesh filename was interpreted in the local code page");
    });
    test("absolute_path_uppercase_extension", [&](const fs::path& directory) {
        const auto obj = directory / "triangle.OBJ";
        writeText(obj, triangleObj);
        Scene scene(writeScene(directory, sceneDescription({meshObject(obj.string())})).string());
        require(scene.triangles.size() == 1, "Absolute OBJ path failed");
    });
    test("degenerate_faces_excluded_from_bounds", [&](const fs::path& directory) {
        writeText(directory / "mixed.obj", triangleObj
            + "v 100 0 0\nv 101 0 0\nv 102 0 0\nf 4 5 6\n");
        Scene scene(writeScene(directory, sceneDescription({meshObject("mixed.obj")})).string());
        require(scene.triangles.size() == 1, "Degenerate face was retained");
        require(scene.geoms[0].boundsMax.x < 3, "Discarded face expanded the bounds");
    });
    test("tiny_valid_triangle", [&](const fs::path& directory) {
        writeText(directory / "tiny.obj", "v 0 0 0\nv 1e-20 0 0\nv 0 1e-20 0\nf 1 2 3\n");
        Scene scene(writeScene(directory, sceneDescription({meshObject("tiny.obj")})).string());
        require(scene.triangles.size() == 1 && near(scene.triangles[0].normal.z, 1), "Normal computation underflowed");
        require(scene.geoms[0].boundsMax.x < 2e-20f, "Fixed-size padding overwhelmed a tiny mesh");
        checkBoundsAndNormals(scene);
    });
    test("large_valid_triangle", [&](const fs::path& directory) {
        writeText(directory / "large.obj", "v 0 0 0\nv 1e20 0 0\nv 0 1e20 0\nf 1 2 3\n");
        Scene scene(writeScene(directory, sceneDescription({meshObject("large.obj")})).string());
        require(scene.triangles.size() == 1 && near(scene.triangles[0].normal.z, 1), "Normal computation overflowed");
        checkBoundsAndNormals(scene);
    });
    test("culling_flag_loading_only", [&](const fs::path& directory) {
        writeText(directory / "triangle.obj", triangleObj);
        auto description = sceneDescription({meshObject("triangle.obj")});
        Scene defaults(writeScene(directory, description).string());
        require(defaults.state.enableMeshCulling, "Wrong default culling flag");
        description["Camera"]["MESH_CULLING"] = false;
        Scene disabled(writeScene(directory, description).string());
        require(!disabled.state.enableMeshCulling && disabled.triangles.size() == defaults.triangles.size(),
                "Culling flag changed loading");
    });
    test("sample_cube", [&](const fs::path&) {
        Scene scene((source / "scenes/mesh_cpu_validation.json").string());
        require(scene.geoms.size() == 3 && scene.triangles.size() == 12, "Sample cube load failed");
        double area = 0;
        for (const auto& triangle : scene.triangles) {
            area += glm::length(glm::cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0)) * 0.5;
        }
        require(std::abs(area - 24) < 1e-5, "Cube area is wrong");
        require(scene.bvhStats.nodeCount == 7 && scene.bvhStats.leafCount == 4
            && scene.bvhStats.maxDepth == 2, "Unexpected cube BVH topology");
        checkBoundsAndNormals(scene);
    });

    test("bvh_real_torus_topology", [&](const fs::path&) {
        Scene scene((source / "scenes/mesh_aabb_torus_exterior.json").string());
        checkBoundsAndNormals(scene);
        require(scene.bvhStats.nodeCount == 2054 && scene.bvhStats.leafCount == 1028
            && scene.bvhStats.maxDepth == 10 && scene.bvhTriangleIndices.size() == 4108,
            "Unexpected torus plus light topology");
    });
    test("bvh_rebuild_is_deterministic_and_preserves_triangles", [&](const fs::path&) {
        Scene scene((source / "scenes/mesh_cpu_validation.json").string());
        const auto nodes = scene.bvhNodes;
        const auto indices = scene.bvhTriangleIndices;
        const auto triangles = scene.triangles;
        for (int iteration = 0; iteration < 3; ++iteration) {
            scene.rebuildMeshBVHs();
            checkSceneBVH(scene);
            require(sameNodes(nodes, scene.bvhNodes) && indices == scene.bvhTriangleIndices,
                "Rebuild changed structure or appended old nodes");
            require(triangles.size() == scene.triangles.size(), "Triangle count changed");
            for (size_t i = 0; i < triangles.size(); ++i) {
                for (int axis = 0; axis < 3; ++axis) {
                    require(triangles[i].v0[axis] == scene.triangles[i].v0[axis]
                        && triangles[i].v1[axis] == scene.triangles[i].v1[axis]
                        && triangles[i].v2[axis] == scene.triangles[i].v2[axis]
                        && triangles[i].normal[axis] == scene.triangles[i].normal[axis],
                        "BVH construction modified triangle data");
                }
            }
        }
    });
    test("bvh_local_space_independent_of_transform", [&](const fs::path& directory) {
        writeText(directory / "triangle.obj", triangleObj);
        auto transformed = meshObject("triangle.obj");
        transformed["TRANS"] = {10, 20, -30}; transformed["ROTAT"] = {27, 48, 15};
        transformed["SCALE"] = {-2, 0.5, 4};
        Scene scene(writeScene(directory, sceneDescription({meshObject("triangle.obj"), transformed})).string());
        checkBoundsAndNormals(scene);
        const auto& a = scene.bvhNodes[scene.geoms[0].bvhRoot];
        const auto& b = scene.bvhNodes[scene.geoms[1].bvhRoot];
        for (int axis = 0; axis < 3; ++axis) {
            require(a.boundsMin[axis] == b.boundsMin[axis] && a.boundsMax[axis] == b.boundsMax[axis],
                "Object transform was baked into the BVH");
        }
    });
    test("bvh_rebuild_after_replacing_geometry_and_clearing", [&](const fs::path&) {
        Scene scene((source / "scenes/mesh_cpu_validation.json").string());
        Geom mesh{}; mesh.type = MESH; mesh.triangleCount = 1;
        scene.geoms = {mesh}; scene.triangles = {scene.triangles.front()};
        scene.rebuildMeshBVHs();
        checkSceneBVH(scene);
        require(scene.bvhNodes.size() == 1 && scene.bvhTriangleIndices.size() == 1, "Stale tree after replacement");
        scene.triangles.clear(); scene.geoms.clear();
        scene.rebuildMeshBVHs();
        checkSceneBVH(scene);
        require(scene.bvhNodes.empty() && scene.bvhTriangleIndices.empty(), "Stale tree after clear");
    });
    test("bvh_failed_rebuild_does_not_commit_partial_arrays", [&](const fs::path& directory) {
        writeText(directory / "triangle.obj", triangleObj);
        Scene scene(writeScene(directory, sceneDescription({meshObject("triangle.obj"), meshObject("triangle.obj")})).string());
        const auto nodes = scene.bvhNodes;
        const auto indices = scene.bvhTriangleIndices;
        const auto geoms = scene.geoms;
        scene.geoms[1].triangleCount = 100;
        bool rejected = false;
        try { scene.rebuildMeshBVHs(); }
        catch (const std::exception& error) {
            rejected = std::string(error.what()).find("BVH mesh 1") != std::string::npos;
        }
        require(rejected, "Invalid second mesh was accepted");
        require(sameNodes(nodes, scene.bvhNodes) && indices == scene.bvhTriangleIndices,
            "Failed rebuild committed partial storage");
        for (size_t i = 0; i < geoms.size(); ++i) {
            require(geoms[i].bvhRoot == scene.geoms[i].bvhRoot
                && geoms[i].bvhNodeCount == scene.geoms[i].bvhNodeCount
                && geoms[i].bvhIndexStart == scene.geoms[i].bvhIndexStart,
                "Failed rebuild committed partial metadata");
        }
        scene.geoms[1].triangleCount = geoms[1].triangleCount;
        scene.rebuildMeshBVHs(); checkSceneBVH(scene);
    });
    test("bvh_culling_flag_does_not_change_construction", [&](const fs::path& directory) {
        writeText(directory / "triangle.obj", triangleObj);
        auto description = sceneDescription({meshObject("triangle.obj")});
        Scene enabled(writeScene(directory, description).string());
        description["Camera"]["MESH_CULLING"] = false;
        Scene disabled(writeScene(directory, description).string());
        require(sameNodes(enabled.bvhNodes, disabled.bvhNodes)
            && enabled.bvhTriangleIndices == disabled.bvhTriangleIndices,
            "Culling flag changed the CPU tree");
    });

    test("bvh_switch_defaults_off", [&](const fs::path& directory) {
        writeText(directory / "triangle.obj", triangleObj);
        Scene scene(writeScene(directory, sceneDescription({meshObject("triangle.obj")})).string());
        require(!scene.state.enableBVH, "BVH default must preserve previous rendering mode");
        checkSceneBVH(scene);
    });
    test("bvh_switch_independent_of_culling_and_cpu_build", [&](const fs::path& directory) {
        writeText(directory / "triangle.obj", triangleObj);
        auto description = sceneDescription({meshObject("triangle.obj")});
        description["Camera"]["BVH"] = false;
        Scene disabled(writeScene(directory, description).string());
        for (bool culling : {false, true}) {
            description["Camera"]["BVH"] = true;
            description["Camera"]["MESH_CULLING"] = culling;
            Scene enabled(writeScene(directory, description).string());
            require(enabled.state.enableBVH && enabled.state.enableMeshCulling == culling,
                "JSON flags not read independently");
            require(sameNodes(enabled.bvhNodes, disabled.bvhNodes)
                && enabled.bvhTriangleIndices == disabled.bvhTriangleIndices,
                "BVH switch changed tree construction");
        }
    });
    test("bvh_switch_rejects_non_boolean", [&](const fs::path& directory) {
        writeText(directory / "triangle.obj", triangleObj);
        auto description = sceneDescription({meshObject("triangle.obj")});
        description["Camera"]["BVH"] = "true";
        expectFailure(writeScene(directory, description), "type_error");
    });

    test("dof_defaults_preserve_pinhole", [&](const fs::path& directory) {
        Scene scene(writeScene(directory, sceneDescription({})).string());
        require(!scene.state.enableDepthOfField && scene.state.camera.lensRadius == 0.0f,
            "Legacy scene enabled a finite lens");
        require(near(scene.state.camera.focalDistance, std::sqrt(40.0f)), "Default focus is not EYE-to-LOOKAT distance");
    });
    test("dof_explicit_values_and_zero_aperture", [&](const fs::path& directory) {
        auto description = sceneDescription({});
        description["Camera"]["DEPTH_OF_FIELD"] = true;
        description["Camera"]["FOCAL_DISTANCE"] = 3.25;
        for (float radius : {0.0f, 0.08f}) {
            description["Camera"]["LENS_RADIUS"] = radius;
            Scene scene(writeScene(directory, description).string());
            require(scene.state.enableDepthOfField && near(scene.state.camera.lensRadius, radius)
                && near(scene.state.camera.focalDistance, 3.25f), "Explicit DOF parameters changed");
        }
    });
    test("dof_parameters_do_not_enable_switch", [&](const fs::path& directory) {
        auto description = sceneDescription({});
        description["Camera"]["LENS_RADIUS"] = 0.2;
        description["Camera"]["FOCAL_DISTANCE"] = 9;
        Scene scene(writeScene(directory, description).string());
        require(!scene.state.enableDepthOfField && near(scene.state.camera.lensRadius, 0.2f)
            && near(scene.state.camera.focalDistance, 9.0f), "DOF flag coupled to parameter values");
    });
    test("dof_up_validation_preserves_legacy_axes", [&](const fs::path& directory) {
        auto description = sceneDescription({});
        description["Camera"]["UP"] = {0, 7, 0};
        Scene scene(writeScene(directory, description).string());
        require(scene.state.camera.up.y == 7.0f, "Lens validation changed pinhole UP");
        require(near(glm::length(scene.state.camera.right), 1.0f), "Legacy right direction changed");
    });
    struct InvalidCamera { const char* name; const char* key; json value; const char* error; };
    const std::vector<InvalidCamera> invalidCameras = {
        {"dof_string_flag", "DEPTH_OF_FIELD", "true", "must be boolean"},
        {"dof_numeric_flag", "DEPTH_OF_FIELD", 1, "must be boolean"},
        {"dof_null_flag", "DEPTH_OF_FIELD", nullptr, "must be boolean"},
        {"dof_negative_radius", "LENS_RADIUS", -0.1, "LENS_RADIUS must be >= 0"},
        {"dof_huge_radius", "LENS_RADIUS", 1e100, "LENS_RADIUS must be a finite float"},
        {"dof_string_radius", "LENS_RADIUS", "0.1", "LENS_RADIUS must be a number"},
        {"dof_boolean_radius", "LENS_RADIUS", true, "LENS_RADIUS must be a number"},
        {"dof_null_radius", "LENS_RADIUS", nullptr, "LENS_RADIUS must be a number"},
        {"dof_zero_focus", "FOCAL_DISTANCE", 0, "FOCAL_DISTANCE must be > 0"},
        {"dof_negative_focus", "FOCAL_DISTANCE", -2, "FOCAL_DISTANCE must be > 0"},
        {"dof_huge_focus", "FOCAL_DISTANCE", 1e100, "FOCAL_DISTANCE must be a finite float"},
        {"dof_underflow_focus", "FOCAL_DISTANCE", 1e-100, "FOCAL_DISTANCE must be > 0"},
        {"dof_string_focus", "FOCAL_DISTANCE", "5", "FOCAL_DISTANCE must be a number"},
        {"dof_boolean_focus", "FOCAL_DISTANCE", false, "FOCAL_DISTANCE must be a number"},
        {"dof_null_focus", "FOCAL_DISTANCE", nullptr, "FOCAL_DISTANCE must be a number"},
        {"camera_coincident_eye_lookat", "LOOKAT", {0, 2, 6}, "finite non-zero distance"},
        {"camera_zero_up", "UP", {0, 0, 0}, "UP must be non-zero"},
        {"camera_parallel_up", "UP", {0, -2, -6}, "not parallel"},
        {"camera_short_eye", "EYE", {0, 2}, "EYE must contain three numbers"},
        {"camera_boolean_eye", "EYE", {0, true, 6}, "EYE must be a number"},
        {"camera_overflow_eye", "EYE", {1e100, 2, 6}, "EYE must be a finite float"},
        {"camera_overflow_lookat", "LOOKAT", {0, 1e100, 0}, "LOOKAT must be a finite float"},
        {"camera_overflow_up", "UP", {0, 1e100, 0}, "UP must be a finite float"}
    };
    for (const auto& fixture : invalidCameras) {
        test(fixture.name, [&, fixture](const fs::path& directory) {
            auto description = sceneDescription({});
            description["Camera"][fixture.key] = fixture.value;
            expectFailure(writeScene(directory, description), fixture.error);
        });
    }

    const std::vector<std::pair<std::string, std::string>> invalidObjects = {
        {"empty_mesh", "# No geometry\n"},
        {"all_degenerate", "v 0 0 0\nv 1 0 0\nv 2 0 0\nf 1 2 3\n"},
        {"invalid_triangle_index", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 99\n"},
        {"invalid_quad_index", triangleObj + "f 1 2 3 99\n"},
        {"invalid_negative_index", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -4 -2 -1\n"},
        {"zero_index", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 2 3\n"},
        {"nonfinite_coordinate", "v 1e400 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"},
        {"nan_coordinate", "v nan 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"},
        {"malformed_coordinate", "v junk 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"},
        {"missing_coordinate", "v 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"}
    };
    for (const auto& fixture : invalidObjects) {
        test(fixture.first, [&, fixture](const fs::path& directory) {
            writeText(directory / "invalid.obj", fixture.second);
            expectFailure(writeScene(directory, sceneDescription({meshObject("invalid.obj")})), "OBJ");
        });
    }
    test("missing_obj", [&](const fs::path& directory) {
        expectFailure(writeScene(directory, sceneDescription({meshObject("missing.obj")})), "missing.obj");
    });
    test("unsupported_mesh_extension", [&](const fs::path& directory) {
        expectFailure(writeScene(directory, sceneDescription({meshObject("mesh.gltf")})), ".obj extension");
    });
    test("empty_mesh_filename", [&](const fs::path& directory) {
        expectFailure(writeScene(directory, sceneDescription({meshObject("")})), "FILE must not be empty");
    });
    test("unknown_geometry", [&](const fs::path& directory) {
        auto object = meshObject("unused.obj");
        object["TYPE"] = "meshh";
        expectFailure(writeScene(directory, sceneDescription({object})), "Unknown geometry type");
    });
    test("unknown_material", [&](const fs::path& directory) {
        auto object = meshObject("unused.obj");
        object["MATERIAL"] = "missing";
        expectFailure(writeScene(directory, sceneDescription({object})), "Unknown material");
    });
    test("zero_scale", [&](const fs::path& directory) {
        auto object = meshObject("unused.obj");
        object["SCALE"] = {1, 0, 1};
        expectFailure(writeScene(directory, sceneDescription({object})), "non-zero");
    });
    test("nonfinite_transform", [&](const fs::path& directory) {
        auto object = meshObject("unused.obj");
        object["TRANS"] = {1e100, 0, 0};
        expectFailure(writeScene(directory, sceneDescription({object})), "finite");
    });
    test("malformed_transform", [&](const fs::path& directory) {
        auto object = meshObject("unused.obj");
        object["SCALE"] = {1, 2};
        expectFailure(writeScene(directory, sceneDescription({object})), "three numbers");
    });
    test("missing_scene", [&](const fs::path& directory) {
        expectFailure(directory / "missing.json", "Unable to open scene file");
    });
    test("unsupported_scene_extension", [&](const fs::path& directory) {
        expectFailure(directory / "scene", ".json extension");
    });
    test("malformed_json", [&](const fs::path& directory) {
        writeText(directory / "broken.json", "{broken");
        expectFailure(directory / "broken.json", "parse_error");
    });

    std::cout << "RESULT: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}
