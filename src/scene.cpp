#include "scene.h"
#include "cameraSampling.h"
#include "appearance.h"
#include <stb_image.h>
#include <memory>

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include "objLoader.h"

#include <filesystem>
#include <cfloat>
#include <limits>
#include <cmath>
#include <cctype>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;


namespace {

glm::vec3 readGeometryVector(const json& object, const char* key) {
    const auto& values = object.at(key);
    if (!values.is_array() || values.size() != 3) {
        throw std::runtime_error(std::string(key) + " must contain three numbers");
    }
    glm::vec3 result;
    for (int axis = 0; axis < 3; ++axis) {
        result[axis] = values.at(axis).get<float>();
        if (!std::isfinite(result[axis])) {
            throw std::runtime_error(std::string(key) + " must be finite");
        }
    }
    return result;
}

float readCameraNumber(const json& value, const std::string& key) {
    if (!value.is_number()) {
        throw std::runtime_error("Camera." + key + " must be a number");
    }
    const double number = value.get<double>();
    if (!std::isfinite(number) || std::abs(number) > FLT_MAX) {
        throw std::runtime_error("Camera." + key + " must be a finite float");
    }
    return static_cast<float>(number);
}

glm::vec3 readCameraVector(const json& camera, const char* key) {
    const auto& values = camera.at(key);
    if (!values.is_array() || values.size() != 3) {
        throw std::runtime_error(std::string("Camera.") + key + " must contain three numbers");
    }
    glm::vec3 result;
    for (int axis = 0; axis < 3; ++axis) {
        result[axis] = readCameraNumber(values.at(axis), key);
    }
    return result;
}

bool finiteMatrix(const glm::mat4& matrix) {
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            if (!std::isfinite(matrix[column][row])) return false;
        }
    }
    return true;
}

} // namespace

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    std::string extension = std::filesystem::path(filename).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (extension != ".json") {
        throw std::runtime_error("Scene file must use the .json extension: " + filename);
    }
    loadFromJSON(filename);
    rebuildMeshBVHs();
}

void Scene::rebuildMeshBVHs(const BVHBuildOptions& options)
{
    // Build into temporary storage so errors cannot leave half-updated BVH ranges.
    std::vector<BVHNode> newNodes;
    std::vector<int> newIndices;
    std::vector<Geom> newGeoms = geoms;
    BVHStats totals;
    std::ostringstream report;
    for (size_t i = 0; i < newGeoms.size(); ++i) {
        Geom& geom = newGeoms[i];
        geom.bvhRoot = -1;
        geom.bvhNodeCount = 0;
        geom.bvhIndexStart = 0;
        if (geom.type != MESH) continue;
        try {
            const auto built = buildMeshBVH(triangles, geom.triangleStart, geom.triangleCount,
                newNodes, newIndices, options);
            geom.bvhRoot = built.root;
            geom.bvhNodeCount = built.nodeCount;
            geom.bvhIndexStart = built.indexStart;
            const auto checked = validateMeshBVH(triangles, geom, newNodes, newIndices, options.maxDepth);
            if (checked.nodeCount != built.stats.nodeCount || checked.leafCount != built.stats.leafCount
                || checked.maxDepth != built.stats.maxDepth
                || checked.maxLeafTriangles != built.stats.maxLeafTriangles) {
                throw std::runtime_error("builder and validator statistics disagree");
            }
            totals.nodeCount += checked.nodeCount;
            totals.leafCount += checked.leafCount;
            totals.maxDepth = (std::max)(totals.maxDepth, checked.maxDepth);
            totals.maxLeafTriangles = (std::max)(totals.maxLeafTriangles, checked.maxLeafTriangles);
            totals.buildMilliseconds += built.stats.buildMilliseconds;
            totals.validationMilliseconds += checked.validationMilliseconds;
            report << "BVH mesh " << i << ": root=" << geom.bvhRoot
                << ", nodes=" << checked.nodeCount << ", leaves=" << checked.leafCount
                << ", maxDepth=" << checked.maxDepth << ", indexStart=" << geom.bvhIndexStart
                << ", build_ms=" << built.stats.buildMilliseconds
                << ", validate_ms=" << checked.validationMilliseconds << '\n';
        } catch (const std::exception& error) {
            throw std::runtime_error("BVH mesh " + std::to_string(i) + ": " + error.what());
        }
    }
    bvhNodes.swap(newNodes);
    bvhTriangleIndices.swap(newIndices);
    // Keep the geometry vector's storage stable for callers holding Geom references.
    for (size_t i = 0; i < geoms.size(); ++i) {
        geoms[i].bvhRoot = newGeoms[i].bvhRoot;
        geoms[i].bvhNodeCount = newGeoms[i].bvhNodeCount;
        geoms[i].bvhIndexStart = newGeoms[i].bvhIndexStart;
    }
    bvhStats = totals;
    cout << report.str();
}

void Scene::loadObjMesh(const std::string& filename, Geom& geom)
{
    std::string extension = std::filesystem::u8path(filename).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (extension != ".obj") {
        throw std::runtime_error("Mesh file must use the .obj extension: " + filename);
    }
    const obj::Mesh source = obj::read(filename);
    std::vector<Triangle> loadedTriangles;
    std::vector<TriangleSurface> loadedSurfaces;
    glm::vec3 boundsMin(FLT_MAX);
    glm::vec3 boundsMax(-FLT_MAX);
    size_t skippedDegenerate = 0;
    const size_t maximumIndex = static_cast<size_t>((std::numeric_limits<int>::max)());

    auto appendTriangle = [&](const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                              obj::Index i0, obj::Index i1, obj::Index i2) {
        Triangle triangle{v0, v1, v2, glm::vec3(0.0f)};
        // Double precision prevents normal computation from overflowing or
        // underflowing for otherwise representable float positions.
        const glm::dvec3 edge1 = glm::dvec3(v1) - glm::dvec3(v0);
        const glm::dvec3 edge2 = glm::dvec3(v2) - glm::dvec3(v0);
        const glm::dvec3 faceNormal = glm::cross(edge1, edge2);
        const double normalLength2 = glm::dot(faceNormal, faceNormal);
        if (normalLength2 == 0.0) {
            ++skippedDegenerate;
            return;
        }
        triangle.normal = glm::vec3(faceNormal / std::sqrt(normalLength2));
        if (triangles.size() > maximumIndex
            || loadedTriangles.size() >= maximumIndex - triangles.size()) {
            throw std::runtime_error("OBJ triangle count exceeds int indexing: " + filename);
        }
        TriangleSurface surface{};
        const obj::Index indices[3] = {i0, i1, i2};
        for (int k = 0; k < 3; ++k) {
            surface.normals[k] = triangle.normal;
            if (geom.smoothNormals && indices[k].normal >= 0) {
                const size_t index = size_t(indices[k].normal);
                if (index >= source.normals.size()) throw std::runtime_error("Invalid OBJ normal index: " + filename);
                glm::vec3 n(source.normals[index][0], source.normals[index][1], source.normals[index][2]);
                if (!std::isfinite(n.x) || !std::isfinite(n.y) || !std::isfinite(n.z)
                    || !std::isfinite(glm::dot(n,n)) || glm::dot(n,n) <= 0)
                    throw std::runtime_error("Invalid OBJ normal: " + filename);
                surface.normals[k] = glm::normalize(n);
            }
            if (geom.textured) {
                const int index = indices[k].uv;
                if (index < 0 || size_t(index) >= source.texcoords.size())
                    throw std::runtime_error("Textured mesh requires valid OBJ UVs: " + filename);
                surface.uvs[k] = glm::vec2(source.texcoords[size_t(index)][0], source.texcoords[size_t(index)][1]);
                if (!std::isfinite(surface.uvs[k].x) || !std::isfinite(surface.uvs[k].y))
                    throw std::runtime_error("Invalid OBJ UV: " + filename);
            }
        }
        loadedTriangles.push_back(triangle);
        loadedSurfaces.push_back(surface);
        for (const auto& corner : {v0, v1, v2}) {
            boundsMin = glm::min(boundsMin, corner);
            boundsMax = glm::max(boundsMax, corner);
        }
    };

    for (const auto& face : source.faces) {
        const auto indices = obj::triangulate(source, face, filename);
        if (indices.empty()) { ++skippedDegenerate; continue; }
        for (const auto& triangle : indices) {
            const auto i0 = face.corners[triangle[0]];
            const auto i1 = face.corners[triangle[1]];
            const auto i2 = face.corners[triangle[2]];
            const auto point = [&](int index) {
                const auto& p = source.positions[index];
                return glm::vec3(p[0], p[1], p[2]);
            };
            appendTriangle(point(i0.vertex), point(i1.vertex), point(i2.vertex), i0, i1, i2);
        }
    }
    if (loadedTriangles.empty()) {
        throw std::runtime_error("OBJ contains no valid triangles: " + filename);
    }

    // Expand in local space, including zero-thickness axes. Scale the margin
    // to the stored coordinates rather than imposing a fixed minimum size.
    double coordinateScale = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        coordinateScale = (std::max)(coordinateScale,
            (std::max)(std::abs(double(boundsMin[axis])), std::abs(double(boundsMax[axis]))));
    }
    const double margin = 8.0 * std::numeric_limits<float>::epsilon() * coordinateScale;
    const double limit = (std::numeric_limits<float>::max)();
    for (int axis = 0; axis < 3; ++axis) {
        boundsMin[axis] = static_cast<float>((std::max)(-limit, double(boundsMin[axis]) - margin));
        boundsMax[axis] = static_cast<float>((std::min)(limit, double(boundsMax[axis]) + margin));
    }

    const int start = static_cast<int>(triangles.size());
    const int count = static_cast<int>(loadedTriangles.size());
    triangles.insert(triangles.end(), loadedTriangles.begin(), loadedTriangles.end());
    surfaces.insert(surfaces.end(), loadedSurfaces.begin(), loadedSurfaces.end());
    geom.triangleStart = start;
    geom.triangleCount = count;
    geom.boundsMin = boundsMin;
    geom.boundsMax = boundsMax;

    cout << "Loaded OBJ '" << filename << "': faces=" << source.faces.size()
         << ", triangles=" << count << ", triangleStart=" << start
         << ", skipped_degenerate=" << skippedDegenerate << '\n'
         << "  Local bounds: " << glm::to_string(boundsMin)
         << " -> " << glm::to_string(boundsMax) << endl;
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    if (!f) {
        throw std::runtime_error("Unable to open scene file: " + jsonName);
    }
    json data = json::parse(f);
    const auto sceneDirectory = std::filesystem::absolute(jsonName).parent_path();
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    std::unordered_map<std::string, int> textureCache;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
        }
        if (p["TYPE"] == "CoatedDiffuse") {
            newMaterial.color = readGeometryVector(p, "RGB");
            newMaterial.coatWeight = p.value("COAT_WEIGHT", 0.12f);
            if (!std::isfinite(newMaterial.coatWeight) || newMaterial.coatWeight < 0 || newMaterial.coatWeight > 1)
                throw std::runtime_error("COAT_WEIGHT must be in [0,1]");
        }
        if (p.contains("BASE_COLOR_TEXTURE")) {
            const auto path = (sceneDirectory / std::filesystem::u8path(p.at("BASE_COLOR_TEXTURE").get<std::string>())).lexically_normal();
            const auto key = path.string();
            const auto cached = textureCache.find(key);
            if (cached != textureCache.end()) newMaterial.baseColorTexture = cached->second;
            else {
                int w = 0, h = 0, channels = 0;
                std::unique_ptr<unsigned char, decltype(&stbi_image_free)> bytes(
                    stbi_load(key.c_str(), &w, &h, &channels, 3), &stbi_image_free);
                if (!bytes || w <= 0 || h <= 0) throw std::runtime_error("Unable to load base-color texture: " + key);
                const size_t count = size_t(w)*size_t(h);
                if (count > size_t(INT_MAX) - texturePixels.size()) throw std::runtime_error("Texture pixels exceed int indexing");
                const int id = int(textures.size());
                textures.push_back({w, h, int(texturePixels.size())});
                for (size_t i = 0; i < count; ++i)
                    texturePixels.emplace_back(decodeSRGB(bytes.get()[i*3]/255.f),
                        decodeSRGB(bytes.get()[i*3+1]/255.f), decodeSRGB(bytes.get()[i*3+2]/255.f));
                textureCache[key] = id;
                newMaterial.baseColorTexture = id;
            }
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const std::string type = p.at("TYPE").get<std::string>();
        Geom newGeom{};
        if (type == "cube") {
            newGeom.type = CUBE;
        }
        else if (type == "sphere") {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh") {
            newGeom.type = MESH;
        }
        else {
            throw std::runtime_error("Unknown geometry type: " + type);
        }
        const std::string materialName = p.at("MATERIAL").get<std::string>();
        const auto material = MatNameToID.find(materialName);
        if (material == MatNameToID.end()) {
            throw std::runtime_error("Unknown material: " + materialName);
        }
        newGeom.materialid = static_cast<int>(material->second);
        newGeom.smoothNormals = p.value("SMOOTH_NORMALS", false);
        newGeom.textured = materials[newGeom.materialid].baseColorTexture >= 0;
        if (newGeom.textured && newGeom.type != MESH)
            throw std::runtime_error("BASE_COLOR_TEXTURE currently requires a UV-mapped mesh");
        newGeom.translation = readGeometryVector(p, "TRANS");
        newGeom.rotation = readGeometryVector(p, "ROTAT");
        newGeom.scale = readGeometryVector(p, "SCALE");
        if (newGeom.scale.x == 0.0f || newGeom.scale.y == 0.0f || newGeom.scale.z == 0.0f) {
            throw std::runtime_error("SCALE components must be non-zero");
        }
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        if (!finiteMatrix(newGeom.transform) || !finiteMatrix(newGeom.inverseTransform)
            || !finiteMatrix(newGeom.invTranspose)) {
            throw std::runtime_error("Geometry transform must have a finite inverse");
        }
        if (newGeom.type == MESH) {
            const std::string filename = p.at("FILE").get<std::string>();
            if (filename.empty()) {
                throw std::runtime_error("Mesh FILE must not be empty");
            }
            const auto meshPath = (sceneDirectory / std::filesystem::u8path(filename)).lexically_normal();
            loadObjMesh(meshPath.u8string(), newGeom);
        }

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"].get<int>();
    if (state.traceDepth < 1) {
        throw std::runtime_error("Camera.DEPTH must be >= 1");
    }
    state.imageName = cameraData["FILE"];
    state.displayTransform = cameraData.value("DISPLAY_TRANSFORM", false);
    state.exposure = cameraData.value("EXPOSURE", 0.0f);
    if (!std::isfinite(state.exposure) || std::abs(state.exposure) > 20)
        throw std::runtime_error("Camera.EXPOSURE must be finite and within [-20,20]");
    state.enableAntialiasing = cameraData.value("ANTIALIASING", false);
    state.enableStreamCompaction = cameraData.value("STREAM_COMPACTION", false);
    state.enableMaterialSorting = cameraData.value("MATERIAL_SORTING", false);
    state.enableMeshCulling = cameraData.value("MESH_CULLING", true);
    state.enableBVH = cameraData.value("BVH", false);
    camera.position = readCameraVector(cameraData, "EYE");
    camera.lookAt = readCameraVector(cameraData, "LOOKAT");
    camera.up = readCameraVector(cameraData, "UP");
    const glm::vec3 viewOffset = camera.lookAt - camera.position;
    const float lookAtDistance = glm::length(viewOffset);
    if (!std::isfinite(lookAtDistance) || !(lookAtDistance > 0.0f)) {
        throw std::runtime_error("Camera.EYE and LOOKAT must define a finite non-zero distance");
    }
    cameraSampling::LensFrame lensFrame;
    if (!cameraSampling::makeLensFrame(viewOffset, camera.up, lensFrame)) {
        throw std::runtime_error("Camera.UP must be non-zero and not parallel to the view direction");
    }
    // Validate a separate lens frame without changing the legacy pinhole axes.
    if (cameraData.contains("DEPTH_OF_FIELD") && !cameraData.at("DEPTH_OF_FIELD").is_boolean()) {
        throw std::runtime_error("Camera.DEPTH_OF_FIELD must be boolean");
    }
    state.enableDepthOfField = cameraData.value("DEPTH_OF_FIELD", false);
    camera.lensRadius = cameraData.contains("LENS_RADIUS")
        ? readCameraNumber(cameraData.at("LENS_RADIUS"), "LENS_RADIUS") : 0.0f;
    camera.focalDistance = cameraData.contains("FOCAL_DISTANCE")
        ? readCameraNumber(cameraData.at("FOCAL_DISTANCE"), "FOCAL_DISTANCE") : lookAtDistance;
    if (!(camera.lensRadius >= 0.0f)) {
        throw std::runtime_error("Camera.LENS_RADIUS must be >= 0");
    }
    if (!(camera.focalDistance > 0.0f)) {
        throw std::runtime_error("Camera.FOCAL_DISTANCE must be > 0");
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
