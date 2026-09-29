#pragma once

#include "sceneStructs.h"
#include "bvh.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadObjMesh(const std::string& filename, Geom& geom);
public:
    Scene(std::string filename, const BVHBuildOptions& options = {});
    // Rebuild after replacing triangle/mesh data; construction and validation are CPU-only.
    // Successful rebuilds replace the old arrays; failed rebuilds do not commit partial data.
    void rebuildMeshBVHs(const BVHBuildOptions& options = BVHBuildOptions{});

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    // Mesh Geom records address this shared array using triangleStart/count.
    std::vector<Triangle> triangles;
    std::vector<TriangleSurface> surfaces;
    std::vector<TextureInfo> textures;
    std::vector<glm::vec3> texturePixels;
    std::vector<BVHNode> bvhNodes;
    std::vector<int> bvhTriangleIndices;
    BVHStats bvhStats;
    BVHBuildOptions bvhBuildOptions;
    RenderState state;
};
