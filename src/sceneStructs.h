#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

// Flat-shaded triangle in mesh-local space; uploaded as one shared, contiguous GPU array.
struct Triangle
{
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
    glm::vec3 normal;
};

// Per-corner appearance data, separate from the compact intersection geometry.
struct TriangleSurface
{
    glm::vec3 normals[3];
    glm::vec2 uvs[3];
};
struct TextureInfo { int width = 0; int height = 0; int offset = 0; };

// Flat CPU/GPU BVH; indexCount > 0 denotes a leaf.
struct BVHNode
{
    // Scalar arrays guarantee a trivially-copyable layout even with the old GLM
    // bundled in this project (whose vec3 has a user-defined copy constructor).
    float boundsMin[3] = {0.0f, 0.0f, 0.0f};
    float boundsMax[3] = {0.0f, 0.0f, 0.0f};
    int leftChild = -1;
    int rightChild = -1;
    int firstIndex = -1;
    int indexCount = 0;
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    bool smoothNormals = false;
    bool textured = false;
    int triangleStart = 0;
    int triangleCount = 0;

    glm::vec3 boundsMin = glm::vec3(0.0f);
    glm::vec3 boundsMax = glm::vec3(0.0f);

    // One contiguous node block per mesh; the root is its first node.
    // bvhIndexStart addresses the index array, not the Triangle array.
    int bvhRoot = -1;
    int bvhNodeCount = 0;
    int bvhIndexStart = 0;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    int baseColorTexture = -1;
    float coatWeight = 0.0f;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    // Thin-lens parameters in world units; focalDistance is axial, not focal length.
    float lensRadius = 0.0f;
    float focalDistance = 1.0f;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    bool enableAntialiasing = false;
    bool enableDepthOfField = false;
    bool enableStreamCompaction = false;
    bool enableMaterialSorting = false;
    bool enableMeshCulling = true;
    bool enableBVH = false;
    bool displayTransform = false;
    float exposure = 0.0f;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec3 geometricNormal;
  glm::vec2 uv;
  bool useShadingNormal = false;
  bool hasUV = false;
};
