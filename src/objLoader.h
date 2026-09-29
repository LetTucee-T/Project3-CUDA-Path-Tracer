#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

// CPU-only OBJ polygon reader. Parsing and triangulation use the C++ standard
// library; rendering materials and smoothing policy remain scene JSON settings.
namespace obj {
struct Index {
    int vertex = -1;
    int uv = -1;
    int normal = -1;
};
struct Face {
    std::vector<Index> corners;
    std::size_t line = 0;
};
struct Mesh {
    std::vector<std::array<float, 3>> positions;
    std::vector<std::array<float, 2>> texcoords;
    std::vector<std::array<float, 3>> normals;
    std::vector<Face> faces;
};

Mesh read(const std::string& filename);

// Returns indices into Face::corners, preserving attribute seams and winding.
// Supports simple planar polygons; collinear polygons return no triangles.
// Self-intersecting/nonplanar polygons are rejected instead of repaired silently.
std::vector<std::array<std::size_t, 3>> triangulate(
    const Mesh& mesh, const Face& face, const std::string& filename);
}
