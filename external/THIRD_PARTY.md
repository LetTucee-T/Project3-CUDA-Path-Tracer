# Additional dependencies

## tinyobjloader

- Version: `v2.0.0rc13`, C++ single-header implementation.
- Upstream: https://github.com/tinyobjloader/tinyobjloader/tree/v2.0.0rc13
- Header: `include/tiny_obj_loader.h`

The CPU loader reads polygon mesh geometry and computes flat face normals.
It also retains independent corner UV and normal indices for base-color texture sampling and optional smooth shading.
The scene JSON assigns a material and optional texture to each mesh; MTL shading is not imported automatically.
Curves, points, skinning and animation are outside this scope.
Non-triangular faces are projected onto their dominant plane and triangulated using Earcut, preserving the input winding.
Nonplanar/self-intersecting polygons should be triangulated in the authoring tool.

## Earcut

- Version: `v2.2.4`, C++ single-header implementation.
- Upstream: https://github.com/mapbox/earcut.hpp/tree/v2.2.4
- Header: `include/mapbox/earcut.hpp`

TinyObjLoader's built-in quadrilateral shortcut can choose the exterior diagonal of a concave quad.
The loader therefore validates the original polygon indices and triangulates polygons directly with Earcut.
Both clockwise and counterclockwise concave quadrilaterals have regression tests.
