# Additional dependencies

## tinyobjloader

- Version: `v2.0.0rc13`, C++ single-header implementation.
- Upstream: https://github.com/tinyobjloader/tinyobjloader/tree/v2.0.0rc13
- Header: `include/tiny_obj_loader.h` (unmodified).
- License: MIT, retained in `licenses/tinyobjloader-LICENSE.txt` and in the header.
- Header SHA-256: `a55a0933a29caa0e9e84ef62a45cbe8cb25dd0e17bc2850c591c6b06f95d5ebe`.
- License SHA-256: `19e1b0242709694c6671369609544993a2dfcef8477f1aa19bd30de35ec58caa`.

The CPU loader reads polygon mesh geometry and computes flat face normals.
It also retains independent corner UV and normal indices for base-color texture sampling and optional smooth shading.
The scene JSON assigns a material and optional texture to each mesh; MTL shading is not imported automatically.
Curves, points, skinning and animation are outside this scope.
Non-triangular faces are projected onto their dominant plane and triangulated using Earcut, preserving the input winding.
Nonplanar/self-intersecting polygons should be triangulated in the authoring tool.

## Earcut

- Version: `v2.2.4`, C++ single-header implementation.
- Upstream: https://github.com/mapbox/earcut.hpp/tree/v2.2.4
- Header: `include/mapbox/earcut.hpp` (unmodified).
- License: ISC, retained in `licenses/earcut-LICENSE.txt`.
- Header SHA-256: `3b838bc3f643d34ced850b6d53b19e2d9abdf6c2c067425b02868d131379b3cc`.
- License SHA-256: `828f2aed51b6526881a236758ec9b08cd69928fbfc70346d9d44a0b3a3444fe1`.

TinyObjLoader's built-in quadrilateral shortcut can choose the exterior diagonal of a concave quad.
The loader therefore validates the original polygon indices and triangulates polygons directly with Earcut.
Both clockwise and counterclockwise concave quadrilaterals have regression tests.
