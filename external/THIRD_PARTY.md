# Dependencies and OBJ input scope

OBJ parsing and polygon triangulation are implemented in [objLoader.cpp](../src/objLoader.cpp) using only the C++17 standard library. No external OBJ parser or triangulation library is required.

## Supported OBJ input

- Polygon meshes with `v`, `vt`, `vn`, and `f`; independent vertex/UV/normal indices, positive and relative negative references, and face corners written as `v`, `v/vt`, `v//vn`, or `v/vt/vn`.
- Triangle faces keep their original corner and face order. Simple planar polygons, including concave faces, use the project's ear-clipping triangulator with preserved winding and corner attributes.
- Whitespace, comments, UTF-8 filenames/BOM, scientific notation, and continued lines are supported. UVs use the first two components; missing V defaults to zero. Optional vertex weights/colors are not used.
- `o`, `g`, `s`, `mtllib`, and `usemtl` do not change rendering policy: all polygon faces are retained, while scene JSON assigns one material per mesh and controls smooth normals. Missing normals use geometric face normals; textured meshes require UVs. MTL materials are not imported.
- Invalid numbers/indices produce errors with a filename and line number. Degenerate faces are discarded. Self-intersecting and nonplanar polygons must be triangulated before export. Curves, freeform surfaces, and other unsupported records are rejected.

## Framework dependencies

The framework still uses CUDA/Thrust for GPU execution and generic algorithms, GLM for rendering math, nlohmann/json for scene descriptions, stb for image I/O, GLFW/GLEW/OpenGL for the preview, and Dear ImGui for controls. Install the CUDA toolkit and a compatible graphics driver; the remaining headers/sources and Windows GLFW/GLEW libraries are bundled. Their existing copyright and license notices are retained.

## Historical OBJ libraries

Earlier captures used tinyobjloader v2.0.0rc13 and Earcut v2.2.4. Both libraries have been removed from the current build; their headers and standalone license files are excluded from submission. Historical evidence keeps its original provenance. [Replacement and appearance verification](../docs/data/obj_loader_verification.json) records the unchanged output.
