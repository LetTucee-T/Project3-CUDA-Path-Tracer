# CUDA Path Tracer

**Author: Qingyao Tian** · University of Pennsylvania, CIS 565

A progressive CUDA path tracer with OBJ meshes, GPU BVH traversal, and a thin-lens camera.

![Shatterseal Drakesnest in a moonlit hall, rendered by this CUDA path tracer](docs/images/showcase.png)

*Shatterseal Drakesnest — 1200 × 800, 3072 samples per pixel (spp), maximum depth 8; 86,355 triangles across 16 meshes. One capture took 810.481 s in the render loop on an RTX 4070 Laptop GPU, excluding loading, initialization, and warmup.*

The fractured blade reflects the windows and candles above a textured stone floor. This is an actual renderer output, with exposure/tone mapping and no denoising. The distant exterior uses an AI-generated texture on geometry; asset sources and [scene details](docs/SCENE.md) are documented separately.

## Features

- Cosine-weighted diffuse scattering, ideal mirror reflection, emissive surfaces, and multiple path bounces.
- Stochastic pixel sampling, Thrust stream compaction, and independently toggleable material sorting.
- OBJ meshes with optional smooth normals and UV textures; brute-force, mesh AABB, and BVH intersection modes.
- Physical depth of field with adjustable aperture radius and focus distance.
- Textured diffuse/specular mixtures, interactive controls, and headless PNG / linear HDR capture.

## Image quality

### Stochastic antialiasing

Each sample traces a ray through a random position within its pixel. The comparison below isolates edge coverage with an emissive sphere, avoiding indirect-lighting noise: 161 × 159, 1000 spp, depth 1.

| Pixel-center sampling | Stochastic sampling |
| --- | --- |
| ![AA disabled](docs/images/aa_edge_off.png) | ![AA enabled](docs/images/aa_edge_on.png) |

Partial pixel coverage smooths the silhouette. [Full-scene comparisons and measurements](docs/RESULTS.md#antialiasing) use matched settings within the same renderer version.

### Physical depth of field

The camera samples a circular lens uniformly by area, then aims toward the pinhole ray's intersection with the focal plane. A zero lens radius reproduces the pinhole camera. Lens sampling uses an independent random stream.

![Pinhole and two aperture radii at a fixed focus distance](docs/images/dof_aperture.png)

*640 × 400, 2048 spp, depth 8; focus distance 10.5. Increasing lens radius from 0.10 to 0.35 strengthens defocus while keeping the middle vase in focus.*

At 320 × 200 and 128 spp, the still-life render-loop median changed from **637.333 to 643.098 ms** over six rounds. The 0.9% increase has overlapping timing ranges; it is not a universal overhead. Lens sampling also introduces additional sampling variance. [Focus changes, convergence, and timing details](docs/RESULTS.md#physical-depth-of-field) are available in the results appendix.

## Performance

Recorded on **Windows 11, Intel Core i7-14650HX, approximately 32 GiB RAM, RTX 4070 Laptop GPU (8 GiB), CUDA 12.9, VS 2022, Release**. Each experiment compares settings within its own source snapshot; timings from different stages should not be treated as one cumulative speedup. GPU clocks were not locked.

### Stream compaction and material sorting

After shading, terminated paths save their contribution and `thrust::remove_if` packs surviving paths. Pixel identities and random seeds stay attached to their paths. Open and closed scenes use the same camera; the closed version adds a wall behind it.

| Scene | Compaction off | Compaction on | Speedup |
| --- | ---: | ---: | ---: |
| Open | 19.653 s | 15.025 s | 1.308× |
| Closed | 21.394 s | 21.264 s | 1.006× |

*800 × 800, 1000 spp, depth 8, AA on; three-run medians after warmup. These are whole-program times, including initialization, preview, and saving.*

![Unterminated paths after each bounce in open and closed scenes](docs/images/active_paths_by_bounce.png)

*Counts are from sample iteration 1 after each shading pass; the last drop includes depth-limit termination.*

Escaping paths make compaction useful in the open scene. The closed scene retains more paths, so selection and movement nearly cancel the savings. Both off/on image pairs are byte-identical.

Material sorting uses `thrust::sort_by_key` on paired path/intersection records before shading. In its separate experiment, with compaction enabled, sorting changed open-scene runtime from **13.946 to 47.561 s** and closed-scene runtime from **20.121 to 64.881 s**. Classification, sorting, and moving records outweighed the benefit of grouping these simple BSDFs. Output stayed identical; sorting is disabled in the showcase. [Methods, raw data, and phase breakdown](docs/RESULTS.md#path-organization).

### OBJ meshes and BVH

OBJ polygons are triangulated on the CPU. A mesh-level AABB rejects misses before scanning triangles. The BVH further partitions each mesh into a binary hierarchy, built on the CPU and traversed on the GPU with an explicit stack, near-child ordering, and closest-hit pruning.

| Scene | Brute force | Mesh AABB | BVH | BVH speedup vs. brute force |
| --- | ---: | ---: | ---: | ---: |
| Simple meshes, 36 triangles | 0.911 s | 0.914 s | 0.930 s | 0.98× |
| Torus, exterior view | 5.666 s | 4.383 s | 0.939 s | 6.04× |
| Torus, inside its AABB | 5.856 s | 5.425 s | 0.952 s | 6.15× |

*256 × 256, 64 spp, depth 8; AA and compaction on, sorting off. Six-round whole-program medians after warmup; torus scenes contain 4,108 triangles.*

BVH helps the larger meshes even when the ray starts inside the root box. For tiny meshes, overlapping leaf bounds and traversal overhead can outweigh pruning. All three modes produced identical comparison images. [Image comparisons, GPU timings, CPU comparison, and future optimizations](docs/RESULTS.md#obj-meshes-and-bvh) use the final, numerically validated dataset.

## Build and run

The tested setup uses CMake 3.24+, Visual Studio 2022 C++ tools, CUDA 12.9, and the bundled Windows graphics dependencies. From the repository root, adjusting the CUDA installation path if needed:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" -DBUILD_TESTING=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
.\build\bin\Release\cis565_path_tracer.exe scenes/shatterseal_moonlit_hall_refined_preview.json
```

The preview uses 600 × 400, 256 spp, and a pinhole camera. Use `scenes/shatterseal_moonlit_hall_refined.json` for the cover settings. Scene asset paths are relative to the JSON file; the external extraction directory is not needed.

Headless capture writes PNG, HDR, float32 RGB, and a JSON timing record:

```powershell
.\build\bin\Release\scene_render.exe scenes/shatterseal_moonlit_hall_refined.json build/captures/moonlit_hall.json
```

Mouse: left drag orbits, right drag zooms, middle drag pans. `S` saves; `Esc` saves and exits; `Space` resets the look-at point. ImGui exposes AA, compaction, sorting, mesh culling/BVH, and lens controls. Changes restart accumulation. JSON fields and defaults are listed in the [results appendix](docs/RESULTS.md#configuration-and-validation).

CMake additions beyond source registration include CTest targets, their include paths/CUDA settings/test definitions, and the `scene_render` executable. A clean build from the selected submission files passed **9/9 CTest targets**; archived GPU memory checks reported zero errors and leaks. [Validation scope and records](docs/RESULTS.md#configuration-and-validation).

## Limitations and credits

Current materials use ideal reflection and a diffuse/specular mixture, without rough microfacet reflection, refraction, or normal mapping. Small emitters are found through scattered paths; direct-light sampling would improve their convergence. The exterior is a textured backdrop, not a full outdoor model or environment-map system.

- Framework: University of Pennsylvania CIS 565 CUDA Path Tracer starter, including its bundled dependencies.
- Added libraries: tinyobjloader (MIT) and Earcut (ISC); [versions, licenses, and loader scope](external/THIRD_PARTY.md).
- Weapon geometry and original textures: **Monster Hunter Wilds, CAPCOM**, supplied as an extracted asset and adapted for this renderer. [Asset notes](docs/SCENE.md#weapon-asset).
- Hall, candles, and procedural material assets: generated by project scripts. Concept references and the distant exterior texture used OpenAI image generation; [provenance and prompt](docs/SCENE.md#asset-sources). The displayed final renders were produced by the CUDA renderer.

[Detailed results and data index](docs/RESULTS.md) · [Repository contents](docs/SUBMISSION.md)

