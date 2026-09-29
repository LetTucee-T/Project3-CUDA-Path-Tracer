# CUDA Path Tracer

**Author: Qingyao Tian** · University of Pennsylvania, CIS 565

A progressive CUDA path tracer with OBJ meshes, GPU BVH traversal, and a thin-lens camera.

![Broken blade in a moonlit hall, rendered by this CUDA path tracer](docs/images/showcase.png)

*Broken Blade — 1200 × 800, 3072 samples per pixel (spp), maximum depth 8; 86,355 triangles across 16 meshes. One capture took 810.481 s in the render loop on an RTX 4070 Laptop GPU, excluding loading, initialization, and warmup.*

The fractured blade reflects Gothic windows and candles above a textured stone floor. A shared distant backdrop and larger fill emitters shape the lighting. This image comes from the CUDA renderer with exposure/tone mapping and no denoising.

## Features

- Cosine-weighted diffuse scattering, ideal mirror reflection, emissive surfaces, and multiple path bounces.
- Stochastic pixel sampling, Thrust stream compaction, and independently toggleable material sorting.
- An in-tree OBJ reader and polygon triangulator, optional smooth normals and UV textures; brute-force, mesh AABB, and BVH intersection modes.
- Physical depth of field with adjustable aperture radius and focus distance.
- Textured diffuse/specular mixtures, interactive controls, and headless PNG / linear HDR capture.

## Image quality

### Stochastic antialiasing

Each sample traces a ray through a random position within its pixel. The comparison below isolates edge coverage with an emissive sphere, avoiding indirect-lighting noise: 161 × 159, 1000 spp, depth 1.

| Pixel-center sampling | Stochastic sampling |
| --- | --- |
| ![AA disabled](docs/images/aa_edge_off.png) | ![AA enabled](docs/images/aa_edge_on.png) |

Partial pixel coverage smooths the silhouette. [Additional comparisons and raw timings](#additional-results-and-data) use matched settings within the same renderer version.

### Physical depth of field

The camera samples a circular lens uniformly by area, then aims toward the pinhole ray's intersection with the focal plane. A zero lens radius reproduces the pinhole camera. Lens sampling uses an independent random stream.

![Pinhole and two aperture radii at a fixed focus distance](docs/images/dof_aperture.png)

*640 × 400, 2048 spp, depth 8; focus distance 10.5. Increasing lens radius from 0.10 to 0.35 strengthens defocus while keeping the middle vase in focus.*

At 320 × 200, 128 spp, depth 8, with AA/compaction/BVH on and sorting off, the still-life render-loop median changed from **637.333 to 643.098 ms** over six rounds. The 0.9% increase has overlapping timing ranges; it is not a universal overhead. Lens sampling also introduces additional sampling variance.

Independent lens samples map naturally to GPU threads; a hypothetical CPU version could use threads/SIMD but performs the same sampling arithmetic. BVH and compaction reduce downstream work, and zero radius skips lens sampling. Further improvements include precomputing the lens frame, testing more uniform samples, and resetting accumulation without re-uploading geometry.

## Performance

Recorded on **Windows 11, Intel Core i7-14650HX, RTX 4070 Laptop GPU (8 GiB), 32 GiB RAM, CUDA 12.9**. Each experiment compares settings within its own source snapshot; timings from different stages should not be treated as one cumulative speedup. GPU clocks were not locked.

Whole-program timings include startup and saving; render-loop timings exclude loading, initialization, warmup, and saving but include synchronization/readback. GPU intervals use CUDA events in a separate, output-checked build. Warmups are excluded from benchmark medians; min–max ranges are not confidence intervals. CPU comparisons are qualitative: no CPU renderer was benchmarked.

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

Material sorting uses `thrust::sort_by_key` on paired path/intersection records before shading. In its separate three-run experiment at the same resolution/spp/depth, with compaction enabled, sorting changed open-scene runtime from **13.946 to 47.561 s** and closed-scene runtime from **20.121 to 64.881 s**. Classification, sorting, and moving records outweighed the benefit of grouping these simple BSDFs. Output stayed identical; sorting is disabled in the showcase. Compact indices or category partitioning are possible future improvements.

### OBJ meshes and BVH

The custom [OBJ reader and ear-clipping triangulator](src/objLoader.cpp) run on the CPU using only the C++17 standard library. Independent corner UVs/normals and polygon winding are preserved; triangles and simple planar polygons, including concave faces, are supported ([input scope](external/THIRD_PARTY.md#supported-obj-input)). A mesh-level AABB rejects misses before scanning triangles. The BVH further partitions each mesh into a binary hierarchy, built on the CPU and traversed on the GPU with an explicit stack, near-child ordering, and closest-hit pruning.

| Scene | Brute force | Mesh AABB | BVH | BVH speedup vs. brute force |
| --- | ---: | ---: | ---: | ---: |
| Simple meshes, 36 triangles | 0.911 s | 0.914 s | 0.930 s | 0.98× |
| Torus, exterior view | 5.666 s | 4.383 s | 0.939 s | 6.04× |
| Torus, inside its AABB | 5.856 s | 5.425 s | 0.952 s | 6.15× |

*256 × 256, 64 spp, depth 8; AA and compaction on, sorting off. Six-round whole-program medians after warmup; torus scenes contain 4,108 triangles.*

BVH helps the larger meshes even when the ray starts inside the root box. For tiny meshes, overlapping leaf bounds and traversal overhead can outweigh pruning. All three modes produced identical comparison images. The switches share CPU build/upload costs; this table compares traversal choices using the final validated dataset.

OBJ parsing and tree construction already run on the CPU. A hypothetical CPU tracer receives the same AABB/BVH pruning; GPU traversal adds ray parallelism but incurs divergent control flow, scattered reads, and per-thread stack storage. Further improvements include caching repeated meshes, small-mesh direct scanning, SAH splits, and leaf/stack tuning.

## Build and run

The tested setup uses CMake 3.24+, Visual Studio 2022 C++ tools, CUDA 12.9, and the bundled Windows graphics dependencies. No external OBJ library is needed; the other [framework dependencies](external/THIRD_PARTY.md#framework-dependencies) remain in use. From the repository root, adjusting the CUDA installation path if needed:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" -DBUILD_TESTING=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
.\build\bin\Release\cis565_path_tracer.exe scenes/shatterseal_moonlit_hall_refined_preview.json
```

The [preview](scenes/shatterseal_moonlit_hall_refined_preview.json) uses 600 × 400, 256 spp, and a pinhole camera. Use the [final scene](scenes/shatterseal_moonlit_hall_refined.json) for the cover settings, or the [detail camera](scenes/shatterseal_moonlit_hall_refined_detail.json) for a closer view. Scene asset paths are relative to the JSON file; all runtime assets are included.

Headless capture writes PNG, HDR, float32 RGB, and a JSON timing record:

```powershell
.\build\bin\Release\scene_render.exe scenes/shatterseal_moonlit_hall_refined.json build/captures/moonlit_hall.json
```

Mouse: left drag orbits, right drag zooms, middle drag pans. `S` saves; `Esc` saves and exits; `Space` resets the look-at point. ImGui exposes AA, compaction, sorting, mesh culling/BVH, and lens controls. Changes restart accumulation.

In `Camera`, `BVH=false, MESH_CULLING=false` selects brute force; `false,true` selects one mesh AABB; `BVH=true` selects hierarchical traversal. Enable `DEPTH_OF_FIELD` to use `LENS_RADIUS` and `FOCAL_DISTANCE`, both in world units. Materials are assigned per mesh in JSON, including `BASE_COLOR_TEXTURE` and `COAT_WEIGHT`; optional `SMOOTH_NORMALS` belongs to the mesh object. MTL shading is not imported automatically.

## Limitations

Current materials use ideal reflection and a diffuse/specular mixture, without rough microfacet reflection, refraction, or normal mapping. Small emitters are found through scattered paths; direct-light sampling would improve their convergence. The exterior is a textured backdrop, not a full outdoor model or environment-map system.

## Additional results and data

<details>
<summary>More comparisons, measurement details, and raw data</summary>

**BVH image equivalence.** These renders use 256 × 256, 256 spp, depth 8; all three intersection modes agree exactly. Benchmark timings above use 64 spp instead.

![Brute force, mesh AABB, BVH, and image differences](docs/images/bvh_comparison.png)

**Changing focus.** At radius 0.35, focus distances 7.7 / 10.5 / 14.2 move the sharp region. Each image is 640 × 400, 2048 spp, depth 8; the panels only arrange rendered images and add labels.

![Near, middle, and far focus](docs/images/dof_focus.png)

**Sorting cost.** Diagnostic CUDA-event intervals include Thrust dispatch/allocation/synchronization gaps and exclude camera generation, gather, and display. They are separate from the whole-program benchmark.

![Material-sorting phase intervals](docs/images/sorting_phase_breakdown.png)

| Experiment | Further figures | Data |
| --- | --- | --- |
| AA | [Cornell off](docs/images/aa_cornell_off.png) / [on](docs/images/aa_cornell_on.png) | [Timings](docs/data/aa_timings.csv) |
| Compaction | [Open](docs/images/compaction_open.png) / [closed](docs/images/compaction_closed.png); off/on images are identical | [Timings](docs/data/compaction_timings.csv), [paths per bounce](docs/data/compaction_path_counts.csv) |
| Material sorting | [Grouping](docs/images/sorting_material_order.png) | [Timings](docs/data/sorting_timings.csv), [summary](docs/data/sorting_summary.csv), [phase intervals](docs/data/sorting_phase_timings.csv), [grouping counts](docs/data/sorting_material_layout.csv) |
| OBJ / BVH | [Mesh example](docs/images/mesh_import.png), [timing ranges](docs/images/bvh_performance.png) | [BVH timings](docs/data/bvh_timings.csv), [summary](docs/data/bvh_summary.csv), [validation](docs/data/bvh_verification.json), [earlier AABB experiment](docs/data/aabb_results.json) |
| Depth of field | [Performance](docs/images/dof_performance.png), [convergence](docs/images/dof_convergence.png) | [Timings](docs/data/dof_timings.csv), [summary](docs/data/dof_summary.csv), [convergence](docs/data/dof_convergence.csv), [validation](docs/data/dof_verification.json) |
| Showcase | [Before lighting refinement](docs/images/hall_before.png) / [final](docs/images/showcase.png), both 1200 × 800 and 3072 spp | [Capture](docs/data/showcase_capture.json), [validation](docs/data/showcase_verification.json) |

DOF convergence uses each mode's own 8192-spp reference with disjoint sample numbers; intentional defocus is not an error. The references still contain noise, so this does not establish runtime at equal quality. The final cover's time is a single capture, not a repeated benchmark.

Measurements come from their recorded development versions. Historical paths inside CSV/JSON files identify capture-time files; the current renderer is the only source version distributed. [Evidence origins and hashes](docs/data/evidence_manifest.json) · [Hardware record](docs/data/host_environment.json) · [Appearance checks](docs/data/appearance_verification.json).

</details>

