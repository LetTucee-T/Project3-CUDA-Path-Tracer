# Repository contents

The submission contains one current renderer implementation, runnable scenes, tests, and selected experimental evidence.

| Include | Purpose |
| --- | --- |
| `src/`, `CMakeLists.txt`, `cmake/` | Current C++/CUDA implementation and build setup |
| `external/` | Required headers, bundled graphics libraries, and licenses |
| `tests/` | Tests for the current implementation; these are not historical source snapshots |
| `tools/render_scene.cpp` | Headless production-renderer capture tool |
| `tools/check_submission.py` | Submission/link/asset audit and optional local export |
| `scenes/` and selected `scenes/models/` assets | Final hall views, DOF example, baseline and validation scenes |
| `README.md`, `docs/RESULTS.md`, `docs/SCENE.md` | Feature overview, analysis, usage, and attribution |
| `docs/images/`, `docs/data/` | Selected comparison figures, measurements, validation records, and file hashes |

The existing starter files are retained. The `.lib` files under `external/lib/` are build dependencies; they must not be removed with generated binaries. OBJ files under `scenes/` are mesh assets.

The selected working-tree content is approximately **51 MiB**, including 12 scenes, 18 selected PNG figures, and small data/validation records. This excludes Git history and build output. The detailed candidate set can be regenerated with the audit command below.

## Local-only material

`.gitignore` excludes `stage_records/`, `readme_materials/`, build directories, executable/debug output, raw HDR/float32 captures, root-level render images, ImGui state, old hall scenes, duplicate weapon exports, and unused game texture maps. The author/permission checklist in `docs/DRAFT_CHECKLIST.md` also stays local. These files remain on disk.

The final hall still uses four assets from `scenes/models/moonlit_hall/`; explicit exceptions retain them. Other assets in that older directory are excluded. Only the weapon's three component meshes, source MTL declarations, and used metal color map are retained.

Images and measurement records in `docs/` are unchanged copies; [evidence_manifest.json](data/evidence_manifest.json) records their SHA-256 and archival origins. `.gitattributes` disables line-ending conversion for these files so the hashes also survive Git checkouts. Historical paths inside CSV/JSON records identify capture-time files, not required repository dependencies. Full stage code and repeated raw outputs are intentionally omitted.

## Verify before committing

From the repository root:

```powershell
python tools/check_submission.py
git status --short
git diff --check
```

The audit checks the union of tracked and non-ignored untracked files, including Markdown links, scene/MTL dependencies, CMake source references, and copied evidence hashes. It does not stage or commit files. `.gitignore` does not untrack files already added to Git; the audit rejects any accidentally tracked historical archive.

To create a fresh local copy containing only those candidate files:

```powershell
python tools/check_submission.py --export build/submission_check/source
```

The destination must be empty and inside `build/`. Configure/build that copy with the commands in the [README](../README.md), then run CTest. The export and its build products are ignored; no Git metadata or local archives are copied.

The September 27 packaging check completed a clean Release build, passed **9/9 tests**, validated **12/12 scenes**, and matched the working renderer's PNG/float32 output for a small DOF-enabled showcase capture. [Verification summary](data/submission_validation.json) · [CTest output](data/submission_ctest.txt). The existing bundled-library LNK4098 warning remains; it did not prevent the build or tests.
