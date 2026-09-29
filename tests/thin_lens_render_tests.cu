#include "pathtrace.h"
#include "cameraSampling.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

// Launch the production camera kernel, not a test-side implementation.
__global__ void generateRayFromCamera(Camera, int, int, bool, bool,
    PathSegment*, unsigned int*);

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void cudaCheck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void near(glm::vec3 a, glm::vec3 b, float tolerance = 4e-5f)
{
    for (int j = 0; j < 3; ++j)
        require(std::isfinite(a[j]) && std::abs(a[j] - b[j]) <= tolerance, "Geometry mismatch");
}
template<class T> struct DeviceArray {
    T* data = nullptr; size_t count;
    explicit DeviceArray(size_t n) : count(n) { cudaCheck(cudaMalloc(&data, n * sizeof(T))); }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    std::vector<T> read() {
        std::vector<T> out(count);
        cudaCheck(cudaMemcpy(out.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
};
struct CameraResult { std::vector<PathSegment> paths; unsigned int invalid; };
CameraResult rays(Camera cam, bool aa, bool dof, int iteration = 3)
{
    const int count = cam.resolution.x * cam.resolution.y;
    DeviceArray<PathSegment> paths(count + 7);
    DeviceArray<unsigned int> errors(1);
    cudaCheck(cudaMemset(paths.data, 0xad, paths.count * sizeof(PathSegment)));
    cudaCheck(cudaMemset(errors.data, 0, sizeof(unsigned int)));
    generateRayFromCamera<<<dim3((cam.resolution.x + 7) / 8, (cam.resolution.y + 7) / 8), dim3(8, 8)>>>(
        cam, iteration, 8, aa, dof, paths.data, errors.data);
    cudaCheck(cudaGetLastError()); cudaCheck(cudaDeviceSynchronize());
    auto out = paths.read();
    const unsigned char* tail = reinterpret_cast<const unsigned char*>(out.data() + count);
    for (size_t j = 0; j < 7 * sizeof(PathSegment); ++j)
        require(tail[j] == 0xad, "Camera kernel wrote outside the image");
    out.resize(count);
    return {out, errors.read()[0]};
}
Camera camera()
{
    Camera cam{};
    cam.resolution = glm::ivec2(33, 17);
    cam.position = glm::vec3(2, 3, 7);
    cam.view = glm::normalize(glm::vec3(-0.2f, -0.3f, -1));
    cam.right = glm::normalize(glm::cross(cam.view, glm::vec3(0, 1, 0)));
    cam.up = glm::cross(cam.right, cam.view);
    cam.pixelLength = glm::vec2(0.045f);
    cam.lensRadius = 0.18f; cam.focalDistance = 6;
    return cam;
}
bool equalImage(const std::vector<glm::vec3>& a, const std::vector<glm::vec3>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(glm::vec3)) == 0;
}
struct RenderSession {
    DeviceArray<uchar4> output;
    explicit RenderSession(Scene& s, const PathtraceOptions& options = {}) : output(s.state.image.size()) { pathtraceInit(&s, options); }
    ~RenderSession() { pathtraceFree(); }
};
std::vector<glm::vec3> render(Scene& s, int samples)
{
    RenderSession session(s);
    for (int i = 1; i <= samples; ++i) pathtrace(session.output.data, 0, i);
    pathtraceReadback();
    for (auto p : s.state.image) {
        require(cameraSampling::finite(p), "Image contains NaN/Inf");
        require(p.x >= 0 && p.y >= 0 && p.z >= 0, "Negative radiance");
    }
    return s.state.image;
}
void smallScene(Scene& s)
{
    s.state.camera.resolution = glm::ivec2(33, 17);
    s.state.camera.pixelLength *= 3.0f;
    s.state.camera.lensRadius = 0.22f;
    s.state.camera.focalDistance = 4.0f;
    s.state.image.resize(33 * 17);
    s.state.traceDepth = 5;
}
}

int main(int argc, char** argv)
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    if (argc != 2) return 1;
    const auto fixture = std::filesystem::path(argv[1]) / "scenes/mesh_gpu_validation.json";
    int passed = 0, failed = 0;
    auto test = [&](const std::string& name, const std::function<void()>& body) {
        try { body(); ++passed; std::cout << "PASS " << name << '\n'; }
        catch (const std::exception& e) { ++failed; std::cerr << "FAIL " << name << ": " << e.what() << '\n'; }
    };
    for (bool aa : {false, true}) {
        test("production camera: pinhole identity, AA=" + std::to_string(aa), [&] {
            Camera cam = camera();
            const auto off = rays(cam, aa, false);
            cam.lensRadius = 0;
            const auto zero = rays(cam, aa, true);
            require(off.invalid == 0 && zero.invalid == 0, "Pinhole generated invalid samples");
            for (size_t i = 0; i < off.paths.size(); ++i) {
                require(std::memcmp(&off.paths[i], &zero.paths[i], sizeof(PathSegment)) == 0, "Zero lens changed pinhole path");
                require(off.paths[i].pixelIndex == int(i) && off.paths[i].remainingBounces == 8, "Path identity/depth changed");
                near(off.paths[i].color, glm::vec3(1), 0);
            }
        });
        test("production camera: focal plane and aperture, AA=" + std::to_string(aa), [&] {
            const Camera cam = camera();
            const auto pinhole = rays(cam, aa, false);
            const auto lens = rays(cam, aa, true);
            require(lens.invalid == 0, "Valid camera rejected");
            float meanRadius2 = 0;
            for (size_t i = 0; i < lens.paths.size(); ++i) {
                const auto& p = pinhole.paths[i]; const auto& q = lens.paths[i];
                const glm::vec3 offset = q.ray.origin - cam.position;
                require(std::abs(glm::dot(offset, cam.view)) < 2e-6f, "Origin outside lens plane");
                const float radius2 = glm::dot(offset, offset);
                require(radius2 <= cam.lensRadius * cam.lensRadius * 1.0001f, "Origin outside lens disk");
                meanRadius2 += radius2;
                const auto focus = cam.position + p.ray.direction * (cam.focalDistance / glm::dot(p.ray.direction, cam.view));
                const auto actual = q.ray.origin + q.ray.direction *
                    (glm::dot(focus - q.ray.origin, cam.view) / glm::dot(q.ray.direction, cam.view));
                near(actual, focus);
                require(std::abs(glm::length(q.ray.direction) - 1) < 2e-6f, "Lens ray not unit length");
                require(q.pixelIndex == int(i) && q.remainingBounces == 8, "Lens path metadata changed");
                near(q.color, glm::vec3(1), 0);
            }
            meanRadius2 /= lens.paths.size() * cam.lensRadius * cam.lensRadius;
            require(meanRadius2 > 0.45f && meanRadius2 < 0.55f, "Aperture samples are not spread across disk");
        });
    }
    test("lens samples independent of AA and repeatable across iterations", [&] {
        const auto cam = camera();
        const auto a = rays(cam, false, true), b = rays(cam, true, true);
        const auto again = rays(cam, false, true), next = rays(cam, false, true, 4);
        size_t changed = 0;
        for (size_t i = 0; i < a.paths.size(); ++i) {
            near(a.paths[i].ray.origin, b.paths[i].ray.origin, 0);
            require(std::memcmp(&a.paths[i], &again.paths[i], sizeof(PathSegment)) == 0, "Non-reproducible camera path");
            changed += std::memcmp(&a.paths[i].ray.origin, &next.paths[i].ray.origin, sizeof(glm::vec3)) != 0;
        }
        require(changed == a.paths.size(), "Lens stream reused across iterations");
    });
    for (int invalid = 0; invalid < 4; ++invalid) {
        test("invalid active camera rejected, case=" + std::to_string(invalid), [&] {
            Camera cam = camera();
            if (invalid == 0) cam.focalDistance = 0;
            if (invalid == 1) cam.lensRadius = -0.1f;
            if (invalid == 2) cam.lensRadius = std::numeric_limits<float>::infinity();
            if (invalid == 3) cam.up = cam.view;
            const auto result = rays(cam, true, true);
            require(result.invalid == result.paths.size(), "Invalid rays not reported");
            for (const auto& p : result.paths)
                require(p.remainingBounces == 0, "Invalid ray remains active");
        });
    }
    for (bool aa : {false, true}) {
        test("full render disabled/zero aperture identity, AA=" + std::to_string(aa), [&] {
            Scene s(fixture.string()); smallScene(s); s.state.enableAntialiasing = aa;
            const auto off = render(s, 16);
            s.state.enableDepthOfField = true; s.state.camera.lensRadius = 0;
            require(equalImage(off, render(s, 16)), "Zero lens changed accumulated float image");
            s.state.camera.lensRadius = 0.22f;
            require(!equalImage(off, render(s, 16)), "Enabled DOF did not affect full render");
        });
        std::vector<glm::vec3> reference;
        for (int mode = 0; mode < 16; ++mode) {
            test("DOF full render flags, AA=" + std::to_string(aa) + " mode=" + std::to_string(mode), [&] {
                Scene s(fixture.string()); smallScene(s);
                s.state.enableAntialiasing = aa; s.state.enableDepthOfField = true;
                s.state.enableStreamCompaction = (mode & 1) != 0;
                s.state.enableMaterialSorting = (mode & 2) != 0;
                s.state.enableMeshCulling = (mode & 4) != 0;
                s.state.enableBVH = (mode & 8) != 0;
                const auto image = render(s, 12);
                require(std::any_of(image.begin(), image.end(), [](glm::vec3 p) { return p.x + p.y + p.z > 0; }), "Black image");
                if (mode == 0) reference = image;
                else require(equalImage(reference, image), "Scheduling/acceleration changed DOF image");
            });
        }
    }
    test("repeated parameter reset discards old accumulated samples", [&] {
        Scene s(fixture.string()); smallScene(s);
        const auto pose = s.state.camera;
        for (int cycle = 0; cycle < 8; ++cycle) {
            s.state.enableDepthOfField = cycle != 0 && cycle != 7;
            s.state.camera.lensRadius = 0.06f * cycle;
            s.state.camera.focalDistance = 3.0f + cycle;
            s.state.enableAntialiasing = (cycle & 1) != 0;
            s.state.enableStreamCompaction = true; s.state.enableMaterialSorting = true; s.state.enableBVH = true;
            render(s, 5);
            const auto resetImage = render(s, 1);
            Scene fresh(fixture.string()); smallScene(fresh); fresh.state = s.state;
            // A deliberately dirty host buffer must be overwritten, not accumulated.
            std::fill(fresh.state.image.begin(), fresh.state.image.end(), glm::vec3(9000));
            require(equalImage(resetImage, render(fresh, 1)), "Stale samples survived reset");
            near(s.state.camera.position, pose.position, 0); near(s.state.camera.view, pose.view, 0);
            near(s.state.camera.up, pose.up, 0); near(s.state.camera.right, pose.right, 0);
        }
        pathtraceFree();
    });
    test("explicit readback, display, profiling, synchronization and pruning agree", [&] {
        std::vector<glm::vec3> reference;
        std::vector<uchar4> displayReference;
        for (int mode = 0; mode < 32; ++mode) {
            Scene s(fixture.string()); smallScene(s);
            s.state.enableDepthOfField = true; s.state.enableAntialiasing = true;
            s.state.enableBVH = true; s.state.enableStreamCompaction = true;
            const bool each = (mode & 8) != 0, display = (mode & 16) != 0;
            PathtraceOptions options;
            options.synchronizeEachStage = (mode & 1) != 0;
            options.crossMeshPruning = (mode & 2) != 0;
            options.profile = (mode & 4) != 0;
            RenderSession session(s, options);
            std::fill(s.state.image.begin(), s.state.image.end(), glm::vec3(-17));
            for (int i = 1; i <= 3; ++i) {
                pathtrace(display ? session.output.data : nullptr, 0, i);
                if (each) pathtraceReadback();
                else require(s.state.image[0].x == -17, "Step unexpectedly read back host image");
            }
            pathtraceReadback();
            const auto statistics = pathtraceGetMetrics();
            require(statistics.samples == 3 && statistics.imageReadbacks == (each ? 3 : 1), "Wrong readback/sample count");
            require(statistics.readbackBytes == statistics.imageReadbacks * s.state.image.size() * sizeof(glm::vec3), "Wrong transfer byte count");
            if (reference.empty()) reference = s.state.image;
            else require(equalImage(reference, s.state.image), "Execution options changed radiance");
            if (display) {
                const auto pixels = session.output.read();
                if (displayReference.empty()) displayReference = pixels;
                else require(std::memcmp(pixels.data(), displayReference.data(), pixels.size()*sizeof(uchar4)) == 0, "Execution options changed preview");
            }
        }
    });
    test("BVH layouts, SAH leaf sizes and sampled replay preserve the rendered image", [&] {
        std::vector<glm::vec3> reference;
        for(auto method : {BVHSplitMethod::Median, BVHSplitMethod::BinnedSAH})
        for(int leaf : {1,2,4,8}) for(bool compact : {false,true})
        for(int stride : {0,1,7}) for(bool pruning : {false,true}) {
            Scene s(fixture.string(), {leaf,BVH_MAX_DEPTH,method,16}); smallScene(s);
            s.state.enableBVH=true; s.state.enableDepthOfField=true;
            s.state.enableStreamCompaction=true; s.state.enableMaterialSorting=true;
            PathtraceOptions options; options.profileIntersections=stride!=0;
            options.intersectionProfileStride=stride;
            options.crossMeshPruning=pruning;
            options.compactBVHNodes=compact;
            RenderSession session(s,options);
            for(int i=1;i<=3;++i) pathtrace(nullptr,0,i);
            pathtraceReadback();
            if(reference.empty()) reference=s.state.image;
            else require(equalImage(reference,s.state.image), "SAH/profiling changed radiance");
            const auto m=pathtraceGetMetrics();
            if(stride) {
                require(m.intersectionWork[0].sampledRays==3*((s.state.image.size()+stride-1)/stride), "Wrong sampled ray count");
                unsigned long long triangleTests=0;
                for(const auto& bounce:m.intersectionWork) {
                    require(bounce.mismatches==0,"Replay mismatch");
                    for(const auto& row:bounce.perGeometry) {
                        require(row.queries==bounce.sampledRays,"Per-object population changed");
                        require(row.fallbacks==0,"Valid SAH tree fell back to brute force");
                        triangleTests+=row.triangleTests;
                        if(!pruning) require(row.boundedQueries==0,"Unbounded query recorded a distance cap");
                    }
                }
                require(triangleTests>0,"Diagnostic never visited any mesh triangles");
            } else require(m.intersectionWork.empty(),"Disabled profiler collected work");
        }
    });
    test("production renderer reports invalid lens before exposing the image", [&] {
        Scene s(fixture.string()); smallScene(s);
        s.state.enableDepthOfField = true; s.state.camera.focalDistance = 0;
        bool rejected = false;
        try { render(s, 1); }
        catch (const std::runtime_error& e) { rejected = std::string(e.what()).find("Thin-lens camera produced") != std::string::npos; }
        require(rejected, "Invalid camera silently rendered");
    });
    std::cout << "THIN LENS RENDER RESULT: " << passed << " passed, " << failed << " failed\n";
    cudaDeviceReset();
    return failed == 0 ? 0 : 1;
}
