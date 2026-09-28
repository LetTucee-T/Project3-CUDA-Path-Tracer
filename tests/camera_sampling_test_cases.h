#pragma once

#include "cameraSampling.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cameraTest {
struct Query {
    Camera camera;
    Ray pinhole;
    glm::vec2 sample = glm::vec2(0.36f, 0.125f);
    uint32_t iteration = 1, pixel = 17;
    bool enabled = true;
};
struct Result {
    cameraSampling::LensFrame frame;
    Ray ray;
    glm::vec2 disk = glm::vec2(0), randomSample, randomDisk;
    uint32_t randomWord;
    bool frameOK, rayOK;
};

__host__ __device__ inline Result evaluate(const Query& q)
{
    Result result;
    result.frameOK = cameraSampling::makeLensFrame(q.camera.view, q.camera.up, result.frame);
    if (q.sample.x >= 0 && q.sample.x <= 1 && q.sample.y >= 0 && q.sample.y <= 1)
        result.disk = cameraSampling::sampleUnitDisk(q.sample);
    auto rng = cameraSampling::makeLensRandomEngine(q.iteration, q.pixel);
    auto copy = rng;
    result.randomWord = copy();
    thrust::uniform_real_distribution<float> uniform(0, 1);
    result.randomSample.x = uniform(rng);
    result.randomSample.y = uniform(rng);
    result.randomDisk = cameraSampling::sampleUnitDisk(result.randomSample);
    result.rayOK = cameraSampling::makeThinLensRay(q.camera, q.pinhole, q.enabled, q.sample, result.ray);
    return result;
}

inline Query query()
{
    Query q{};
    q.camera.position = glm::vec3(0);
    q.camera.view = glm::vec3(0, 0, -1);
    q.camera.up = glm::vec3(0, 1, 0);
    q.camera.right = glm::vec3(1, 0, 0);
    q.camera.lookAt = glm::vec3(0, 0, -6);
    q.camera.lensRadius = 0.3f;
    q.camera.focalDistance = 6;
    q.pinhole.origin = q.camera.position;
    q.pinhole.direction = q.camera.view;
    return q;
}

inline void require(bool ok, const std::string& message)
{
    if (!ok) throw std::runtime_error(message);
}
inline void near(float actual, float expected, float tolerance = 3e-5f)
{
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
        "Expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}
inline void nearVector(const glm::vec3& a, const glm::vec3& b, float tolerance = 3e-5f)
{
    for (int axis = 0; axis < 3; ++axis) near(a[axis], b[axis], tolerance);
}
inline bool sameBits(float a, float b)
{
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}
inline void exactRay(const Ray& a, const Ray& b)
{
    for (int axis = 0; axis < 3; ++axis)
        require(sameBits(a.origin[axis], b.origin[axis]) && sameBits(a.direction[axis], b.direction[axis]),
            "Pinhole ray was not preserved bit-for-bit");
}
inline void orthonormal(const cameraSampling::LensFrame& frame)
{
    near(glm::length(frame.forward), 1); near(glm::length(frame.right), 1); near(glm::length(frame.up), 1);
    near(glm::dot(frame.forward, frame.right), 0); near(glm::dot(frame.forward, frame.up), 0);
    near(glm::dot(frame.right, frame.up), 0);
    nearVector(glm::cross(frame.right, frame.up), -frame.forward);
}
inline glm::vec3 atDepth(const Ray& ray, float depth)
{
    return ray.origin + ((-depth - ray.origin.z) / ray.direction.z) * ray.direction;
}

template<class Probe> int runSuite(Probe probe)
{
    int passed = 0, failed = 0;
    auto test = [&](const char* name, const std::function<void()>& body) {
        try { body(); ++passed; std::cout << "[PASS] " << name << '\n'; }
        catch (const std::exception& e) { ++failed; std::cerr << "[FAIL] " << name << ": " << e.what() << '\n'; }
    };
    auto one = [&](const Query& q) { return probe(std::vector<Query>{q}).front(); };

    test("camera and render-state defaults disable finite aperture", [&] {
        Camera camera{}; RenderState state{};
        require(camera.lensRadius == 0 && camera.focalDistance > 0 && !state.enableDepthOfField, "Unsafe defaults");
    });
    test("axial frame has expected orientation", [&] {
        auto r = one(query()); require(r.frameOK, "Frame rejected"); orthonormal(r.frame);
        nearVector(r.frame.forward, glm::vec3(0, 0, -1));
        nearVector(r.frame.right, glm::vec3(1, 0, 0)); nearVector(r.frame.up, glm::vec3(0, 1, 0));
    });
    test("tilted and scaled axes produce a unit lens frame", [&] {
        auto q = query(); q.camera.view = glm::vec3(2, -3, -4); q.camera.up = glm::vec3(0, 7, 1);
        auto r = one(q); require(r.frameOK, "Tilted frame rejected"); orthonormal(r.frame);
        nearVector(r.frame.forward, glm::normalize(q.camera.view));
        q.camera.view *= 23; q.camera.up *= 0.25f;
        auto scaled = one(q); require(scaled.frameOK, "Scaled frame rejected");
        nearVector(r.frame.right, scaled.frame.right); nearVector(r.frame.up, scaled.frame.up);
    });
    test("frame normalization handles very small and large finite axes", [&] {
        for (float scale : {1e-30f, 1e30f}) {
            auto q = query(); q.camera.view *= scale; q.camera.up *= scale;
            auto r = one(q); require(r.frameOK && r.rayOK, "Finite scale rejected"); orthonormal(r.frame);
        }
    });
    test("degenerate and non-finite frames are rejected", [&] {
        std::vector<Query> qs(7, query());
        qs[0].camera.view = glm::vec3(0); qs[1].camera.up = glm::vec3(0);
        qs[2].camera.up = glm::vec3(0, 0, -5); qs[3].camera.up = glm::vec3(0, 0, 5);
        qs[4].camera.up = glm::vec3(0, 1e-8f, -1);
        qs[5].camera.view.x = std::numeric_limits<float>::infinity();
        qs[6].camera.up.x = std::numeric_limits<float>::quiet_NaN();
        auto rs = probe(qs);
        for (size_t i = 0; i < qs.size(); ++i) {
            require(!rs[i].frameOK && !rs[i].rayOK, "Invalid frame accepted"); exactRay(rs[i].ray, qs[i].pinhole);
        }
    });
    test("disk center radius and cardinal directions", [&] {
        const std::array<glm::vec2, 5> uv = {glm::vec2(0, 0.31f), glm::vec2(1, 0),
            glm::vec2(1, 0.25f), glm::vec2(1, 0.5f), glm::vec2(0.25f, 0.75f)};
        const std::array<glm::vec2, 5> expected = {glm::vec2(0), glm::vec2(1, 0),
            glm::vec2(0, 1), glm::vec2(-1, 0), glm::vec2(0, -0.5f)};
        for (size_t i = 0; i < uv.size(); ++i) {
            auto q = query(); q.sample = uv[i]; auto r = one(q);
            near(r.disk.x, expected[i].x); near(r.disk.y, expected[i].y);
        }
    });
    test("lens random stream is reproducible independent of query order", [&] {
        auto a = query(), b = query(); b.iteration = 937; b.pixel = 123456;
        auto rs = probe({a, b, b, a});
        for (auto pair : {std::pair<int, int>{0, 3}, {1, 2}}) {
            const auto& x = rs[pair.first]; const auto& y = rs[pair.second];
            require(x.randomWord == y.randomWord && sameBits(x.randomSample.x, y.randomSample.x)
                && sameBits(x.randomSample.y, y.randomSample.y), "Random stream depends on execution order");
        }
    });
    test("iteration and pixel each change the lens stream", [&] {
        auto a = query(), b = a, c = a; ++b.iteration; ++c.pixel;
        auto rs = probe({a, b, c});
        require(rs[0].randomWord != rs[1].randomWord && rs[0].randomWord != rs[2].randomWord,
            "Seed ignored iteration or pixel");
    });
    test("random seeds accept full unsigned input range", [&] {
        std::vector<Query> qs;
        for (uint32_t iter : {0u, 0x80000000u, 0xffffffffu})
            for (uint32_t pixel : {0u, 0x80000000u, 0xffffffffu}) {
                auto q = query(); q.iteration = iter; q.pixel = pixel; qs.push_back(q);
            }
        for (auto& r : probe(qs)) for (int axis = 0; axis < 2; ++axis)
            require(r.randomSample[axis] >= 0 && r.randomSample[axis] <= 1, "Invalid random sample");
    });
    test("32768 lens samples are uniform by area and angle", [&] {
        constexpr int count = 32768;
        std::vector<Query> qs(count, query());
        for (int i = 0; i < count; ++i) { qs[i].pixel = uint32_t(i); qs[i].iteration = 19; }
        auto rs = probe(qs);
        std::array<int, 4> rings{}, quadrants{};
        double meanX = 0, meanY = 0, meanR2 = 0;
        for (const auto& r : rs) {
            const float x = r.randomDisk.x, y = r.randomDisk.y, r2 = x*x + y*y;
            require(std::isfinite(r2) && r2 <= 1.000002f, "Sample outside disk");
            ++rings[(std::min)(3, int(r2 * 4))]; ++quadrants[(x < 0 ? 1 : 0) + (y < 0 ? 2 : 0)];
            meanX += x; meanY += y; meanR2 += r2;
        }
        near(float(meanX/count), 0, 0.012f); near(float(meanY/count), 0, 0.012f);
        near(float(meanR2/count), 0.5f, 0.012f);
        for (int n : rings) near(float(n)/count, 0.25f, 0.012f);
        for (int n : quadrants) near(float(n)/count, 0.25f, 0.012f);
        std::cout << "DISK moments: mean_x=" << meanX/count << " mean_y=" << meanY/count
            << " mean_radius_squared=" << meanR2/count << '\n';
    });
    test("off-axis focus is a plane at z=-6 rather than a sphere", [&] {
        auto q = query(); q.pinhole.direction = glm::vec3(0.6f, 0, -0.8f);
        for (glm::vec2 sample : {glm::vec2(0), glm::vec2(1, 0), glm::vec2(1, 0.25f), glm::vec2(0.7f, 0.9f)}) {
            q.sample = sample; auto r = one(q); require(r.rayOK, "Ray rejected");
            nearVector(atDepth(r.ray, 6), glm::vec3(4.5f, 0, -6));
            near(glm::length(r.ray.direction), 1);
        }
    });
    test("non-unit pinhole directions have the same focus", [&] {
        auto q = query(); q.pinhole.direction = glm::vec3(6, 0, -8);
        auto r = one(q); require(r.rayOK, "Non-unit ray rejected");
        nearVector(atDepth(r.ray, 6), glm::vec3(4.5f, 0, -6));
    });
    test("translated tilted rays stay inside lens and intersect the focus plane", [&] {
        std::vector<Query> qs;
        for (int i = 0; i < 65; ++i) {
            auto q = query(); q.camera.position = glm::vec3(13, -7, 2);
            q.camera.view = glm::vec3(2, -3, -4); q.camera.up = glm::vec3(0, 7, 1);
            q.camera.right = glm::vec3(99); // The helper must not depend on legacy right's scale.
            q.pinhole.origin = q.camera.position; q.pinhole.direction = glm::vec3(1, -1, -4);
            q.sample = glm::vec2(float(i)/64, float(i%13)/13); qs.push_back(q);
        }
        auto rs = probe(qs);
        const glm::vec3 v = glm::normalize(glm::vec3(2, -3, -4));
        const glm::vec3 expected(13.0f + 6.0f/(21.0f/std::sqrt(29.0f)),
            -7.0f - 6.0f/(21.0f/std::sqrt(29.0f)), 2.0f - 24.0f/(21.0f/std::sqrt(29.0f)));
        for (size_t i = 0; i < rs.size(); ++i) {
            const auto& ray = rs[i].ray; require(rs[i].rayOK, "Tilted ray rejected");
            const glm::vec3 offset = ray.origin - qs[i].camera.position;
            near(glm::dot(offset, v), 0);
            require(glm::length(offset) <= 0.30002f, "Lens radius depends on camera axis scale");
            const float t = (6 - glm::dot(offset, v)) / glm::dot(ray.direction, v);
            nearVector(ray.origin + t*ray.direction, expected, 6e-5f);
        }
    });
    test("ray bundle narrows at focus and expands behind it", [&] {
        auto q = query(); q.sample = glm::vec2(1, 0); auto r = one(q);
        require(r.rayOK, "Ray rejected");
        nearVector(r.ray.origin, glm::vec3(0.3f, 0, 0));
        nearVector(atDepth(r.ray, 3), glm::vec3(0.15f, 0, -3));
        nearVector(atDepth(r.ray, 6), glm::vec3(0, 0, -6));
        nearVector(atDepth(r.ray, 9), glm::vec3(-0.15f, 0, -9));
    });
    test("aperture radius scales lens offsets without moving focus", [&] {
        auto q = query(); q.sample = glm::vec2(1, 0); q.camera.lensRadius = 0.6f;
        auto r = one(q); require(r.rayOK, "Ray rejected");
        nearVector(atDepth(r.ray, 3), glm::vec3(0.3f, 0, -3));
        nearVector(atDepth(r.ray, 6), glm::vec3(0, 0, -6));
    });
    test("focal distance moves convergence without moving the lens", [&] {
        auto q = query(); q.sample = glm::vec2(1, 0); q.camera.focalDistance = 3;
        auto r = one(q); require(r.rayOK, "Ray rejected");
        nearVector(r.ray.origin, glm::vec3(0.3f, 0, 0));
        nearVector(atDepth(r.ray, 3), glm::vec3(0, 0, -3));
        nearVector(atDepth(r.ray, 6), glm::vec3(-0.3f, 0, -6));
    });
    test("disabled lens preserves the pinhole ray including signed zero", [&] {
        auto q = query(); q.enabled = false; q.pinhole.origin.x = -0.0f;
        q.pinhole.direction = glm::vec3(-0.0f, 0.1234567f, -7.12345f);
        q.camera.focalDistance = -1; q.sample.x = std::numeric_limits<float>::quiet_NaN();
        auto r = one(q); require(r.rayOK, "Disabled lens rejected"); exactRay(r.ray, q.pinhole);
    });
    test("zero aperture preserves the pinhole ray without normalization", [&] {
        auto q = query(); q.camera.lensRadius = 0; q.pinhole.direction = glm::vec3(0, 0, -13);
        q.camera.focalDistance = -1; q.sample = glm::vec2(-1);
        auto r = one(q); require(r.rayOK, "Zero lens rejected"); exactRay(r.ray, q.pinhole);
    });
    test("active lens rejects invalid aperture and focus", [&] {
        for (float value : {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            auto q = query(); q.camera.lensRadius = value;
            auto r = one(q); require(!r.rayOK, "Invalid aperture accepted"); exactRay(r.ray, q.pinhole);
            q = query(); q.camera.focalDistance = value; r = one(q);
            require(!r.rayOK, "Invalid focus accepted"); exactRay(r.ray, q.pinhole);
        }
        auto q = query(); q.camera.focalDistance = 0;
        require(!one(q).rayOK, "Zero focus accepted");
    });
    test("active lens rejects invalid square samples", [&] {
        for (glm::vec2 sample : {glm::vec2(-0.01f, 0.5f), glm::vec2(1.01f, 0.5f),
            glm::vec2(0.5f, -0.01f), glm::vec2(0.5f, 1.01f),
            glm::vec2(std::numeric_limits<float>::quiet_NaN(), 0)}) {
            auto q = query(); q.sample = sample; auto r = one(q);
            require(!r.rayOK, "Invalid sample accepted"); exactRay(r.ray, q.pinhole);
        }
    });
    test("active lens rejects zero backwards tangent and non-finite directions", [&] {
        for (glm::vec3 direction : {glm::vec3(0), glm::vec3(0, 0, 1), glm::vec3(1, 0, 0),
            glm::vec3(0, 0, -std::numeric_limits<float>::infinity())}) {
            auto q = query(); q.pinhole.direction = direction; auto r = one(q);
            require(!r.rayOK, "Invalid direction accepted"); exactRay(r.ray, q.pinhole);
        }
    });
    test("active lens rejects displaced pinhole origin and overflowing position", [&] {
        auto q = query(); q.pinhole.origin.x = 1;
        require(!one(q).rayOK, "Displaced pinhole accepted");
        q = query(); q.camera.position.x = FLT_MAX; q.pinhole.origin = q.camera.position;
        q.camera.lensRadius = FLT_MAX; q.sample = glm::vec2(1, 0);
        require(!one(q).rayOK, "Overflowing origin accepted");
    });

    std::cout << "Camera bytes=" << sizeof(Camera) << '\n';
    std::cout << "RESULT: " << passed << " passed, " << failed << " failed.\n";
    return failed ? 1 : 0;
}
} // namespace cameraTest
