#pragma once
#include "sceneStructs.h"
#include <cmath>

namespace cameraOrbit {
// Use eye-minus-target, retaining the azimuth sign in all four quadrants.
inline void angles(const Camera& camera, float& phi, float& theta, float& radius) {
    const glm::vec3 offset = camera.position - camera.lookAt;
    radius = glm::length(offset);
    phi = atan2f(offset.x, offset.z);
    theta = acosf(glm::clamp(offset.y / radius, -1.0f, 1.0f));
}
inline void apply(Camera& camera, float phi, float theta, float radius) {
    const glm::vec3 offset(radius * sinf(phi) * sinf(theta), radius * cosf(theta),
                           radius * cosf(phi) * sinf(theta));
    camera.position = offset + camera.lookAt;
    camera.view = -glm::normalize(offset);
    glm::vec3 right = glm::cross(camera.view, glm::vec3(0, 1, 0));
    // A vertical view has no unique world-up azimuth; use the previous frame.
    if (glm::dot(right, right) < 1e-12f) right = camera.right;
    camera.right = glm::normalize(right);
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));
}
}
