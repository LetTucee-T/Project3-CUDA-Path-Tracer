#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment& path,
    glm::vec3 hitPoint,
    glm::vec3 normal,
    const Material& material,
    thrust::default_random_engine& rng, const glm::vec3* geometricNormal)
{
    glm::vec3 d = glm::normalize(path.ray.direction);
    glm::vec3 n = glm::normalize(normal);

    if (glm::dot(d, n) > 0.0f) {
        n = -n;
    }

    glm::vec3 offsetNormal = n;
    if (geometricNormal) {
        offsetNormal = glm::normalize(*geometricNormal);
        if (glm::dot(d, offsetNormal) > 0) offsetNormal = -offsetNormal;
        if (glm::dot(n, offsetNormal) < 0) n = -n;
        if (glm::dot(d, n) >= 0) n = offsetNormal;
    }
    glm::vec3 newDirection;
    bool whiteCoat = false;
    if (material.coatWeight > 0) {
        thrust::uniform_real_distribution<float> u01(0,1);
        whiteCoat = u01(rng) < material.coatWeight;
    }

    if (material.hasReflective > 0.0f || whiteCoat) {
        newDirection = glm::reflect(d, n);
    } else {
        newDirection =
            calculateRandomDirectionInHemisphere(n, rng);
    }
    // A normalized mixture q*white_delta + (1-q)*Lambertian. Sampling the
    // corresponding mixture weight cancels q or 1-q from f*cos/pdf.
    if (!whiteCoat) path.color *= material.color;
    // A shading normal must not send reflection through the geometric surface.
    if (geometricNormal && glm::dot(newDirection, offsetNormal) <= 0) path.color = glm::vec3(0);

    path.ray.origin = hitPoint + 1e-4f * offsetNormal;
    path.ray.direction = glm::normalize(newDirection);
}