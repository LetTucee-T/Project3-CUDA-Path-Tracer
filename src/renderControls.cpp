#include "renderControls.h"
#include "ImGui/imgui.h"

#include <cmath>

bool drawRenderSettingsControls(RenderState& state)
{
    bool changed = ImGui::Checkbox("Antialiasing", &state.enableAntialiasing);
    changed |= ImGui::Checkbox("Depth of field", &state.enableDepthOfField);
    ImGui::BeginDisabled(!state.enableDepthOfField);
    float radius = state.camera.lensRadius;
    if (ImGui::InputFloat("Lens radius", &radius, 0.01f, 0.1f, "%.4f")
        && std::isfinite(radius) && radius >= 0.0f && radius != state.camera.lensRadius) {
        state.camera.lensRadius = radius;
        changed = true;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Aperture radius in world units. Zero gives a pinhole camera.");
    float distance = state.camera.focalDistance;
    if (ImGui::InputFloat("Focus distance", &distance, 0.1f, 1.0f, "%.4f")
        && std::isfinite(distance) && distance > 0.0f && distance != state.camera.focalDistance) {
        state.camera.focalDistance = distance;
        changed = true;
    }
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Distance along the camera's forward axis, in world units.");
    if (state.enableDepthOfField && state.camera.lensRadius == 0.0f)
        ImGui::TextUnformatted("Zero lens radius uses the pinhole camera.");

    changed |= ImGui::Checkbox("Stream Compaction", &state.enableStreamCompaction);
    changed |= ImGui::Checkbox("Material Sorting", &state.enableMaterialSorting);
    changed |= ImGui::Checkbox("Mesh BVH", &state.enableBVH);
    ImGui::BeginDisabled(state.enableBVH);
    changed |= ImGui::Checkbox("Mesh AABB Culling", &state.enableMeshCulling);
    ImGui::EndDisabled();
    if (state.enableBVH)
        ImGui::TextUnformatted("BVH includes hierarchical AABB culling.");
    return changed;
}
