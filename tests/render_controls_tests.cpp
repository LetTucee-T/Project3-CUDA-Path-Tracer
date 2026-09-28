#include "renderControls.h"
#include "ImGui/imgui.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct UI {
    RenderState state{};
    ImVec2 start; float rowHeight = 0, frameHeight = 0;
    static constexpr float fieldWidth = 180;
    UI() {
        ImGui::CreateContext();
        auto& io = ImGui::GetIO(); io.IniFilename = nullptr;
        io.DisplaySize = ImVec2(640, 480); io.DeltaTime = 1.0f / 60;
        unsigned char* pixels; int w, h; io.Fonts->GetTexDataAsRGBA32(&pixels, &w, &h);
        state.enableDepthOfField = true;
        state.camera.lensRadius = 0.1f; state.camera.focalDistance = 6;
        state.camera.position = glm::vec3(2, 3, 4);
        state.camera.view = glm::vec3(0.1f, -0.2f, -0.9f);
        state.camera.right = glm::vec3(0.8f, 0.1f, 0.2f);
        state.camera.up = glm::vec3(0, 2, 0);
        frame(); frame();
    }
    ~UI() { ImGui::DestroyContext(); }
    bool frame() {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10)); ImGui::SetNextWindowSize(ImVec2(550, 440));
        ImGui::Begin("Controls test", nullptr, ImGuiWindowFlags_NoSavedSettings);
        ImGui::PushItemWidth(fieldWidth);
        start = ImGui::GetCursorScreenPos(); frameHeight = ImGui::GetFrameHeight();
        rowHeight = ImGui::GetFrameHeightWithSpacing();
        const bool changed = drawRenderSettingsControls(state);
        ImGui::PopItemWidth(); ImGui::End(); ImGui::Render();
        return changed;
    }
    bool click(int row, float x) {
        auto& io = ImGui::GetIO();
        io.AddMousePosEvent(start.x + x, start.y + row * rowHeight + frameHeight * 0.5f);
        io.AddMouseButtonEvent(0, true); bool changed = frame();
        io.AddMouseButtonEvent(0, false); changed |= frame();
        return changed;
    }
    bool plus(int row) { return click(row, fieldWidth - frameHeight * 0.5f); }
    bool minus(int row) { return click(row, fieldWidth - frameHeight * 1.5f - ImGui::GetStyle().ItemInnerSpacing.x); }
    bool text(int row, const char* value) {
        click(row, 30);
        auto& io = ImGui::GetIO();
        io.AddKeyEvent(ImGuiKey_ModCtrl, true); io.AddKeyEvent(ImGuiKey_A, true); frame();
        io.AddKeyEvent(ImGuiKey_A, false); io.AddKeyEvent(ImGuiKey_ModCtrl, false); frame();
        io.AddInputCharactersUTF8(value); const bool changed = frame();
        io.AddKeyEvent(ImGuiKey_Enter, true); frame();
        io.AddKeyEvent(ImGuiKey_Enter, false); frame();
        return changed;
    }
};
void samePose(const Camera& a, const Camera& b) {
    for (const auto pair : {std::make_pair(a.position, b.position), std::make_pair(a.view, b.view),
        std::make_pair(a.up, b.up), std::make_pair(a.right, b.right)})
        require(std::memcmp(&pair.first, &pair.second, sizeof(glm::vec3)) == 0, "GUI moved the camera");
}
}
int main()
{
    int passed = 0, failed = 0;
    auto test = [&](const char* name, const std::function<void()>& body) {
        try { body(); ++passed; std::cout << "PASS " << name << '\n'; }
        catch (const std::exception& e) { ++failed; std::cerr << "FAIL " << name << ": " << e.what() << '\n'; }
    };
    test("idle controls do not reset accumulation", [] {
        UI ui; require(!ui.frame() && !ui.frame(), "Idle UI requests reset");
    });
    test("DOF toggle requests reset and preserves pose", [] {
        UI ui; const auto pose = ui.state.camera;
        require(ui.click(1, 6) && !ui.state.enableDepthOfField, "DOF checkbox did not update");
        require(!ui.frame(), "Reset remains set on idle frame"); samePose(pose, ui.state.camera);
    });
    test("lens radius editing requests reset and preserves pose", [] {
        UI ui; const auto pose = ui.state.camera;
        require(ui.plus(2) && std::abs(ui.state.camera.lensRadius - 0.11f) < 1e-6f, "Radius step failed");
        require(!ui.frame(), "Radius triggers repeated reset"); samePose(pose, ui.state.camera);
    });
    test("focus editing requests reset and preserves pose", [] {
        UI ui; const auto pose = ui.state.camera;
        require(ui.plus(3) && std::abs(ui.state.camera.focalDistance - 6.1f) < 1e-6f, "Focus step failed");
        samePose(pose, ui.state.camera);
    });
    test("disabled lens controls cannot alter parameters", [] {
        UI ui; ui.state.enableDepthOfField = false; ui.frame();
        require(!ui.plus(2) && ui.state.camera.lensRadius == 0.1f, "Disabled radius changed");
        require(!ui.plus(3) && ui.state.camera.focalDistance == 6, "Disabled focus changed");
    });
    test("step buttons reject negative radius and nonpositive focus", [] {
        UI ui; ui.state.camera.lensRadius = 0; ui.state.camera.focalDistance = 0.1f; ui.frame();
        require(!ui.minus(2) && ui.state.camera.lensRadius == 0, "Negative radius accepted");
        require(!ui.minus(3) && ui.state.camera.focalDistance == 0.1f, "Zero focus accepted");
    });
    test("typed valid parameters request reset", [] {
        UI ui;
        require(ui.text(2, "0.25") && ui.state.camera.lensRadius == 0.25f, "Typed radius failed");
        require(ui.text(3, "9.5") && ui.state.camera.focalDistance == 9.5f, "Typed focus failed");
    });
    test("typed invalid parameters preserve valid state", [] {
        UI ui;
        require(!ui.text(2, "-3") && ui.state.camera.lensRadius == 0.1f, "Negative typed radius accepted");
        require(!ui.text(3, "0") && ui.state.camera.focalDistance == 6, "Zero typed focus accepted");
        require(!ui.text(3, "1e39") && ui.state.camera.focalDistance == 6, "Infinite focus accepted");
    });
    test("existing controls still request reset and preserve pose", [] {
        UI ui; const auto pose = ui.state.camera;
        require(ui.click(0, 6) && ui.state.enableAntialiasing, "AA checkbox failed");
        require(ui.click(4, 6) && ui.state.enableStreamCompaction, "Compaction checkbox failed");
        require(ui.click(5, 6) && ui.state.enableMaterialSorting, "Sorting checkbox failed");
        require(ui.click(6, 6) && ui.state.enableBVH, "BVH checkbox failed");
        require(!ui.click(7, 6) && ui.state.enableMeshCulling, "Disabled AABB checkbox changed");
        samePose(pose, ui.state.camera);
    });
    std::cout << "RENDER CONTROLS RESULT: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
