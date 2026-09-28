#pragma once

#include "sceneStructs.h"

// Draw controls in the current ImGui window. True requests a fresh accumulation;
// it does not move the camera or alter its projection axes.
bool drawRenderSettingsControls(RenderState& state);
