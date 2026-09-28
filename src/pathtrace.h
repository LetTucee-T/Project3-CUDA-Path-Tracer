#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

#ifdef PATHTRACE_TESTING
#include "intersections.h"
// Read-only access to the actual renderer allocations for GPU upload probes.
BVHDeviceView pathtraceBVHForTesting();
const Geom* pathtraceGeomsForTesting();
#endif
