#pragma once
#include "ShaderTypes.h"

// Packed constant-buffer structs that match Slang's GlobalParams layout.
// Slang packs all `uniform` globals from a file into a single constant buffer.

struct GlobalParams {
    CameraData camera;
    FrameData frame;
};

struct MaterialGenParams {
    simd_int3 worldOrigin;
    uint32_t totalItems;
};
