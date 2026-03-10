#include "shader_macros.h"

#if defined(PLATFORM_METAL)
#include <metal_stdlib>
using namespace metal;
#endif

#include "cumath.h"
#include "tables.h"

// ============================================================================
// SHADER: Lookup Tables
// ============================================================================

// Ray Mask Optimization Lookup Table
// Used in GetStepPos for efficient empty-space skipping
constant uint64_t RayMaskOptimizationLUT[512] = {
    // LUT entries will be generated here
    // Placeholder - actual table defined in tables.h
};

// This kernel can be used to initialize GPU-side tables if needed
// For now, tables are defined as constant buffers
