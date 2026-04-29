#pragma once

// =================================================================================
// RVGRT GLOBAL SHADER SETTINGS
// Shared between C++ (CPU) and Metal/CUDA (GPU)
// =================================================================================

// ---------------------------------------------------------------------------------
// FEATURE TOGGLES
// ---------------------------------------------------------------------------------
#define VOLUMETRIC_FOG 1
#define INDIRECT_LIGHTING 1
#define REFLECTIONS 1
#define SHADOWS 1

#define CHARACTER_MODELS 1
#define USE_METALFX 0

// ---------------------------------------------------------------------------------
// SHADOW CASTING SETTINGS (Distance & Quality)
// ---------------------------------------------------------------------------------
// Standard primary ray shadows
#define SHADOW_MAXDIST 256.0f
#define SHADOW_STEPS 64

// Water specific shadows (often needs more steps due to reflection angles)
#define WATER_SHADOW_MAXDIST 128.0f
#define WATER_SHADOW_STEPS 32

// Reflection shadows (traced from the reflection hit point)
#define REFLECTION_SHADOW_MAXDIST 64.0f
#define REFLECTION_SHADOW_STEPS 16

// Indirect illumination shadows (how far bounced light respects occlusion)
#define INDIRECT_SHADOW_MAXDIST 100.0f
#define INDIRECT_SHADOW_STEPS 16

// Volumetric sun shafts / god rays shadow precision
#define VOLUMETRIC_SHADOW_MAXDIST 200.0f
#define VOLUMETRIC_SHADOW_STEPS 32

// ---------------------------------------------------------------------------------
// VOLUMETRIC FOG SETTINGS
// ---------------------------------------------------------------------------------
#define VOLUMETRIC_MAXDIST 300.0f
#define VOLUMETRIC_STEPS 8
#define FOG_DENSITY 0.005f
#define FOG_COLOR float3(0.6f, 0.7f, 0.8f)
#define FOG_ANISOTROPY                                                         \
  0.6f // 0 = Isotropic, closer to 1 = Stronger God Rays forward scattering

// ---------------------------------------------------------------------------------
// POST-PROCESSING & COLOR GRADING
// ---------------------------------------------------------------------------------
// Saturation adjustments
#define IMAGE_SATURATION 1.4f
#define SKY_IMAGE_SATURATION 1.05f

// Depth fog applied in composite pass
#define COMPOSITE_FOG_START 60.0f
#define COMPOSITE_FOG_DENSITY 0.0002f
#define COMPOSITE_FOG_COLOR float3(0.5f, 0.7f, 0.9f)

// Indirect bounce ray limits
#define INDIRECT_BOUNCE_MAX_ITERS 256

// Distance approximation ray limits and LOD mode
#define DIST_APPROX_MAX_ITERS 256

// Character model culling
#define CHARACTER_MAX_TRACE_DIST 500.0f
#define CHARACTER_MAX_TRACE_DIST_SQ                                            \
  (CHARACTER_MAX_TRACE_DIST * CHARACTER_MAX_TRACE_DIST)
