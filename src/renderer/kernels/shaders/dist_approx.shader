#include "shader_macros.h"

#if defined(PLATFORM_METAL)
#include <metal_stdlib>
using namespace metal;
#endif

#include "cumath.h"
#include "raytracing_functions.h"
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"

// ============================================================================
// KERNEL: Distance Approximation (Half-Resolution)
// 
// Performs a coarse ray trace to find approximate distance to first hit.
// This is used as a starting point for the main GBuffer pass.
// ============================================================================

KERNEL(distApproximationKernel)(
    PARAM_TEXTURE_WRITE(texture2d<float, access::write>, distTex, 0),
    
    PARAM_CONSTANT(CameraData, camera, 0),
    PARAM_CONSTANT(FrameData, frame, 1),
    
    PARAM_TEXTURE_READ(texture3d<uint, access::read>, indirection, 2),
    PARAM_BUFFER(device SectorInfo*, sectorBuffer, 3),
    PARAM_BUFFER(device ulong*, occupancyBuffer, 4),
    PARAM_BUFFER(device uchar*, dataBuffer, 5),
    PARAM_BUFFER(device ulong*, sectorMaskBuffer, 6),
    PARAM_CONSTANT(CharacterGPUData*, charData, 7),
    
    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    CHECK_BOUNDS(distTex);
    uint width = distTex.get_width();
    uint height = distTex.get_height();
    uint2 gid = GET_GID();
#else
    CHECK_BOUNDS(_width, _height);
    int2 gid = GET_GID();
    int width = _width;
    int height = _height;
#endif

    float2 uv = (float2(gid) + 0.5f) / float2(width, height);
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    // Trace ray through voxel data
    hitInfo hit = trace(camera.position, dir, indirection, sectorBuffer, 
                        occupancyBuffer, dataBuffer, sectorMaskBuffer, 
                        frame.worldOrigin, charData);
    
    float dist = hit.hit ? length(hit.pos - camera.position) : 5000.0f;
    
    // Safety padding for the main raymarch
    dist = max(0.0f, dist - 8.0f);

    TEX_WRITE_2D(distTex, float4(dist, 0, 0, 0), gid);
}
