#include "shader_macros.h"

#if defined(PLATFORM_METAL)
#include <metal_stdlib>
using namespace metal;
#endif

#include "cumath.h"
#include "raytracing_functions.h"
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
#include "renderer/shader_settings.h"

// ============================================================================
// KERNEL: Indirect Bounce (1-bounce global illumination)
// 
// Samples a random direction in the hemisphere and traces a bounce ray.
// Combines with direct lighting at hit point for approximate GI.
// ============================================================================

KERNEL(IndirectBounce)(
#if defined(PLATFORM_METAL)
    PARAM_TEXTURE_WRITE(tex2d_f32_w, texRawIndirect, 0),
    PARAM_TEXTURE_READ(tex2d_f32_r, texNormal, 1),
    PARAM_TEXTURE_READ(tex2d_f32_r, texDepth, 2),
#else
    PARAM_TEXTURE_WRITE(texture2d<float, access::write>, texRawIndirect, 0),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texNormal, 1),
    PARAM_TEXTURE_READ(texture2d<float, access::read>, texDepth, 2),
#endif
    
    PARAM_CONSTANT(CameraData, camera, 0),
    PARAM_CONSTANT(FrameData, frame, 1),
    
#if defined(PLATFORM_METAL)
    PARAM_TEXTURE_READ(tex2d_arr_f32_s, textureAtlas, 8),
    PARAM_TEXTURE_READ(tex3d_u32, indirection, 3),
#else
    PARAM_TEXTURE_READ(texture2d_array<float, access::sample>, textureAtlas, 8),
    PARAM_TEXTURE_READ(texture3d<uint, access::read>, indirection, 3),
#endif
    PARAM_BUFFER(SectorInfo, sectorBuffer, 3),
    PARAM_BUFFER(ulong, occupancyBuffer, 4),
    PARAM_BUFFER(uchar, dataBuffer, 5),
    PARAM_BUFFER(ulong, sectorMaskBuffer, 6),
    PARAM_CONSTANT(CharacterGPUData, charData, 7),
    
    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    CHECK_BOUNDS(texRawIndirect);
    int width = texRawIndirect.get_width();
    int height = texRawIndirect.get_height();
#else
    CHECK_BOUNDS(_width, _height);
    int2 gid = GET_GID();
    int width = _width;
    int height = _height;
#endif

#if !INDIRECT_LIGHTING
    TEX_WRITE_2D(texRawIndirect, float4(0.0f), gid);
    return;
#endif

    float depth = TEX_READ_2D(texDepth, gid).r;
    if (depth > 50000.0f) {
        TEX_WRITE_2D(texRawIndirect, float4(0,0,0,0), gid);
        return;
    }

    half3 normal = (half3)TEX_READ_2D(texNormal, gid).rgb;
    float2 uv = (float2(gid) + 0.5f) / float2(width, height);
    float3 pos = reconstructPos(depth, uv, camera);

    uint voxelHash = hash3_to_1(int3(pos * 1024.f));
    uint seed = voxelHash + uint(frame.time * 123456.0f);

    // Orthonormal Basis
    float3 N = (float3)normal;
    if (dot(N, N) < 0.5f) {
        TEX_WRITE_2D(texRawIndirect, float4(0.02f, 0.02f, 0.02f, 1.0f), gid);
        return;
    }

    float3 helper = abs(N.x) > 0.99f ? float3(0,0,1) : float3(1,0,0);
    float3 Tangent = normalize(cross(N, helper));
    float3 Bitangent = cross(N, Tangent);

    // Hemisphere Sampling
    float r1 = rand_float(seed);
    float r2 = rand_float(seed);
    float phi = 2.0f * 3.14159f * r1;
    float cosTheta = sqrt(1.0f - r2);
    float sinTheta = sqrt(r2);
    float3 localDir = float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
    float3 rayDir = normalize(localDir.x * Tangent + localDir.y * N + localDir.z * Bitangent);

    // Bounce Trace
    hitInfo hit = trace(pos + (float3)normal * 0.01f, rayDir, indirection, sectorBuffer,
                        occupancyBuffer, dataBuffer, sectorMaskBuffer, frame.worldOrigin, IND_X, IND_Y, IND_Z, &charData);
    
    half3 incomingLight = half3(0.0h);

    if (hit.hit) {
#if SHADOWS
        bool isShadowed = traceShadow(hit.pos + (float3)hit.normal * 0.01f,
                                      frame.sunDirection,
                                      INDIRECT_SHADOW_MAXDIST,
                                      INDIRECT_SHADOW_STEPS,
                                      indirection, sectorBuffer, occupancyBuffer,
                                      dataBuffer, sectorMaskBuffer, frame.worldOrigin, IND_X, IND_Y, IND_Z, &charData);
#else
        bool isShadowed = false;
#endif
        
        float distSq = (depth * depth) + dot(hit.pos - pos, hit.pos - pos);
        half3 bouncedAlbedo = sampleTexture(hit.uv, hit.matID, hit.normal, textureAtlas, distSq);
        
        half NdotL = max(dot(hit.normal, (half3)frame.sunDirection), 0.0h);
        half3 directLightAtHit = c_sunColor * NdotL * (isShadowed ? 0.0h : 1.0h);
        
        incomingLight = (directLightAtHit * bouncedAlbedo) + (bouncedAlbedo * 0.05h);
    } else {
        half3 skyLight = sampleSky(rayDir, frame.sunDirection);
        float luma = dot((float3)skyLight, float3(0.3f, 0.59f, 0.11f));
        incomingLight = mix(skyLight, half3(luma), 0.6h) * 0.25h;
    }
    
    TEX_WRITE_2D(texRawIndirect, float4((float3)incomingLight, 1.0f), gid);
}
