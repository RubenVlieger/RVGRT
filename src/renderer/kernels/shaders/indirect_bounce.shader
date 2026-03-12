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
    PARAM_TEXTURE_WRITE(tex2d_f32_w, texRawIndirect, 0),
    PARAM_TEXTURE_READ(tex2d_f32_r, texNormal, 1),
    PARAM_TEXTURE_READ(tex2d_f32_r, texDepth, 2),
    
    PARAM_CONSTANT(CameraData, camera, 0),
    PARAM_CONSTANT(FrameData, frame, 1),
    
    PARAM_TEXTURE_READ(tex2d_arr_f32_s, textureAtlas, 8),
    PARAM_INDIRECTION(indirection, 3),
    PARAM_BUFFER(SectorInfo, sectorBuffer, 3),
    PARAM_BUFFER(ulong, occupancyBuffer, 4),
    PARAM_BUFFER(uchar, dataBuffer, 5),
    PARAM_BUFFER(ulong, sectorMaskBuffer, 6),
    PARAM_CONSTANT_PTR(CharacterGPUData, charData, 7),
    
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
#if defined(PLATFORM_METAL)
    TEX_WRITE_2D(texRawIndirect, float4(0.0f), gid);
#else
    ushort4 zeroHalf4 = make_ushort4(0, 0, 0, 0);
    TEX_WRITE_2D_RGBA16F(texRawIndirect, zeroHalf4, gid);
#endif
    return;
#endif

    float4 depthData = TEX_READ_2D(texDepth, gid);
#if defined(PLATFORM_METAL)
    float depth = depthData.r;
#else
    float depth = depthData.x;
#endif
    if (depth > 50000.0f) {
#if defined(PLATFORM_METAL)
        TEX_WRITE_2D(texRawIndirect, float4(0,0,0,0), gid);
#else
        ushort4 zeroHalf4 = make_ushort4(0, 0, 0, 0);
        TEX_WRITE_2D_RGBA16F(texRawIndirect, zeroHalf4, gid);
#endif
        return;
    }

#if defined(PLATFORM_METAL)
    half3 normal = HALF3_FROM_FLOAT3(TEX_READ_2D(texNormal, gid).rgb);
#else
    float4 normalData = TEX_READ_2D(texNormal, gid);
    half3 normal = HALF3_FROM_FLOAT3(make_float3(normalData.x, normalData.y, normalData.z));
#endif
    float2 uv = (AS_FLOAT2(gid) + 0.5f) / make_float2(width, height);
    float3 pos = reconstructPos(depth, uv, camera);

    uint voxelHash = hash3_to_1(INT3(pos.x * 1024.f, pos.y * 1024.f, pos.z * 1024.f));
    uint seed = voxelHash + uint(frame.time * 123456.0f);

    // Orthonormal Basis
    float3 N = (float3)normal;
    if (dot(N, N) < 0.5f) {
#if defined(PLATFORM_METAL)
        TEX_WRITE_2D(texRawIndirect, float4(0.02f, 0.02f, 0.02f, 1.0f), gid);
#else
        ushort4 valHalf4;
        valHalf4.x = __float2half_rn(0.02f);
        valHalf4.y = __float2half_rn(0.02f);
        valHalf4.z = __float2half_rn(0.02f);
        valHalf4.w = __float2half_rn(1.0f);
        TEX_WRITE_2D_RGBA16F(texRawIndirect, valHalf4, gid);
#endif
        return;
    }

    float3 helper = abs(N.x) > 0.99f ? FLOAT3(0,0,1) : FLOAT3(1,0,0);
    float3 Tangent = normalize(cross(N, helper));
    float3 Bitangent = cross(N, Tangent);

    // Hemisphere Sampling
    float r1 = rand_float(seed);
    float r2 = rand_float(seed);
    float phi = 2.0f * 3.14159f * r1;
    float cosTheta = sqrt(1.0f - r2);
    float sinTheta = sqrt(r2);
    float3 localDir = FLOAT3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
    float3 rayDir = normalize(localDir.x * Tangent + localDir.y * N + localDir.z * Bitangent);

    // Bounce Trace
    hitInfo hit = trace(pos + (float3)normal * 0.01f, rayDir, indirection, sectorBuffer,
                        occupancyBuffer, dataBuffer, sectorMaskBuffer, frame.worldOrigin, SECTOR_IND_X, SECTOR_IND_Y, SECTOR_IND_Z, charData);
    
    half3 incomingLight = HALF3(0.0f, 0.0f, 0.0f);

    if (hit.hit) {
#if SHADOWS
        bool isShadowed = traceShadow(hit.pos + (float3)hit.normal * 0.01f,
                                      frame.sunDirection,
                                      INDIRECT_SHADOW_MAXDIST,
                                      INDIRECT_SHADOW_STEPS,
                                      indirection, sectorBuffer, occupancyBuffer,
                                      dataBuffer, sectorMaskBuffer, frame.worldOrigin, SECTOR_IND_X, SECTOR_IND_Y, SECTOR_IND_Z, charData);
#else
        bool isShadowed = false;
#endif
        
        float distSq = (depth * depth) + dot(hit.pos - pos, hit.pos - pos);
        half3 bouncedAlbedo = sampleTexture(hit.uv, hit.matID, hit.normal, textureAtlas, distSq);
        
        half NdotL = max(dot(hit.normal, HALF3_FROM_FLOAT3(frame.sunDirection)), HALF_LITERAL(0.0f));
        half3 directLightAtHit = c_sunColor * NdotL * (isShadowed ? HALF_LITERAL(0.0f) : HALF_LITERAL(1.0f));
        
        incomingLight = (directLightAtHit * bouncedAlbedo) + (bouncedAlbedo * HALF_LITERAL(0.05f));
    } else {
        half3 skyLight = sampleSky(rayDir, frame.sunDirection);
#if defined(PLATFORM_METAL)
        float luma = dot((float3)skyLight, float3(0.3f, 0.59f, 0.11f));
#else
        float luma = skyLight.x * 0.3f + skyLight.y * 0.59f + skyLight.z * 0.11f;
#endif
        incomingLight = mix(skyLight, HALF3(luma, luma, luma), HALF_LITERAL(0.6f)) * HALF_LITERAL(0.25f);
    }
    
#if defined(PLATFORM_METAL)
    TEX_WRITE_2D(texRawIndirect, float4((float3)incomingLight, 1.0f), gid);
#else
    ushort4 incomingHalf4;
    incomingHalf4.x = __float2half_rn(incomingLight.x);
    incomingHalf4.y = __float2half_rn(incomingLight.y);
    incomingHalf4.z = __float2half_rn(incomingLight.z);
    incomingHalf4.w = __float2half_rn(1.0f);
    TEX_WRITE_2D_RGBA16F(texRawIndirect, incomingHalf4, gid);
#endif
}
