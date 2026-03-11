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
// KERNEL: Volumetric Fog
// 
// Ray-marches through the scene to accumulate volumetric lighting.
// Uses temporal reprojection for stability.
// ============================================================================

KERNEL(VolumetricFog)(
    PARAM_TEXTURE_WRITE(tex2d_f32_w, texVolumetric, 0),
    PARAM_TEXTURE_READ(tex2d_f32_r, texDepth, 1),
    PARAM_TEXTURE_READ(tex2d_f32_s, texHistory, 2),
    
    PARAM_CONSTANT(CameraData, camera, 0),
    PARAM_CONSTANT(FrameData, frame, 1),
    
    PARAM_INDIRECTION(indirection, 3),
    PARAM_BUFFER(SectorInfo, sectorBuffer, 3),
    PARAM_BUFFER(ulong, occupancyBuffer, 4),
    PARAM_BUFFER(ulong, sectorMaskBuffer, 6),
    PARAM_CONSTANT(CharacterGPUData, charData, 7),

    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    CHECK_BOUNDS(texVolumetric);
    int width = texVolumetric.get_width();
    int height = texVolumetric.get_height();
#else
    CHECK_BOUNDS(_width, _height);
    int2 gid = GET_GID();
    int width = _width;
    int height = _height;
#endif

#if !VOLUMETRIC_FOG
#if defined(PLATFORM_METAL)
    TEX_WRITE_2D(texVolumetric, float4(0.0f), gid);
#else
    float4 zeroVal = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    TEX_WRITE_2D(texVolumetric, zeroVal, gid);
#endif
    return;
#endif

    // Half-resolution coordinate
#if defined(PLATFORM_METAL)
    uint2 fullResCoord = gid * 2;
#else
    uint2 fullResCoord = make_uint2(gid.x * 2, gid.y * 2);
#endif

    // Clamp to valid depth coord
    if (fullResCoord.x >= (unsigned int)width * 2) fullResCoord.x = (unsigned int)width * 2 - 1;
    if (fullResCoord.y >= (unsigned int)height * 2) fullResCoord.y = (unsigned int)height * 2 - 1;

#if defined(PLATFORM_METAL)
    float depth = TEX_READ_2D(texDepth, fullResCoord).r;
#else
    float4 depth4 = TEX_READ_2D(texDepth, fullResCoord);
    float depth = depth4.x;
#endif

#if defined(PLATFORM_METAL)
    float2 uv = (float2(gid) + 0.5f) / float2(width, height);
#else
    float2 uv = make_float2((gid.x + 0.5f) / (float)width, (gid.y + 0.5f) / (float)height);
#endif

    float fogMaxDist = VOLUMETRIC_MAXDIST;
    float clampedDepth = fminf(depth, fogMaxDist);
    
    float3 endPos = reconstructPos(clampedDepth, uv, camera);
    float3 startPos = camera.position;
    
    float3 rayVec = endPos - startPos;
    float rayLength = length(rayVec);
    float3 rayDir = normalize(rayVec);

    // Ray marching with dithering
#if defined(PLATFORM_METAL)
    float dither = fract(52.9829189f * fract(0.06711056f * (float2(gid).x + frame.time * 5.588f) + 0.00583715f * (float2(gid).y)));
#else
    float noisePos = (gid.x + 0.5f) * 0.06711056f + frame.time * 5.588f + (gid.y + 0.5f) * 0.00583715f;
    float dither = fmodf(52.9829189f * fmodf(noisePos, 1.0f), 1.0f);
#endif
    
    const int STEPS = VOLUMETRIC_STEPS;
    float stepSize = rayLength / (float)STEPS;
    float currentT = stepSize * dither;
    
#if defined(PLATFORM_METAL)
    float3 accumulatedLight = float3(0.0f);
#else
    float3 accumulatedLight = make_float3(0.0f, 0.0f, 0.0f);
#endif
    float accumulatedTransmittance = 1.0f;
    
    const float density = FOG_DENSITY;
#if defined(PLATFORM_METAL)
    const float3 fogColor = float3(FOG_COLOR);
#else
    const float3 fogColor = make_float3(0.6f, 0.7f, 0.8f);
#endif
    const float anisotropy = FOG_ANISOTROPY;
    
#if defined(PLATFORM_METAL)
    float cosTheta = dot(rayDir, frame.sunDirection);
#else
    float cosTheta = rayDir.x * frame.sunDirection.x + rayDir.y * frame.sunDirection.y + rayDir.z * frame.sunDirection.z;
#endif
    float denom = 1.0f + anisotropy * anisotropy - 2.0f * anisotropy * cosTheta;
    float phase = (1.0f - anisotropy * anisotropy) / (4.0f * 3.14159f * powf(denom, 1.5f));
    
#if defined(PLATFORM_METAL)
    float3 sunColor = float3(c_sunColor);
#else
    half3 sunColorHalf = c_sunColor;
    float3 sunColor = make_float3((float)sunColorHalf.x, (float)sunColorHalf.y, (float)sunColorHalf.z);
#endif

    for(int i = 0; i < STEPS; i++) {
        float3 pos = startPos + rayDir * currentT;
        
        bool isShadowed = false;
        if (currentT > 2.0f && currentT < 200.0f && phase > 0.04f) {
#if SHADOWS
            isShadowed = traceShadow(pos, frame.sunDirection, VOLUMETRIC_SHADOW_MAXDIST,
                                    VOLUMETRIC_SHADOW_STEPS, indirection, sectorBuffer,
                                    occupancyBuffer, (uchar*)0, sectorMaskBuffer, frame.worldOrigin, IND_X, IND_Y, IND_Z, &charData);
#else
            isShadowed = false;
#endif
        }

        if (!isShadowed) {
            accumulatedLight = accumulatedLight + sunColor * phase * density * accumulatedTransmittance * stepSize;
        }
        accumulatedLight = accumulatedLight + fogColor * 0.05f * density * accumulatedTransmittance * stepSize;
        accumulatedTransmittance *= expf(-density * stepSize);
        currentT += stepSize;
    }

    // Reprojection
#if defined(PLATFORM_METAL)
    float4 prevClip = camera.prevUnjitteredViewProjection * float4(endPos, 1.0f);
#else
    float4 pos4 = make_float4(endPos.x, endPos.y, endPos.z, 1.0f);
    float4 prevClip = camera.prevUnjitteredViewProjection * pos4;
#endif
    
#if defined(PLATFORM_METAL)
    float3 history = float3(0.0f);
#else
    float3 history = make_float3(0.0f, 0.0f, 0.0f);
#endif
    float blendFactor = 0.0f;

    if (prevClip.w > 0.01f) {
        float2 prevNDC;
        prevNDC.x = prevClip.x / prevClip.w;
        prevNDC.y = prevClip.y / prevClip.w;
        
        float2 prevUV;
        prevUV.x = prevNDC.x * 0.5f + 0.5f;
        prevUV.y = 0.5f - prevNDC.y * 0.5f;

        if (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f) {
            DECLARE_SAMPLER(sLinear, linear, clamp_to_edge);
#if defined(PLATFORM_METAL)
            history = TEX_SAMPLE_2D(texHistory, prevUV).rgb;
#else
            float4 history4 = TEX_SAMPLE_2D(texHistory, prevUV);
            history = make_float3(history4.x, history4.y, history4.z);
#endif
            
#if defined(PLATFORM_METAL)
            float diff = length(history - accumulatedLight);
#else
            float diff = length(history - accumulatedLight);
#endif
            blendFactor = 0.8f;
            if (diff > 1.0f) blendFactor = 0.4f;
        }
    }

#if defined(PLATFORM_METAL)
    float3 result = mix(accumulatedLight, history, blendFactor);
    TEX_WRITE_2D(texVolumetric, float4(result, 1.0f), gid);
#else
    float3 result = mix(accumulatedLight, history, blendFactor);
    float4 result4 = make_float4(result.x, result.y, result.z, 1.0f);
    TEX_WRITE_2D(texVolumetric, result4, gid);
#endif
}
