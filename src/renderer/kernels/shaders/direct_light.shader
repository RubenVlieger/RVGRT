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
// KERNEL: GBuffer + Direct Light
// 
// Primary ray tracing kernel. Generates GBuffer (albedo, normal, depth, motion)
// and computes direct lighting with shadows and water reflections.
// ============================================================================

KERNEL(GBufferAndDirectLight)(
    PARAM_TEXTURE_WRITE(texture2d<float, access::write>, texDirectLight, 0),
    PARAM_TEXTURE_WRITE(texture2d<float, access::write>, texAlbedo, 1),
    PARAM_TEXTURE_WRITE(texture2d<float, access::write>, texNormal, 2),
    PARAM_TEXTURE_WRITE(texture2d<float, access::write>, texMotion, 3),
    PARAM_TEXTURE_WRITE(texture2d<float, access::write>, texDepth, 4),

    PARAM_CONSTANT(CameraData, camera, 0),
    PARAM_CONSTANT(FrameData, frame, 1),

    PARAM_TEXTURE_READ(texture3d<uint, access::read>, indirection, 5),
    PARAM_BUFFER(device SectorInfo*, sectorBuffer, 3),
    PARAM_BUFFER(device ulong*, occupancyBuffer, 4),
    PARAM_BUFFER(device uchar*, dataBuffer, 5),
    PARAM_BUFFER(device ulong*, sectorMaskBuffer, 6),
    PARAM_CONSTANT(CharacterGPUData*, charData, 7),
    
    PARAM_TEXTURE_READ(texture2d_array<float, access::sample>, textureAtlas, 8),
    PARAM_TEXTURE_READ(texture2d<float, access::sample>, halfDistTex, 9),

    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    CHECK_BOUNDS(texDirectLight);
    uint2 gid = GET_GID();
    int width = texDirectLight.get_width();
    int height = texDirectLight.get_height();
#else
    CHECK_BOUNDS(_width, _height);
    int2 gid = GET_GID();
    int width = _width;
    int height = _height;
#endif

    // Ray Gen
    float2 pixelCenter = float2(gid) + 0.5f;
    float2 jitteredCoord = pixelCenter + camera.jitter;
    float2 uv = jitteredCoord / float2(width, height);
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    // Read Dist Approx
    DECLARE_SAMPLER(sLinear, linear, clampededge);
    float startDist = TEX_SAMPLE_2D(halfDistTex, uv).r;

    hitInfo hit = trace(camera.position + startDist * dir, dir, indirection, sectorBuffer,
                        occupancyBuffer, dataBuffer, sectorMaskBuffer, frame.worldOrigin, charData);

    float depth = 100000.0f;
    half3 irradiance = half3(0.0h);
    half3 albedo = half3(0.0h);
    half3 normal = half3(0.0h);

    if (hit.hit) {
        depth = length(hit.pos - camera.position);
        normal = hit.normal;

        // Motion Vectors
        float2 motionVector = float2(0.0f);
        if (depth < 50000.0f) {
            float4 currentClipPos = camera.unjitteredViewProjection * float4(hit.pos, 1.0f);
            float4 previousClipPos = camera.prevUnjitteredViewProjection * float4(hit.pos, 1.0f);
            if (previousClipPos.w > 0.0f && currentClipPos.w > 0.0f) {
                float2 prevNDC = previousClipPos.xy / previousClipPos.w;
                float2 currNDC = currentClipPos.xy / currentClipPos.w;
                motionVector = 0.5f * (currNDC - prevNDC);
                motionVector.y = -motionVector.y;
            }
        }
        TEX_WRITE_2D(texMotion, float4(motionVector.x, motionVector.y, 0, 0), gid);

        // Water Logic
        bool isWater = (hit.pos.y <= 3.001f && normal.y > 0.8h);
        if (isWater) {
            albedo = sampleTexture(hit.uv, hit.matID, hit.normal, textureAtlas, depth * depth);
            float nx = fbm3D(hit.pos.x, hit.pos.z, frame.time, 3, 0.06f, 2.0f, 0.6f);
            float ny = fbm3D(hit.pos.z, hit.pos.x, frame.time + 112.0f, 3, 0.06f, 2.0f, 0.6f);
            half3 distNormal = normalize(normal + half3(half(nx)*0.1h, 0.0h, half(ny)*0.1h));
            
            float3 reflDir = reflect(dir, (float3)distNormal);
            
#if REFLECTIONS
            hitInfo reflHit = trace(hit.pos, reflDir, indirection, sectorBuffer, occupancyBuffer,
                                   dataBuffer, sectorMaskBuffer, frame.worldOrigin, charData);
#else
            hitInfo reflHit;
            reflHit.hit = false;
            reflHit.pos = hit.pos;
#endif
            
            half3 reflectColor;
#if REFLECTIONS
            if (reflHit.hit) {
                float distSq = dot(reflHit.pos - hit.pos, reflHit.pos - hit.pos);
                half3 rAlbedo = sampleTexture(reflHit.uv, reflHit.matID, reflHit.normal, textureAtlas, distSq);
                
#if SHADOWS
                bool rShadow = traceShadow(reflHit.pos + (float3)reflHit.normal * 0.01f,
                                           frame.sunDirection, REFLECTION_SHADOW_MAXDIST, REFLECTION_SHADOW_STEPS,
                                           indirection, sectorBuffer, occupancyBuffer, dataBuffer,
                                           sectorMaskBuffer, frame.worldOrigin, charData);
#else
                bool rShadow = false;
#endif
                reflectColor = rAlbedo * (rShadow ? 0.05h : (half3)c_sunColor);
            } else {
                reflectColor = sampleSky(reflDir, frame.sunDirection);
            }
#else
            reflectColor = sampleSky(reflDir, frame.sunDirection);
#endif
            
            float3 viewDir = -dir;
            float3 halfVec = normalize(viewDir + frame.sunDirection);
            float NdotH = max(dot((float3)distNormal, halfVec), 0.0f);
            half specular = pow(NdotH, 512.0f) * 4.0f;
            half NdotV = max(dot(distNormal, (half3)viewDir), 0.0h);
            half fresnel = 0.02h + (0.98h) * pow(1.0h - NdotV, 5.0h);
            
#if SHADOWS
            bool waterShadow = traceShadow(reflHit.pos, frame.sunDirection, WATER_SHADOW_MAXDIST,
                                          WATER_SHADOW_STEPS, indirection, sectorBuffer, occupancyBuffer,
                                          dataBuffer, sectorMaskBuffer, frame.worldOrigin, charData);
#else
            bool waterShadow = false;
#endif
            
            irradiance = (reflectColor * fresnel) + (c_sunColor * specular * (waterShadow ? 0.0h : 1.0h));
            irradiance /= (albedo + 0.001h);
        } else {
            // Standard Solid Block
            albedo = sampleTexture(hit.uv, hit.matID, hit.normal, textureAtlas, depth * depth);

#if SHADOWS
            bool isShadowed = traceShadow(hit.pos + (float3)normal * 0.005f, frame.sunDirection,
                                          SHADOW_MAXDIST, SHADOW_STEPS, indirection, sectorBuffer,
                                          occupancyBuffer, dataBuffer, sectorMaskBuffer, frame.worldOrigin, charData);
#else
            bool isShadowed = false;
#endif
            
            half NdotL = max(dot(normal, (half3)frame.sunDirection), 0.0h);
            irradiance = c_sunColor * NdotL * (isShadowed ? 0.02h : 1.0h);
        }
    } else {
        // Sky
        irradiance = sampleSky(dir, frame.sunDirection);
        albedo = half3(1.0h);
        
        float3 fakePos = camera.position + dir * 1000.0f;
        float4 currentClipPos = camera.unjitteredViewProjection * float4(fakePos, 1.0f);
        float4 previousClipPos = camera.prevUnjitteredViewProjection * float4(fakePos, 1.0f);
        float2 mv = float2(0.0);
        if (previousClipPos.w > 0.0f && currentClipPos.w > 0.0f) {
            float2 prevNDC = previousClipPos.xy / previousClipPos.w;
            float2 currNDC = currentClipPos.xy / currentClipPos.w;
            mv = 0.5f * (currNDC - prevNDC);
            mv.y = -mv.y;
        }
        TEX_WRITE_2D(texMotion, float4(mv.x, mv.y, 0, 0), gid);
    }

    TEX_WRITE_2D(texDirectLight, float4((float3)irradiance, 1.0f), gid);
    TEX_WRITE_2D(texAlbedo, float4((float3)albedo, 1.0f), gid);
    TEX_WRITE_2D(texNormal, float4(((float3)normal), 1.0f), gid);
    TEX_WRITE_2D(texDepth, float4(depth), gid);
}
