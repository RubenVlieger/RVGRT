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
    PARAM_TEXTURE_WRITE(tex2d_f32_w, texDirectLight, 0),
    PARAM_TEXTURE_WRITE(tex2d_f32_w, texAlbedo, 1),
    PARAM_TEXTURE_WRITE(tex2d_f32_w, texNormal, 2),
    PARAM_TEXTURE_WRITE(tex2d_f32_w, texMotion, 3),
    PARAM_TEXTURE_WRITE(tex2d_f32_w, texDepth, 4),

    PARAM_CONSTANT(CameraData, camera, 0),
    PARAM_CONSTANT(FrameData, frame, 1),

    PARAM_TEXTURE_READ(tex3d_u32, indirection, 5),
    PARAM_BUFFER(SectorInfo, sectorBuffer, 3),
    PARAM_BUFFER(ulong, occupancyBuffer, 4),
    PARAM_BUFFER(uchar, dataBuffer, 5),
    PARAM_BUFFER(ulong, sectorMaskBuffer, 6),
    PARAM_CONSTANT(CharacterGPUData, charData, 7),
    
    PARAM_TEXTURE_READ(tex2d_arr_f32_s, textureAtlas, 8),
    PARAM_TEXTURE_READ(tex2d_f32_s, halfDistTex, 9),

    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    CHECK_BOUNDS(texDirectLight);
    int width = texDirectLight.get_width();
    int height = texDirectLight.get_height();
#else
    CHECK_BOUNDS(_width, _height);
    int2 gid = GET_GID();
    int width = _width;
    int height = _height;
#endif

    // Ray Gen
    float2 pixelCenter = AS_FLOAT2(gid) + 0.5f;
    float2 jitteredCoord = pixelCenter + camera.jitter;
    float2 uv = jitteredCoord / make_float2(width, height);
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    // Read Dist Approx
    DECLARE_SAMPLER(sLinear, linear, clamp_to_edge);
    float startDist = TEX_SAMPLE_2D(halfDistTex, uv).r;

    hitInfo hit = trace(camera.position + startDist * dir, dir, indirection, sectorBuffer,
                        occupancyBuffer, dataBuffer, sectorMaskBuffer, frame.worldOrigin, IND_X, IND_Y, IND_Z, &charData);

    float depth = 100000.0f;
    half3 irradiance = HALF3(0.0f, 0.0f, 0.0f);
    half3 albedo = HALF3(0.0f, 0.0f, 0.0f);
    half3 normal = HALF3(0.0f, 0.0f, 0.0f);

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
        } else {
            // Sky pixels have zero motion
            motionVector = float2(0.0f, 0.0f);
        }
        TEX_WRITE_2D(texMotion, float4(motionVector.x, motionVector.y, 0.0f, 0.0f), gid);

        // Water Logic
        bool isWater = (hit.pos.y <= 3.001f && normal.y > HALF_LITERAL(0.8f));
        if (isWater) {
            albedo = sampleTexture(hit.uv, hit.matID, hit.normal, textureAtlas, depth * depth);
            float nx = fbm3D(hit.pos.x, hit.pos.z, frame.time, 3, 0.06f, 2.0f, 0.6f);
            float ny = fbm3D(hit.pos.z, hit.pos.x, frame.time + 112.0f, 3, 0.06f, 2.0f, 0.6f);
            half3 distNormal = normalize(normal + HALF3(HALF_LITERAL(nx*0.1f), HALF_LITERAL(0.0f), HALF_LITERAL(ny*0.1f)));
            
            float3 reflDir = reflect(dir, (float3)distNormal);
            
#if REFLECTIONS
            hitInfo reflHit = trace(hit.pos, reflDir, indirection, sectorBuffer, occupancyBuffer,
                                   dataBuffer, sectorMaskBuffer, frame.worldOrigin, IND_X, IND_Y, IND_Z, &charData);
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
                                           sectorMaskBuffer, frame.worldOrigin, IND_X, IND_Y, IND_Z, &charData);
#else
                bool rShadow = false;
#endif
                reflectColor = rAlbedo * (rShadow ? HALF3(0.05f, 0.05f, 0.05f) : HALF3_FROM_FLOAT3(c_sunColor));
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
            half NdotV = max(dot(distNormal, HALF3_FROM_FLOAT3(viewDir)), HALF_LITERAL(0.0f));
            half fresnel = HALF_LITERAL(0.02f) + HALF_LITERAL(0.98f) * pow(HALF_LITERAL(1.0f) - NdotV, HALF_LITERAL(5.0f));
            
#if SHADOWS
            bool waterShadow = traceShadow(reflHit.pos, frame.sunDirection, WATER_SHADOW_MAXDIST,
                                          WATER_SHADOW_STEPS, indirection, sectorBuffer, occupancyBuffer,
                                          dataBuffer, sectorMaskBuffer, frame.worldOrigin, IND_X, IND_Y, IND_Z, &charData);
#else
            bool waterShadow = false;
#endif
            
            irradiance = (reflectColor * fresnel) + (HALF3_FROM_FLOAT3(c_sunColor) * specular * (waterShadow ? HALF_LITERAL(0.0f) : HALF_LITERAL(1.0f)));
            irradiance /= (albedo + HALF_LITERAL(0.001f));
        } else {
            // Standard Solid Block
            albedo = sampleTexture(hit.uv, hit.matID, hit.normal, textureAtlas, depth * depth);

#if SHADOWS
            bool isShadowed = traceShadow(hit.pos + (float3)normal * 0.005f, frame.sunDirection,
                                          SHADOW_MAXDIST, SHADOW_STEPS, indirection, sectorBuffer,
                                          occupancyBuffer, dataBuffer, sectorMaskBuffer, frame.worldOrigin, IND_X, IND_Y, IND_Z, &charData);
#else
            bool isShadowed = false;
#endif
            
            half NdotL = max(dot(normal, HALF3_FROM_FLOAT3(frame.sunDirection)), HALF_LITERAL(0.0f));
            irradiance = HALF3_FROM_FLOAT3(c_sunColor) * NdotL * (isShadowed ? HALF_LITERAL(0.02f) : HALF_LITERAL(1.0f));
        }
    } else {
        // Sky - zero motion for sky pixels
        irradiance = sampleSky(dir, frame.sunDirection);
        albedo = HALF3(1.0f, 1.0f, 1.0f);
        
#if defined(PLATFORM_METAL)
        TEX_WRITE_2D(texMotion, float4(0.0f, 0.0f, 0.0f, 0.0f), gid);
#else
        TEX_WRITE_2D(texMotion, make_float4(0.0f, 0.0f, 0.0f, 0.0f), gid);
#endif
    }

    float3 irradianceF3 = make_float3(irradiance.x, irradiance.y, irradiance.z);
    float3 albedoF3 = make_float3(albedo.x, albedo.y, albedo.z);
    float3 normalF3 = make_float3(normal.x, normal.y, normal.z);
#if defined(PLATFORM_METAL)
    TEX_WRITE_2D(texDirectLight, float4(irradianceF3, 1.0f), gid);
    TEX_WRITE_2D(texAlbedo, float4(albedoF3, 1.0f), gid);
    TEX_WRITE_2D(texNormal, float4(normalF3, 1.0f), gid);
    TEX_WRITE_2D(texDepth, float4(depth), gid);
#else
    TEX_WRITE_2D(texDirectLight, make_float4(irradianceF3.x, irradianceF3.y, irradianceF3.z, 1.0f), gid);
    TEX_WRITE_2D(texAlbedo, make_float4(albedoF3.x, albedoF3.y, albedoF3.z, 1.0f), gid);
    TEX_WRITE_2D(texNormal, make_float4(normalF3.x, normalF3.y, normalF3.z, 1.0f), gid);
    TEX_WRITE_2D(texDepth, make_float4(depth, 0.0f, 0.0f, 0.0f), gid);
#endif
}
