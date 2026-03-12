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

    PARAM_INDIRECTION(indirection, 5),
    PARAM_BUFFER(SectorInfo, sectorBuffer, 3),
    PARAM_BUFFER(ulong, occupancyBuffer, 4),
    PARAM_BUFFER(uchar, dataBuffer, 5),
    PARAM_BUFFER(ulong, sectorMaskBuffer, 6),
    PARAM_CONSTANT_PTR(CharacterGPUData, charData, 7),
    
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
    float2 uv = jitteredCoord / make_float2((float)width, (float)height);
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    // Read Dist Approx
    DECLARE_SAMPLER(sLinear, linear, clamp_to_edge);
#if defined(PLATFORM_METAL)
    float startDist = TEX_SAMPLE_2D(halfDistTex, uv).r;
#else
    float4 halfDistSample = TEX_SAMPLE_2D(halfDistTex, uv);
    float startDist = halfDistSample.x;
#endif

    hitInfo hit = trace(camera.position + startDist * dir, dir, indirection, sectorBuffer,
                        occupancyBuffer, dataBuffer, sectorMaskBuffer, frame.worldOrigin, SECTOR_IND_X, SECTOR_IND_Y, SECTOR_IND_Z, charData);

    float depth = 100000.0f;
    half3 irradiance = HALF3(0.0f, 0.0f, 0.0f);
    half3 albedo = HALF3(0.0f, 0.0f, 0.0f);
    half3 normal = HALF3(0.0f, 0.0f, 0.0f);

    if (hit.hit) {
        depth = length(hit.pos - camera.position);
        normal = hit.normal;

        // Motion Vectors
#if defined(PLATFORM_METAL)
        float2 motionVector = float2(0.0f);
#else
        float2 motionVector = make_float2(0.0f, 0.0f);
#endif
        if (depth < 50000.0f) {
#if defined(PLATFORM_METAL)
            float4 currentClipPos = camera.unjitteredViewProjection * float4(hit.pos, 1.0f);
            float4 previousClipPos = camera.prevUnjitteredViewProjection * float4(hit.pos, 1.0f);
#else
            float4 currentClipPos = camera.unjitteredViewProjection * make_float4(hit.pos.x, hit.pos.y, hit.pos.z, 1.0f);
            float4 previousClipPos = camera.prevUnjitteredViewProjection * make_float4(hit.pos.x, hit.pos.y, hit.pos.z, 1.0f);
#endif
            if (previousClipPos.w > 0.0f && currentClipPos.w > 0.0f) {
                float2 prevNDC = make_float2(previousClipPos.x / previousClipPos.w, previousClipPos.y / previousClipPos.w);
                float2 currNDC = make_float2(currentClipPos.x / currentClipPos.w, currentClipPos.y / currentClipPos.w);
                motionVector = 0.5f * (currNDC - prevNDC);
                motionVector.y = -motionVector.y;
            }
        }
#if defined(PLATFORM_METAL)
    TEX_WRITE_2D(texMotion, float4(motionVector.x, motionVector.y, 0, 0), gid);
#else
    // RG16F format for motion vectors
    ushort2 motionHalf2;
    motionHalf2.x = __float2half_rn(motionVector.x);
    motionHalf2.y = __float2half_rn(motionVector.y);
    TEX_WRITE_2D_RG16F(texMotion, motionHalf2, gid);
#endif

        // Water Logic
#if defined(PLATFORM_METAL)
        bool isWater = (hit.pos.y <= 3.001f && normal.y > HALF_LITERAL(0.8f));
#else
        bool isWater = (hit.pos.y <= 3.001f && normal.y > 0.8f);
#endif
        if (isWater) {
            albedo = sampleTexture(hit.uv, hit.matID, hit.normal, textureAtlas, depth * depth);
            float nx = fbm3D(hit.pos.x, hit.pos.z, frame.time, 3, 0.06f, 2.0f, 0.6f);
            float ny = fbm3D(hit.pos.z, hit.pos.x, frame.time + 112.0f, 3, 0.06f, 2.0f, 0.6f);
#if defined(PLATFORM_METAL)
            half3 distNormal = normalize(normal + HALF3(HALF_LITERAL(nx*0.1f), HALF_LITERAL(0.0f), HALF_LITERAL(ny*0.1f)));
#else
            half3 distNormal = normalize(normal + make_float3(nx*0.1f, 0.0f, ny*0.1f));
#endif
            
            float3 reflDir = reflect(dir, (float3)distNormal);
            
#if REFLECTIONS
            hitInfo reflHit = trace(hit.pos, reflDir, indirection, sectorBuffer, occupancyBuffer,
                                   dataBuffer, sectorMaskBuffer, frame.worldOrigin, SECTOR_IND_X, SECTOR_IND_Y, SECTOR_IND_Z, charData);
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
                                           sectorMaskBuffer, frame.worldOrigin, SECTOR_IND_X, SECTOR_IND_Y, SECTOR_IND_Z, charData);
#else
                bool rShadow = false;
#endif
#if defined(PLATFORM_METAL)
                reflectColor = rAlbedo * (rShadow ? HALF3(0.05f, 0.05f, 0.05f) : HALF3_FROM_FLOAT3(c_sunColor));
#else
                reflectColor = rAlbedo * (rShadow ? make_float3(0.05f, 0.05f, 0.05f) : make_float3(c_sunColor.x, c_sunColor.y, c_sunColor.z));
#endif
            } else {
                reflectColor = sampleSky(reflDir, frame.sunDirection);
            }
#else
            reflectColor = sampleSky(reflDir, frame.sunDirection);
#endif
            
            float3 viewDir = -dir;
            float3 halfVec = normalize(viewDir + frame.sunDirection);
            float NdotH = max(dot((float3)distNormal, halfVec), 0.0f);
#if defined(PLATFORM_METAL)
            half specular = pow(NdotH, 512.0f) * 4.0f;
            half NdotV = max(dot(distNormal, HALF3_FROM_FLOAT3(viewDir)), HALF_LITERAL(0.0f));
            half fresnel = HALF_LITERAL(0.02f) + HALF_LITERAL(0.98f) * pow(HALF_LITERAL(1.0f) - NdotV, HALF_LITERAL(5.0f));
#else
            float specular = powf(NdotH, 512.0f) * 4.0f;
            float NdotV = max(dot(distNormal, viewDir), 0.0f);
            float fresnel = 0.02f + 0.98f * powf(1.0f - NdotV, 5.0f);
#endif
            
#if SHADOWS
            bool waterShadow = traceShadow(reflHit.pos, frame.sunDirection, WATER_SHADOW_MAXDIST,
                                          WATER_SHADOW_STEPS, indirection, sectorBuffer, occupancyBuffer,
                                          dataBuffer, sectorMaskBuffer, frame.worldOrigin, SECTOR_IND_X, SECTOR_IND_Y, SECTOR_IND_Z, charData);
#else
            bool waterShadow = false;
#endif
#if defined(PLATFORM_METAL)
            irradiance = (reflectColor * fresnel) + (HALF3_FROM_FLOAT3(c_sunColor) * specular * (waterShadow ? HALF_LITERAL(0.0f) : HALF_LITERAL(1.0f)));
            irradiance /= (albedo + HALF_LITERAL(0.001f));
#else
            irradiance = (reflectColor * fresnel) + (make_float3(c_sunColor.x, c_sunColor.y, c_sunColor.z) * specular * (waterShadow ? 0.0f : 1.0f));
            irradiance = irradiance / (albedo + make_float3(0.001f, 0.001f, 0.001f));
#endif
        } else {
            // Standard Solid Block
            albedo = sampleTexture(hit.uv, hit.matID, hit.normal, textureAtlas, depth * depth);

#if SHADOWS
            bool isShadowed = traceShadow(hit.pos + (float3)normal * 0.005f, frame.sunDirection,
                                          SHADOW_MAXDIST, SHADOW_STEPS, indirection, sectorBuffer,
                                          occupancyBuffer, dataBuffer, sectorMaskBuffer, frame.worldOrigin, SECTOR_IND_X, SECTOR_IND_Y, SECTOR_IND_Z, charData);
#else
            bool isShadowed = false;
#endif
            
            half NdotL = max(dot(normal, HALF3_FROM_FLOAT3(frame.sunDirection)), HALF_LITERAL(0.0f));
            irradiance = HALF3_FROM_FLOAT3(c_sunColor) * NdotL * (isShadowed ? HALF_LITERAL(0.02f) : HALF_LITERAL(1.0f));
        }
    } else {
        // Sky
        irradiance = sampleSky(dir, frame.sunDirection);
        albedo = HALF3(1.0f, 1.0f, 1.0f);
        
        float3 fakePos = camera.position + dir * 1000.0f;
#if defined(PLATFORM_METAL)
        float4 currentClipPos = camera.unjitteredViewProjection * float4(fakePos, 1.0f);
        float4 previousClipPos = camera.prevUnjitteredViewProjection * float4(fakePos, 1.0f);
#else
        float4 currentClipPos = camera.unjitteredViewProjection * make_float4(fakePos.x, fakePos.y, fakePos.z, 1.0f);
        float4 previousClipPos = camera.prevUnjitteredViewProjection * make_float4(fakePos.x, fakePos.y, fakePos.z, 1.0f);
#endif
        float2 mv = make_float2(0.0f, 0.0f);
        if (previousClipPos.w > 0.0f && currentClipPos.w > 0.0f) {
            float2 prevNDC = make_float2(previousClipPos.x / previousClipPos.w, previousClipPos.y / previousClipPos.w);
            float2 currNDC = make_float2(currentClipPos.x / currentClipPos.w, currentClipPos.y / currentClipPos.w);
            mv = make_float2(0.5f * (currNDC.x - prevNDC.x), 0.5f * (currNDC.y - prevNDC.y));
            mv.y = -mv.y;
        }
#if defined(PLATFORM_METAL)
    TEX_WRITE_2D(texMotion, float4(mv.x, mv.y, 0, 0), gid);
#else
    // RG16F format for motion vectors
    ushort2 mvHalf2;
    mvHalf2.x = __float2half_rn(mv.x);
    mvHalf2.y = __float2half_rn(mv.y);
    TEX_WRITE_2D_RG16F(texMotion, mvHalf2, gid);
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
    // RGBA16F format for direct light and normal
    ushort4 irradianceHalf4, normalHalf4;
    irradianceHalf4.x = __float2half_rn(irradianceF3.x);
    irradianceHalf4.y = __float2half_rn(irradianceF3.y);
    irradianceHalf4.z = __float2half_rn(irradianceF3.z);
    irradianceHalf4.w = __float2half_rn(1.0f);
    normalHalf4.x = __float2half_rn(normalF3.x);
    normalHalf4.y = __float2half_rn(normalF3.y);
    normalHalf4.z = __float2half_rn(normalF3.z);
    normalHalf4.w = __float2half_rn(1.0f);
    TEX_WRITE_2D_RGBA16F(texDirectLight, irradianceHalf4, gid);
    TEX_WRITE_2D_RGBA16F(texNormal, normalHalf4, gid);
    
    // RGBA8 format for albedo
    uchar4 albedoUChar4;
    albedoUChar4.x = (unsigned char)(fminf(fmaxf(albedoF3.x * 255.0f, 0.0f), 255.0f));
    albedoUChar4.y = (unsigned char)(fminf(fmaxf(albedoF3.y * 255.0f, 0.0f), 255.0f));
    albedoUChar4.z = (unsigned char)(fminf(fmaxf(albedoF3.z * 255.0f, 0.0f), 255.0f));
    albedoUChar4.w = 255;
    TEX_WRITE_2D_RGBA8(texAlbedo, albedoUChar4, gid);
    
    // R32F format for depth
    TEX_WRITE_2D_R32F(texDepth, depth, gid);
#endif
}
