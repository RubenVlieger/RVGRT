#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
using namespace metal;





// =================================================================================
// KERNEL 1: G-BUFFER & DIRECT LIGHTING
// =================================================================================
kernel void GBufferAndDirectLight(
    texture2d<float, access::write> texDirectLight [[texture(0)]],
    texture2d<float, access::write> texAlbedo      [[texture(1)]],
    texture2d<float, access::write> texNormal      [[texture(2)]],
    texture2d<float, access::write> texMotion      [[texture(3)]],
    texture2d<float, access::write> texDepth       [[texture(4)]],

    constant const CameraData& camera [[buffer(0)]],
    constant const FrameData& frame   [[buffer(1)]],
    
    texture3d<uint, access::read>     bitsTex     [[texture(5)]],
    texture3d<float, access::sample>  csdf        [[texture(6)]],
    texture2d_array<float, access::sample>  textureAtlas[[texture(7)]],
    texture2d<float, access::sample>  halfDistTex [[texture(8)]],

    texture3d<uint, access::read> matIndirection [[texture(9)]],
    device uchar* matBrickPool                   [[buffer(2)]],


    uint2 gid [[thread_position_in_grid]])
{
    // Bounds Check
    if (gid.x >= texDirectLight.get_width() || gid.y >= texDirectLight.get_height()) return;

    // 1. Ray Generation (Standard Pinhole with TAA Jitter)
    float2 pixelCenter = float2(gid) + 0.5f;
    float2 jitteredCoord = pixelCenter + camera.jitter; 
    
    float2 uv = jitteredCoord / float2(texDirectLight.get_width(), texDirectLight.get_height());
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    // 2.  Read Distance Estimation from Pre-Pass
    constexpr sampler sLinear(filter::linear);
    float startDist = halfDistTex.sample(sLinear, uv).r;
    
    // 3. Primary Ray Trace
    hitInfo hit = trace(camera.position, dir, startDist, bitsTex, csdf);

    float depth = 100000.0f;
    half3 irradiance = half3(0.0h);
    half3 albedo = half3(0.0h);
    half3 normal = half3(0.0h);
    if (hit.hit) 
    {
        depth = length(hit.pos - camera.position);
        normal = hit.normal;

        // --- Motion Vector Calculation ---
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
        texMotion.write(float4(motionVector.x, motionVector.y, 0, 0), gid);

        bool isWater = (hit.pos.y <= 3.001f && normal.y > 0.8h);
        if (isWater) 
        {
            
            albedo = half3(0.04h, 0.1h, 0.25h); 

            // 2. Waves
            float nx_wave = fbm3D(hit.pos.x, hit.pos.z, frame.time, 3, 0.06f, 2.0f, 0.6f);
            float ny_wave = fbm3D(hit.pos.z, hit.pos.x, frame.time + 112.0f, 3, 0.06f, 2.0f, 0.6f);
            half3 distortedNormal = normalize(normal + half3(half(nx_wave) * 0.1h, 0.0h, half(ny_wave) * 0.1h));
            
            // 3. Reflection
            float3 reflDir = reflect(dir, (float3)distortedNormal);
            hitInfo reflHit = trace(hit.pos, reflDir, 0.05f, bitsTex, csdf);
            
            half3 reflectColor;
            if (reflHit.hit) {
                // Hitting geometry
                float3 camToWater = hit.pos - camera.position;
                float3 waterToRefl = reflHit.pos - hit.pos;
                float distSq = dot(camToWater, camToWater) + dot(waterToRefl, waterToRefl);

                half3 rAlbedo = sampleTexture(reflHit.uv, reflHit.pos, reflHit.normal, textureAtlas, distSq, matIndirection, matBrickPool);
                
                // Shadow check for the reflected object
                bool rShadow = traceShadowAnyHitSlow(reflHit.pos + (float3)reflHit.normal * 0.01f, frame.sunDirection, 1000.0f, bitsTex, csdf);
                
                half3 litVal = c_sunColor;
                half3 shadowVal = half3(0.05h); // Neutral dark grey ambient

                reflectColor = rAlbedo * (rShadow ? shadowVal : litVal);
            } else {
                // Hitting Sky
                reflectColor = sampleSky(reflDir, frame.sunDirection);
            }
            
            // 4. Specular
            float3 viewDir = -dir;
            float3 halfVec = normalize(viewDir + frame.sunDirection);
            float NdotH = max(dot((float3)distortedNormal, halfVec), 0.0f);
            half specular = pow(NdotH, 512.0f) * 4.0f; 

            // 5. Fresnel
            half NdotV = max(dot(distortedNormal, (half3)viewDir), 0.0h);
            half fresnel = 0.02h + (0.98h) * pow(1.0h - NdotV, 5.0h);
            
            bool waterShadow = traceShadowAnyHitSlow(hit.pos, frame.sunDirection, 1000.0f, bitsTex, csdf);
            half shadowVal = waterShadow ? 0.0h : 1.0h;

            // 6. Combine
            half3 totalReflection = (reflectColor * fresnel) + (c_sunColor * specular * shadowVal);

            irradiance = totalReflection / (albedo + 0.001h);
        } else 
        {            
            // 1. Texture Sampling
            albedo = sampleTexture(hit.uv, hit.pos, hit.normal, textureAtlas, depth * depth, matIndirection, matBrickPool);
            
            // 2. Shadow Trace
            bool isShadowed = traceShadowAnyHitSlow(hit.pos + (float3)normal * 0.005f, frame.sunDirection, 2000.0f, bitsTex, csdf);
            half shadowFactor = isShadowed ? 0.02h : 1.0h;
            
            half NdotL = max(dot(normal, (half3)frame.sunDirection), 0.0h);
            irradiance = c_sunColor * NdotL * shadowFactor;
        }
    } 
    else 
    {
        // === SKYBOX LOGIC ===
        // 1. Sky Motion Vectors
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
        texMotion.write(float4(mv.x, mv.y, 0, 0), gid);

        // 2. Sky Lighting
        // For the sky, Light * Albedo must equal SkyColor.
        irradiance = sampleSky(dir, frame.sunDirection);
        albedo = half3(1.0h); 
    }

    // --- Write Outputs ---
    texDirectLight.write(float4((float3)irradiance, 1.0f), gid);
    texAlbedo.write(float4((float3)albedo, 1.0f), gid);
    texNormal.write(float4(((float3)normal), 1.0f), gid); 
    texDepth.write(float4(depth), gid);
}


