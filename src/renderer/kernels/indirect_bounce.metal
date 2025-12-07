#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
using namespace metal;


// =================================================================================
// KERNEL: INDIRECT BOUNCE (1 Bounce)
// =================================================================================
kernel void IndirectBounce(
    // --- Output ---
    texture2d<float, access::write> texRawIndirect [[texture(0)]],
    
    // --- Inputs ---
    texture2d<float, access::read> texNormal [[texture(1)]],
    texture2d<float, access::read> texDepth  [[texture(2)]],
    
    constant const CameraData& camera [[buffer(0)]],
    constant const FrameData& frame   [[buffer(1)]],
    
    texture3d<uint, access::read>    bitsTex     [[texture(3)]],
    texture3d<float, access::sample> csdf        [[texture(4)]],
    texture2d_array<float, access::sample> textureAtlas[[texture(5)]],

    texture3d<uint, access::read> matIndirection [[texture(6)]],
    device uchar* matBrickPool                   [[buffer(2)]],
    
    uint2 gid [[thread_position_in_grid]])
{
    // 1. Bounds Check
    if (gid.x >= texRawIndirect.get_width() || gid.y >= texRawIndirect.get_height()) return;
    
    float depth = texDepth.read(gid).r;
    
    if (depth > 50000.0f) {
        texRawIndirect.write(float4(0,0,0,0), gid);
        return;
    }

    half3 normal = (half3)texNormal.read(gid).rgb;
    
    float2 uv = (float2(gid) + 0.5f) / float2(texRawIndirect.get_width(), texRawIndirect.get_height());
    float3 pos = reconstructPos(depth, uv, camera);

    uint voxelHash = hash3_to_1(int3(pos * 1024.f));
    uint seed = voxelHash + uint(frame.time * 123456.0f); // Time dependent for accumulation
    

    // 5. Create Orthonormal Basis (Tangent Space)
    float3 N = (float3)normal;

    // Duff's method or simple helper to find perpendicular vector
    float3 helper = abs(N.x) > 0.99f ? float3(0,0,1) : float3(1,0,0);
    float3 Tangent = normalize(cross(N, helper));
    float3 Bitangent = cross(N, Tangent);

    // 6. Cosine-Weighted Hemisphere Sampling
    float r1 = rand_float(seed);
    float r2 = rand_float(seed);
    
    // Map square random numbers to hemisphere
    float phi = 2.0f * 3.14159f * r1;
    float cosTheta = sqrt(1.0f - r2);
    float sinTheta = sqrt(r2); 
    
    float3 localDir = float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));

    // Transform to World Space
    float3 rayDir = localDir.x * Tangent + localDir.y * N + localDir.z * Bitangent;
    rayDir = normalize(rayDir);

    // 7. Trace the Bounce Ray
    hitInfo hit = trace(pos , rayDir, 0.05f, bitsTex, csdf);
    
    half3 incomingLight = half3(0.0h);

    if (hit.hit) 
    {   
        bool isShadowed = traceShadowAnyHitFast(hit.pos + (float3)hit.normal * 0.01f, frame.sunDirection, 1000.0f, bitsTex, csdf);
        
        half2 hitUV = reconstructUV(hit.pos, hit.normal);
        float3 bounceVec = hit.pos - pos;
        float totalDistSq = (depth * depth) + dot(bounceVec, bounceVec); 
        
        half3 bouncedAlbedo = sampleTexture(hitUV, hit.pos, hit.normal, textureAtlas, totalDistSq, matIndirection, matBrickPool);
        
        // C. Calculate Radiance
        half NdotL = max(dot(hit.normal, (half3)frame.sunDirection), 0.0h);
        
        // This restores color bleeding (e.g. Gold reflecting yellow light)
        half3 directLightAtHit = c_sunColor * NdotL * (isShadowed ? 0.0h : 1.0h); 
        
        // Add a tiny bit of bounce ambient to prevent pitch black corners, 
        half3 bounceAmbient = bouncedAlbedo * 0.05h; 

        incomingLight = (directLightAtHit * bouncedAlbedo) + bounceAmbient;

    } else {
        // We hit the sky
        half3 skyLight = sampleSky(rayDir, frame.sunDirection);
        float luma = dot((float3)skyLight, float3(0.3f, 0.59f, 0.11f));
        half3 desaturatedSky = mix(skyLight, half3(luma), 0.6h); 
        
        incomingLight = desaturatedSky * 0.25h; 
    }
    
    texRawIndirect.write(float4((float3)incomingLight, 1.0f), gid);
}