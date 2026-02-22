#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
using namespace metal;

kernel void IndirectBounce(
    texture2d<float, access::write> texRawIndirect [[texture(0)]],
    texture2d<float, access::read> texNormal [[texture(1)]],
    texture2d<float, access::read> texDepth  [[texture(2)]],
    
    constant const CameraData& camera [[buffer(0)]],
    constant const FrameData& frame   [[buffer(1)]],
    
    texture2d_array<float, access::sample>  textureAtlas[[texture(8)]],

    texture3d<uint, access::read> indirection [[texture(3)]],
    device SectorInfo* sectorBuffer           [[buffer(3)]],
    device ulong* occupancyBuffer             [[buffer(4)]],
    device uchar* dataBuffer                  [[buffer(5)]], 

    uint2 gid [[thread_position_in_grid]])
{
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
    uint seed = voxelHash + uint(frame.time * 123456.0f); 

    // Orthonormal Basis
    float3 N = (float3)normal;
    if (dot(N, N) < 0.5f) {
        // Degenerate normal — output a small ambient value to avoid NaN in history
        texRawIndirect.write(float4(0.02f, 0.02f, 0.02f, 1.0f), gid);
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
    hitInfo hit = trace(pos + (float3)normal * 0.01f, rayDir, indirection, sectorBuffer, occupancyBuffer, dataBuffer);
    
    half3 incomingLight = half3(0.0h);

    if (hit.hit) 
    {   

        bool isShadowed = traceShadow(hit.pos + (float3)hit.normal * 0.01f, frame.sunDirection, 1000.0f, indirection, sectorBuffer, occupancyBuffer, dataBuffer);
        
        float distSq = (depth * depth) + dot(hit.pos - pos, hit.pos - pos); 
        half3 bouncedAlbedo = sampleTexture(hit.uv, hit.matID, hit.normal, textureAtlas, distSq);
        
        half NdotL = max(dot(hit.normal, (half3)frame.sunDirection), 0.0h);
        half3 directLightAtHit = c_sunColor * NdotL * (isShadowed ? 0.0h : 1.0h); 
        
        incomingLight = (directLightAtHit * bouncedAlbedo) + (bouncedAlbedo * 0.05h);
    } else 
    {
        half3 skyLight = sampleSky(rayDir, frame.sunDirection);
        float luma = dot((float3)skyLight, float3(0.3f, 0.59f, 0.11f));
        incomingLight = mix(skyLight, half3(luma), 0.6h) * 0.25h; 
    }
    
    texRawIndirect.write(float4((float3)incomingLight, 1.0f), gid);
}