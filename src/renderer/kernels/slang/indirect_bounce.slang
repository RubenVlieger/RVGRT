// Indirect Bounce (1-bounce global illumination)
// Samples a random direction in the hemisphere and traces a bounce ray.

#include "ShaderTypes.slang"
#include "Intersection.slang"
#include "Raytracing.slang"

RWTexture2D<float4> texRawIndirect;
Texture2D<float4> texNormal;
Texture2D<float4> texDepth;

uniform CameraData camera;
uniform FrameData frame;

Texture2DArray<float> textureAtlas;
SamplerState atlasSampler;
Texture3D<uint> indirection;
StructuredBuffer<SectorInfo> sectorBuffer;
StructuredBuffer<uint64_t> occupancyBuffer;
StructuredBuffer<uint8_t> dataBuffer;
StructuredBuffer<uint64_t> sectorMaskBuffer;
ConstantBuffer<CharacterGPUData> charData;

[shader("compute")]
[numthreads(16, 16, 1)]
void IndirectBounce(uint3 gid : SV_DispatchThreadID)
{
    uint2 dim;
    texRawIndirect.GetDimensions(dim.x, dim.y);
    if (gid.x >= dim.x || gid.y >= dim.y) return;

    #if !INDIRECT_LIGHTING
    texRawIndirect[gid.xy] = float4(0.0f);
    return;
    #endif

    // texDepth and texNormal are full-resolution, but this pass dispatches at half-res.
    // Scale gid by 2 to sample the correct full-res pixel.
    uint2 fullResGid = gid.xy * 2;

    float depth = texDepth[fullResGid].r;
    if (depth > 50000.0f) {
        texRawIndirect[gid.xy] = float4(0,0,0,0);
        return;
    }

    half3 normal = half3(texNormal[fullResGid].rgb);
    float2 uv = (float2(gid.xy) + 0.5f) / float2(dim);
    float3 pos = reconstructPos(depth, uv, camera);

    uint voxelHash = hash3_to_1(int3(pos * 1024.f));
    uint seed = voxelHash + uint(frame.time * 123456.0f);

    // Orthonormal Basis
    float3 N = (float3)normal;
    if (dot(N, N) < 0.5f) {
        texRawIndirect[gid.xy] = float4(0.02f, 0.02f, 0.02f, 1.0f);
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
    hitInfo hit = trace(pos + (float3)normal * 0.01f, rayDir, 
                        frame.worldOrigin, IND_X, IND_Y, IND_Z,
                        INDIRECT_BOUNCE_MAX_ITERS);
    
    half3 incomingLight = half3(0.0h, 0.0h, 0.0h);

    if (hit.hit) {
        #if SHADOWS
        bool isShadowed = traceShadow(hit.pos + (float3)hit.normal * 0.01f,
                                      frame.sunDirection,
                                      INDIRECT_SHADOW_MAXDIST,
                                      INDIRECT_SHADOW_STEPS,
                                      frame.worldOrigin, IND_X, IND_Y, IND_Z);
        #else
        bool isShadowed = false;
        #endif
        
        float distSq = (depth * depth) + dot(hit.pos - pos, hit.pos - pos);
        half3 bouncedAlbedo = sampleTexture(textureAtlas, atlasSampler, hit.uv, uint8_t(hit.matID), hit.normal, distSq);
        
        half NdotL = saturate(dot(hit.normal, half3(frame.sunDirection)));
        half3 directLightAtHit = c_sunColor * NdotL * (isShadowed ? half(0.0h) : half(1.0h));
        
        incomingLight = (directLightAtHit * bouncedAlbedo) + (bouncedAlbedo * half(0.05h));
    } else {
        half3 skyLight = sampleSky(rayDir, frame.sunDirection);
        float luma = dot((float3)skyLight, float3(0.3f, 0.59f, 0.11f));
        incomingLight = lerp(skyLight, half3(half(luma), half(luma), half(luma)), half(0.6h)) * half(0.25h);
    }
    
    texRawIndirect[gid.xy] = float4((float3)incomingLight, 1.0f);
}
