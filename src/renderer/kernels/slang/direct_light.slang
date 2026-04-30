// GBuffer + Direct Light
// Primary ray tracing kernel. Generates GBuffer and computes direct lighting.

#include "ShaderTypes.slang"
#include "Intersection.slang"
#include "Raytracing.slang"
#include "TerrainGen.slang"

RWTexture2D<float4> texDirectLight;
RWTexture2D<float4> texAlbedo;
RWTexture2D<float4> texNormal;
RWTexture2D<float4> texMotion;
RWTexture2D<float4> texDepth;

uniform CameraData camera;
uniform FrameData frame;

Texture3D<uint> indirection;
StructuredBuffer<SectorInfo> sectorBuffer;
StructuredBuffer<uint64_t> occupancyBuffer;
StructuredBuffer<uint8_t> dataBuffer;
StructuredBuffer<uint64_t> sectorMaskBuffer;
ConstantBuffer<CharacterGPUData> charData;

Texture2DArray<float> textureAtlas;
SamplerState atlasSampler;
Texture2D<float> halfDistTex;
SamplerState distSampler;

[shader("compute")]
[numthreads(16, 16, 1)]
void GBufferAndDirectLight(uint3 gid : SV_DispatchThreadID)
{
    uint2 dim;
    texDirectLight.GetDimensions(dim.x, dim.y);
    if (gid.x >= dim.x || gid.y >= dim.y) return;

    // Ray Gen
    float2 pixelCenter = float2(gid.xy) + 0.5f;
    float2 jitteredCoord = pixelCenter + camera.jitter;
    float2 uv = jitteredCoord / float2(dim);
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    // Read Dist Approx
    float startDist = halfDistTex.Sample(distSampler, uv).r;

    hitInfo hit = trace(camera.position + startDist * dir, dir, 
                        frame.worldOrigin, IND_X, IND_Y, IND_Z);

    float depth = 100000.0f;
    half3 irradiance = half3(0.0h, 0.0h, 0.0h);
    half3 albedo = half3(0.0h, 0.0h, 0.0h);
    half3 normal = half3(0.0h, 0.0h, 0.0h);

    if (hit.hit) {
        depth = length(hit.pos - camera.position);
        normal = hit.normal;

        // Motion Vectors
        float2 motionVector = float2(0.0f);
        if (depth < 50000.0f) {
            float4 currentClipPos = mul(camera.unjitteredViewProjection, float4(hit.pos, 1.0f));
            float4 previousClipPos = mul(camera.prevUnjitteredViewProjection, float4(hit.pos, 1.0f));
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
        texMotion[gid.xy] = float4(motionVector, 0.0f, 0.0f);

        // Water Logic
        bool isWater = (hit.pos.y <= 3.001f && normal.y > half(0.8h));
        if (isWater) {
            albedo = sampleTexture(textureAtlas, atlasSampler, hit.uv, uint8_t(hit.matID), hit.normal, depth * depth);
            float nx = fbm3D(hit.pos.x, hit.pos.z, frame.time, 3, 0.06f, 2.0f, 0.6f);
            float ny = fbm3D(hit.pos.z, hit.pos.x, frame.time + 112.0f, 3, 0.06f, 2.0f, 0.6f);
            half3 distNormal = normalize(normal + half3(half(nx*0.1f), half(0.0f), half(ny*0.1f)));
            
            float3 reflDir = reflect(dir, (float3)distNormal);
            
            #if REFLECTIONS
            hitInfo reflHit = trace(hit.pos, reflDir, 
                                   frame.worldOrigin, IND_X, IND_Y, IND_Z);
            #else
            hitInfo reflHit;
            reflHit.hit = false;
            reflHit.pos = hit.pos;
            #endif
            
            half3 reflectColor;
            #if REFLECTIONS
            if (reflHit.hit) {
                float distSq = dot(reflHit.pos - hit.pos, reflHit.pos - hit.pos);
                half3 rAlbedo = sampleTexture(textureAtlas, atlasSampler, reflHit.uv, uint8_t(reflHit.matID), reflHit.normal, distSq);
                
                #if SHADOWS
                bool rShadow = traceShadow(reflHit.pos + (float3)reflHit.normal * 0.01f,
                                           frame.sunDirection, REFLECTION_SHADOW_MAXDIST, REFLECTION_SHADOW_STEPS,
                                           frame.worldOrigin, IND_X, IND_Y, IND_Z);
                #else
                bool rShadow = false;
                #endif
                reflectColor = rAlbedo * (rShadow ? half3(0.05h, 0.05h, 0.05h) : half3(c_sunColor));
            } else {
                reflectColor = sampleSky(reflDir, frame.sunDirection);
            }
            #else
            reflectColor = sampleSky(reflDir, frame.sunDirection);
            #endif
            
            float3 viewDir = -dir;
            float3 halfVec = normalize(viewDir + frame.sunDirection);
            float NdotH = max(dot((float3)distNormal, halfVec), 0.0f);
            half specular = half(pow(NdotH, 512.0f)) * half(4.0f);
            half NdotV = saturate(dot(distNormal, half3(viewDir)));
            half fresnel = half(0.02h) + half(0.98h) * pow(half(1.0h) - NdotV, half(5.0h));
            
            #if SHADOWS
            bool waterShadow = traceShadow(reflHit.pos, frame.sunDirection, WATER_SHADOW_MAXDIST,
                                          WATER_SHADOW_STEPS, frame.worldOrigin, IND_X, IND_Y, IND_Z);
            #else
            bool waterShadow = false;
            #endif
            
            irradiance = (reflectColor * fresnel) + (half3(c_sunColor) * specular * (waterShadow ? half(0.0h) : half(1.0h)));
            irradiance /= (albedo + half(0.001h));
        } else {
            // Standard Solid Block
            albedo = sampleTexture(textureAtlas, atlasSampler, hit.uv, uint8_t(hit.matID), hit.normal, depth * depth);

            #if SHADOWS
            bool isShadowed = traceShadow(hit.pos + (float3)normal * 0.005f, frame.sunDirection,
                                          SHADOW_MAXDIST, SHADOW_STEPS, frame.worldOrigin, IND_X, IND_Y, IND_Z);
            #else
            bool isShadowed = false;
            #endif
            
            half NdotL = saturate(dot(normal, half3(frame.sunDirection)));
            irradiance = half3(c_sunColor) * NdotL * (isShadowed ? half(0.02h) : half(1.0h));
        }
    } else {
        // Sky
        irradiance = sampleSky(dir, frame.sunDirection);
        albedo = half3(1.0h, 1.0h, 1.0h);
        
        float3 fakePos = camera.position + dir * 1000.0f;
        float4 currentClipPos = mul(camera.unjitteredViewProjection, float4(fakePos, 1.0f));
        float4 previousClipPos = mul(camera.prevUnjitteredViewProjection, float4(fakePos, 1.0f));
        float2 mv = float2(0.0f, 0.0f);
        if (previousClipPos.w > 0.0f && currentClipPos.w > 0.0f) {
            float2 prevNDC = float2(previousClipPos.x / previousClipPos.w, previousClipPos.y / previousClipPos.w);
            float2 currNDC = float2(currentClipPos.x / currentClipPos.w, currentClipPos.y / currentClipPos.w);
            mv = float2(0.5f * (currNDC.x - prevNDC.x), 0.5f * (currNDC.y - prevNDC.y));
            mv.y = -mv.y;
        }
        texMotion[gid.xy] = float4(mv, 0.0f, 0.0f);
    }

    texDirectLight[gid.xy] = float4((float3)irradiance, 1.0f);
    texAlbedo[gid.xy] = float4((float3)albedo, 1.0f);
    texNormal[gid.xy] = float4((float3)normal, 1.0f);
    texDepth[gid.xy] = float4(depth, 0.0f, 0.0f, 0.0f);
}
