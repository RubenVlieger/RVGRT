// Volumetric Fog
// Ray-marches through the scene to accumulate volumetric lighting.

#include "ShaderTypes.slang"
#include "Intersection.slang"
#include "Raytracing.slang"

RWTexture2D<float4> texVolumetric;
Texture2D<float4> texDepth;
Texture2D<float4> texHistory;

uniform CameraData camera;
uniform FrameData frame;

Texture3D<uint> indirection;
StructuredBuffer<SectorInfo> sectorBuffer;
StructuredBuffer<uint64_t> occupancyBuffer;
StructuredBuffer<uint8_t> dataBuffer;
StructuredBuffer<uint64_t> sectorMaskBuffer;
ConstantBuffer<CharacterGPUData> charData;

SamplerState historySampler;

static float InterleavedGradientNoise(float2 pos) {
    return fract(52.9829189f * fract(0.06711056f * pos.x + 0.00583715f * pos.y));
}

static float phaseFunction(float3 viewDir, float3 lightDir, float g) {
    float cosTheta = dot(viewDir, lightDir);
    float denom = 1.0f + g * g - 2.0f * g * cosTheta;
    return (1.0f - g * g) / (4.0f * 3.14159f * pow(denom, 1.5f));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void VolumetricFog(uint3 gid : SV_DispatchThreadID)
{
    uint2 dim;
    texVolumetric.GetDimensions(dim.x, dim.y);
    if (gid.x >= dim.x || gid.y >= dim.y) return;

    #if !VOLUMETRIC_FOG
    texVolumetric[gid.xy] = float4(0.0f);
    return;
    #endif

    // Half-resolution coordinate
    uint2 fullResCoord = gid.xy * 2;

    // Clamp to valid depth coord
    if (fullResCoord.x >= dim.x * 2) fullResCoord.x = dim.x * 2 - 1;
    if (fullResCoord.y >= dim.y * 2) fullResCoord.y = dim.y * 2 - 1;

    float depth = texDepth[fullResCoord].r;
    float2 uv = (float2(gid.xy) + 0.5f) / float2(dim);

    float fogMaxDist = VOLUMETRIC_MAXDIST;
    float clampedDepth = min(depth, fogMaxDist);
    
    float3 endPos = reconstructPos(clampedDepth, uv, camera);
    float3 startPos = camera.position;
    
    float3 rayVec = endPos - startPos;
    float rayLength = length(rayVec);
    float3 rayDir = normalize(rayVec);

    // Ray marching with dithering
    float dither = InterleavedGradientNoise(float2(gid.xy) + float2(frame.time * 5.588f));
    
    const int STEPS = VOLUMETRIC_STEPS;
    float stepSize = rayLength / float(STEPS);
    float currentT = stepSize * dither;
    
    float3 accumulatedLight = float3(0.0f);
    float accumulatedTransmittance = 1.0f;
    
    const float density = FOG_DENSITY;
    const float3 fogColor = float3(FOG_COLOR);
    const float anisotropy = FOG_ANISOTROPY;
    float phase = phaseFunction(rayDir, frame.sunDirection, anisotropy);
    float3 sunColor = float3(c_sunColor);

    for(int i = 0; i < STEPS; i++) {
        float3 pos = startPos + rayDir * currentT;
        
        bool isShadowed = false;
        if (currentT > 2.0f && currentT < 200.0f && phase > 0.04f) {
            #if SHADOWS
            isShadowed = traceShadow(pos, frame.sunDirection, VOLUMETRIC_SHADOW_MAXDIST,
                                    VOLUMETRIC_SHADOW_STEPS, frame.worldOrigin, IND_X, IND_Y, IND_Z);
            #else
            isShadowed = false;
            #endif
        }

        if (!isShadowed) {
            accumulatedLight += sunColor * phase * density * accumulatedTransmittance * stepSize;
        }
        accumulatedLight += (fogColor * 0.05f) * density * accumulatedTransmittance * stepSize;
        accumulatedTransmittance *= exp(-density * stepSize);
        currentT += stepSize;
    }

    // Reprojection
    float4 prevClip = mul(camera.prevUnjitteredViewProjection, float4(endPos, 1.0f));
    
    float3 history = float3(0.0f);
    float blendFactor = 0.0f;

    if (prevClip.w > 0.01f) {
        float2 prevNDC = prevClip.xy / prevClip.w;
        
        float2 prevUV;
        prevUV.x = prevNDC.x * 0.5f + 0.5f;
        prevUV.y = 0.5f - prevNDC.y * 0.5f;

        if (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f) {
            history = texHistory.Sample(historySampler, prevUV).rgb;
            
            float diff = length(history - accumulatedLight);
            blendFactor = 0.8f;
            if (diff > 1.0f) blendFactor = 0.4f;
        }
    }

    float3 result = lerp(accumulatedLight, history, blendFactor);
    texVolumetric[gid.xy] = float4(result, 1.0f);
}
