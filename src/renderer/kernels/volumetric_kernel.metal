#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
#include "renderer/shader_settings.h"

using namespace metal;



inline float InterleavedGradientNoise(float2 pos) {
    return fract(52.9829189f * fract(0.06711056f * pos.x + 0.00583715f * pos.y));
}

// =================================================================================
// Controls the "halo" around the sun (anisotropy).
// g = 0 (isotropic), g near 1 (strong forward scattering/god rays)
// =================================================================================
inline float phaseFunction(float3 viewDir, float3 lightDir, float g) {
    float cosTheta = dot(viewDir, lightDir);
    float denom = 1.0f + g * g - 2.0f * g * cosTheta;
    return (1.0f - g * g) / (4.0f * 3.14159f * pow(denom, 1.5f));
}


// =================================================================================
// KERNEL: Volumetric Fog
// =================================================================================
kernel void VolumetricFog(
    texture2d<float, access::write> texVolumetric [[texture(0)]], 
    texture2d<float, access::read>  texDepth      [[texture(1)]], 
    texture2d<float, access::sample> texHistory   [[texture(2)]], 
    
    constant const CameraData& camera [[buffer(0)]],
    constant const FrameData& frame   [[buffer(1)]],
    
    texture3d<uint, access::read> indirection [[texture(3)]],
    device SectorInfo* sectorBuffer           [[buffer(3)]],
    device ulong* occupancyBuffer             [[buffer(4)]],
    device ulong* sectorMaskBuffer            [[buffer(6)]],
    constant CharacterGPUData* charData       [[buffer(7)]],

    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texVolumetric.get_width() || gid.y >= texVolumetric.get_height()) return;

#if VOLUMETRIC_FOG
    // --- 1. SETUP ---
    uint2 fullResCoord = gid * 2;
    if (fullResCoord.x >= texDepth.get_width()) fullResCoord.x = texDepth.get_width() - 1;
    if (fullResCoord.y >= texDepth.get_height()) fullResCoord.y = texDepth.get_height() - 1;

    float depth = texDepth.read(fullResCoord).r;
    float2 uv = (float2(gid) + 0.5f) / float2(texVolumetric.get_width(), texVolumetric.get_height());

    // Clamp depth to fog distance
    float fogMaxDist = VOLUMETRIC_MAXDIST; 
    float clampedDepth = min(depth, fogMaxDist);
    
    float3 endPos = reconstructPos(clampedDepth, uv, camera);
    float3 startPos = camera.position;
    
    float3 rayVec = endPos - startPos;
    float rayLength = length(rayVec);
    float3 rayDir = normalize(rayVec);

    // --- 2. RAY MARCHING (16 Samples) ---
    // Use IGN for dithering. It produces a checkerboard pattern that TAA resolves perfectly.
    float dither = InterleavedGradientNoise(float2(gid) + float2(frame.time * 5.588f));
    
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
        
        // Skip shadow check for first few meters to prevent "face self-shadowing" artifacts
        bool isShadowed = false;
        // If the sun is almost behind the view ray, the phase term is very small;
        // in that case we can safely skip the expensive shadow query.
        // Also skip for very far samples where fog contribution is negligible.
        if (currentT > 2.0f && currentT < 200.0f && phase > 0.04f) { 
            // Volumetric shadows: shorter max distance and fewer traversal iterations.
#if SHADOWS
            isShadowed = traceShadow(pos,
                                     frame.sunDirection,
                                     VOLUMETRIC_SHADOW_MAXDIST,
                                     VOLUMETRIC_SHADOW_STEPS,
                                     indirection,
                                     sectorBuffer,
                                     occupancyBuffer,
                                     0,
                                     sectorMaskBuffer,
                                     frame.worldOrigin,
                                     charData);
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

    // --- 3. ROBUST REPROJECTION ---
    
    // Transform World Position to Previous Clip Space
    float4 prevClip = camera.prevUnjitteredViewProjection * float4(endPos, 1.0f);
    
    float3 history = float3(0.0f);
    float blendFactor = 0.0f; 

    // Safety: Check if point is in front of camera (w > 0)
    if (prevClip.w > 0.01f) {
        float2 prevNDC = prevClip.xy / prevClip.w;
        
        // Manual UV Calculation to fix "Tearing"
        // X: [-1, 1] -> [0, 1]
        // Y: [-1, 1] -> [0, 1] (FLIPPED for Metal texture coords)
        float2 prevUV;
        prevUV.x = prevNDC.x * 0.5f + 0.5f;
        prevUV.y = 0.5f - prevNDC.y * 0.5f; //

        // Check bounds (0.0 to 1.0)
        if (prevUV.x >= 0.0f && prevUV.x <= 1.0f && 
            prevUV.y >= 0.0f && prevUV.y <= 1.0f) 
        {
            constexpr sampler sLinear(filter::linear);
            history = texHistory.sample(sLinear, prevUV).rgb;
            
            float diff = length(history - accumulatedLight);
            
            // Standard blend
            blendFactor = 0.8f; 
            
            // If history is too different, reduce confidence (Ghosting fix)
            if (diff > 1.0f) blendFactor = 0.4f;
        }
    }

    float3 result = mix(accumulatedLight, history, blendFactor);
    texVolumetric.write(float4(result, 1.0f), gid);
#else
    texVolumetric.write(float4(0.0f), gid);
#endif
}


