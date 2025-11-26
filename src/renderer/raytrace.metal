#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
using namespace metal;

// =================================================================================
// HELPER FUNCTIONS
// =================================================================================

// High-quality, fast Pseudo-Random Number Generator (PCG Hash)
// Essential for path tracing to get "good noise" that denoises well.
inline uint pcg_hash(uint seed)
{
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Float random [0, 1]
inline float rand_float(thread uint& seed) {
    seed = pcg_hash(seed);
    return (float)seed / (float)UINT_MAX;
}

// Reconstruct World Position from Depth and Camera info
inline float3 reconstructPos(float depth, float2 uv, constant const CameraData& cam) {
    float2 ndc = uv * 2.0f - 1.0f;
    float3 viewDir = normalize(cam.forward + ndc.x * cam.right + ndc.y * cam.up);
    return cam.position + viewDir * depth;
}

inline half2 reconstructUV(float3 pos, half3 normal) {
    float3 fpos = floor(pos);
    half2 uv;
    if (abs(normal.x) > 0.5h)      uv = half2(pos.y - fpos.y, pos.z - fpos.z);
    else if (abs(normal.y) > 0.5h) uv = half2(pos.x - fpos.x, pos.z - fpos.z);
    else                           uv = half2(pos.x - fpos.x, pos.y - fpos.y);
    return uv;
}


// =================================================================================
// KERNEL 1: G-BUFFER & DIRECT LIGHTING
// =================================================================================
// This kernel traces the primary ray, handles water reflections, 
// shadows from the sun, and outputs data for the GI pass.
kernel void GBufferAndDirectLight(
    // --- Outputs ---
    texture2d<float, access::write> texDirectLight [[texture(0)]], // RGB = Lit Color
    texture2d<float, access::write> texAlbedo      [[texture(1)]], // RGB = Surface Color
    texture2d<float, access::write> texNormal      [[texture(2)]], // RGB = Encoded Normal
    texture2d<float, access::write> texMotion      [[texture(3)]], // RG = Motion Vector
    texture2d<float, access::write> texDepth       [[texture(4)]], // R = Depth

    // --- Inputs ---
    constant const CameraData& camera [[buffer(0)]],
    constant const FrameData& frame   [[buffer(1)]],
    
    texture3d<uint, access::read>     bitsTex     [[texture(5)]],
    texture3d<float, access::sample>  csdf        [[texture(6)]],
    texture2d<float, access::sample>  textureAtlas[[texture(7)]],
    texture2d<float, access::sample>  halfDistTex [[texture(8)]],

    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texDirectLight.get_width() || gid.y >= texDirectLight.get_height()) return;

    float2 pixelCenter = float2(gid) + 0.5f;
    float2 jitteredCoord = pixelCenter + camera.jitter; 
    
    float2 uv = jitteredCoord / float2(texDirectLight.get_width(), texDirectLight.get_height());
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    // Accelerator read
    constexpr sampler sLinear(filter::linear);
    float startDist = halfDistTex.sample(sLinear, uv).r;
    
    // Trace
    hitInfo hit = trace(camera.position, dir, startDist, bitsTex, csdf);

    float depth = 100000.0f;
    half3 finalDirectColor = half3(0.0h);
    half3 albedo = half3(0.0h);
    half3 normal = half3(0.0h); // default 0

    if (hit.hit) 
    {
        depth = length(hit.pos - camera.position);
        normal = hit.normal;
        
        // --- Material Handling ---
        bool isWater = (hit.pos.y < 31.001f && normal.y > 0.8h);

        // Calculate Motion Vector
        float2 motionVector = float2(0.0f);
        if (depth < 50000.0f) {
            float4 currentClipPos = camera.unjitteredViewProjection * float4(hit.pos, 1.0f);
            float4 previousClipPos = camera.prevUnjitteredViewProjection * float4(hit.pos, 1.0f);
            if (previousClipPos.w > 0.0f && currentClipPos.w > 0.0f) {
                float2 prevNDC = previousClipPos.xy / previousClipPos.w;
                float2 currNDC = currentClipPos.xy / currentClipPos.w;
                motionVector = currNDC - prevNDC;
                motionVector.y = -motionVector.y; // Flip Y for texture coords
            }
        }
        texMotion.write(float4(motionVector.x, motionVector.y, 0, 0), gid);

        // --- Shading ---
        if (isWater) 
        {
            // Water logic (Reflections + Blue tint)
            float nx_wave = fbm3D(hit.pos.x, hit.pos.z, frame.time, 3, 0.06f, 2.0f, 0.6f);
            float ny_wave = fbm3D(hit.pos.z, hit.pos.x, frame.time + 112.0f, 3, 0.06f, 2.0f, 0.6f);
            half3 distortedNormal = normalize(normal + half3(half(nx_wave) * 0.1h, 0.0h, half(ny_wave) * 0.1h));
            
            float3 reflDir = reflect(dir, (float3)distortedNormal);
            hitInfo reflHit = trace(hit.pos + (float3)normal * 0.05f, reflDir, 400.0f, bitsTex, csdf);
            
            half3 reflectColor;
            if (reflHit.hit) {
                half3 rAlbedo = sampleTexture(reflHit.uv, reflHit.pos, textureAtlas);
                bool rShadow = traceShadowAnyHit(reflHit.pos + (float3)reflHit.normal * 0.01f, frame.sunDirection, 50.0f, bitsTex, csdf);
                reflectColor = rAlbedo * c_sunColor * (rShadow ? 0.1h : 1.0h);
            } else {
                reflectColor = sampleSky(reflDir, frame.sunDirection);
            }
            
            half NdotV = max(dot(distortedNormal, -(half3)dir), 0.0h);
            half fresnel = 0.02h + (0.98h) * pow(1.0h - NdotV, 5.0h);
            finalDirectColor = lerp(make_half3(0.0h, 0.1h, 0.3h), reflectColor, fresnel);
            
            // For GI: Water has very low diffuse albedo (it absorbs light), mostly specular
            albedo = half3(0.05h, 0.1h, 0.2h); 
        } 
        else 
        {
            half2 localUV = reconstructUV(hit.pos, normal);
            albedo = sampleTexture(localUV, hit.pos, textureAtlas);
            
            bool isShadowed = traceShadowAnyHit(hit.pos + (float3)(normal * 1e-3h), frame.sunDirection, 2000.0f, bitsTex, csdf);
            
            half diffuse = max(dot(normal, (half3)frame.sunDirection), 0.0h);
            finalDirectColor = c_sunColor * diffuse * (isShadowed ? 0.1h : 1.0h);
        }
    } 
    else 
    {
        finalDirectColor = sampleSky(dir, frame.sunDirection);
        texMotion.write(float4(0,0,0,0), gid);
    }

    // Write Outputs
    texDirectLight.write(float4((float3)finalDirectColor, 1.0f), gid);
    texAlbedo.write(float4((float3)albedo, 1.0f), gid);

    texNormal.write(float4(((float3)normal), 1.0f), gid); 
    texDepth.write(float4(depth), gid);
}


// =================================================================================
// KERNEL 2: INDIRECT BOUNCE (Path Tracing 1spp)
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
    texture2d<float, access::sample> textureAtlas[[texture(5)]],
    
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texRawIndirect.get_width() || gid.y >= texRawIndirect.get_height()) return;
    float depth = texDepth.read(gid).r;
    
    if (depth > 5000.0f) {
        texRawIndirect.write(float4(0,0,0,0), gid);
        return;
    }

    half3 normal = (half3)texNormal.read(gid).rgb;
    
    // 2. Reconstruct Position
    float2 uv = (float2(gid) + 0.5f) / float2(texRawIndirect.get_width(), texRawIndirect.get_height());
    float3 pos = reconstructPos(depth, uv, camera);

    // 3. Initialize Random State (Temporal Jitter)
    uint seed = (gid.y * texRawIndirect.get_width() + gid.x) + uint(frame.time * 1000.0f);
    
    // 4. Create Orthonormal Basis (Tangent, Bitangent) around Normal
    float3 N = (float3)normal;
    float3 helper = abs(N.x) > 0.99f ? float3(0,0,1) : float3(1,0,0);
    float3 Tangent = normalize(cross(N, helper));
    float3 Bitangent = cross(N, Tangent);

    // 5. Cosine-Weighted Hemisphere Sampling (Ideally stratified)
    float r1 = rand_float(seed);
    float r2 = rand_float(seed);
    
    float phi = 2.0f * 3.14159f * r1;
    float sqr2 = sqrt(r2);
    float3 localDir = float3(sqr2 * cos(phi), sqrt(1.0f - r2), sqr2 * sin(phi));

    // Transform to World Space (Note: localDir.y is 'up' aligned with Normal)
    float3 rayDir = localDir.x * Tangent + localDir.y * N + localDir.z * Bitangent;
    rayDir = normalize(rayDir);

    hitInfo hit = trace(pos + (float3)normal * 0.05f, rayDir, 64.0f, bitsTex, csdf);
    
    half3 incomingLight = half3(0.0h);

    if (hit.hit) {
        // Shadow check for the bounced point
        bool isShadowed = traceShadowAnyHit(hit.pos + (float3)hit.normal * 0.01f, frame.sunDirection, 1000.0f, bitsTex, csdf);
        
        half2 hitUV = reconstructUV(hit.pos, hit.normal);
        half3 hitAlbedo = sampleTexture(hitUV, hit.pos, textureAtlas);
        
        half diffuse = max(dot(hit.normal, (half3)frame.sunDirection), 0.0h);
        
        incomingLight = c_sunColor * diffuse * (isShadowed ? 0.1h : 1.0h);
    } else {
        incomingLight = sampleSky(rayDir, frame.sunDirection);
    }
    
    texRawIndirect.write(float4((float3)incomingLight, 1.0f), gid);
}

// =================================================================================
// KERNEL 3: TEMPORAL ACCUMULATION (Reprojection)
// =================================================================================
kernel void TemporalAccumulation(
    texture2d<float, access::write> texAccum      [[texture(0)]],
    texture2d<float, access::read>  texRawIndirect[[texture(1)]],
    texture2d<float, access::sample> texHistory   [[texture(2)]], 
    texture2d<float, access::read>  texMotion     [[texture(3)]],
    texture2d<float, access::read>  texDepth      [[texture(4)]],
    // CHANGE: Use access::read instead of sample for precise depth lookup
    texture2d<float, access::read>  texPrevDepth  [[texture(5)]], 
    texture2d<float, access::read>  texDirect     [[texture(6)]],

    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texAccum.get_width() || gid.y >= texAccum.get_height()) return;

    float3 direct = texDirect.read(gid).rgb;
    float3 indirect = texRawIndirect.read(gid).rgb;
    float3 current = direct + indirect; // TOTAL LIGHTING
    
    float2 motion = texMotion.read(gid).xy;
    float currentDepth = texDepth.read(gid).r;
    
    // Calculate Previous UV
    float2 uv = (float2(gid) + 0.5f) / float2(texAccum.get_width(), texAccum.get_height());
    float2 prevUV = uv - motion;

    // Check bounds
    bool validHistory = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f);
    
    float3 history = current; 
    float historyWeight = 0.0f; // Default to 0 (reject)

    if (validHistory) 
    {
        // Use Linear for Color to look smooth (This is fine)
        constexpr sampler sLinear(filter::linear);
        history = texHistory.sample(sLinear, prevUV).rgb;
        
        uint2 prevCoords = uint2(prevUV.x * texPrevDepth.get_width(), prevUV.y * texPrevDepth.get_height());
        
        // Clamp to ensure we don't crash (though validHistory check should cover this)
        prevCoords.x = min(prevCoords.x, texPrevDepth.get_width() - 1);
        prevCoords.y = min(prevCoords.y, texPrevDepth.get_height() - 1);

        float prevDepth = texPrevDepth.read(prevCoords).r;
        
        // --- DEPTH CHECK ---
        float depthDiff = abs(currentDepth - prevDepth);
        
        // Relaxed threshold for Voxel scenes (voxels have hard edges that jump in depth)
        // If the difference is less than 10% of the total depth, accept it.
        float relativeDiff = depthDiff / (currentDepth + 0.001f);

        if (relativeDiff < 0.1f) 
        {
            historyWeight = 0.9f; 
            
            // Reduce ghosting during fast movement
            if (length(motion) > 0.001f) historyWeight = 0.85f;
        }
        else 
        {
            // Disocclusion: Fallback to current
            historyWeight = 0.0f; 
        }
    }
    
    // DEBUG: Uncomment this to visualize rejection.
    // Red = Rejection, Green = Good Accumulation
    // 
    //texAccum.write(float4(historyWeight, validHistory ? 1.0 : 0.0, 0.0, 1.0f), gid); return;

    float3 result = mix(current, history, historyWeight);
    texAccum.write(float4(result, 1.0f), gid);
}
// =================================================================================
// KERNEL 4: SPATIAL DENOISING (Bilateral Filter)
// =================================================================================
kernel void BilateralDenoise(
    texture2d<float, access::write> texDenoised [[texture(0)]],
    texture2d<float, access::read>  texAccum    [[texture(1)]],
    texture2d<float, access::read>  texNormal   [[texture(2)]],
    texture2d<float, access::read>  texDepth    [[texture(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texDenoised.get_width() || gid.y >= texDenoised.get_height()) return;

    float3 centerColor = texAccum.read(gid).rgb;
    float3 centerNormal = texNormal.read(gid).rgb;
    float centerDepth = texDepth.read(gid).r;

    float3 sum = float3(0.0f);
    float weightSum = 0.0f;

    // TWEAK: Relax constraints as objects get further away.
    // At 500 units away, normal differences matter less than at 5 units.
    float distanceFactor = clamp(centerDepth / 200.0f, 0.0f, 1.0f);
    float normalPower = mix(4.0f, 0.1f, distanceFactor); // High sensitivity close, low far
    float depthPhi = mix(2.0f, 0.1f, distanceFactor);    // High sensitivity close, low far

    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            uint2 tapCoord = uint2(gid.x + x, gid.y + y);
            if (tapCoord.x >= texDenoised.get_width() || tapCoord.y >= texDenoised.get_height()) continue;

            float3 tapColor = texAccum.read(tapCoord).rgb;
            float3 tapNormal = texNormal.read(tapCoord).rgb;
            float tapDepth = texDepth.read(tapCoord).r;

            // 1. Spatial Weight (Gaussian)
            float spatialW = exp(-(float)(x*x + y*y) / 4.0f);
            
            // 2. Normal Weight (Relaxed by distance)
            float dotP = max(dot(centerNormal, tapNormal), 0.0f);
            float normalW = pow(dotP, normalPower);

            // 3. Depth Weight (Relative Difference)
            // Using absolute difference fails in the distance because 
            // 0.1 unit difference is huge close up, but microscopic far away.
            // We use relative difference instead.
            float diff = abs(centerDepth - tapDepth);
            float relativeDiff = diff / (centerDepth + 0.001f);
            float depthW = exp(-relativeDiff * relativeDiff * 100.0f * depthPhi);

            float totalWeight = spatialW * normalW * depthW;

            sum += tapColor * totalWeight;
            weightSum += totalWeight;
        }
    }

    if (weightSum < 1e-4f) weightSum = 1.0f;
    texDenoised.write(float4(sum / weightSum, 1.0f), gid);
}

// =================================================================================
// KERNEL 5: COMPOSITE (Combine Direct + Indirect)
// =================================================================================
kernel void Composite(
    texture2d<float, access::write> texFinal   [[texture(0)]],
    texture2d<float, access::read>  texDirect  [[texture(1)]],
    texture2d<float, access::read>  texAccum   [[texture(2)]], // Denoised
    texture2d<float, access::read>  texAlbedo  [[texture(3)]],
    texture2d<float, access::read>  texDepth   [[texture(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texFinal.get_width() || gid.y >= texFinal.get_height()) return;

    float3 totalLight = texAccum.read(gid).rgb;
    float3 albedo = texAlbedo.read(gid).rgb;
    float depth = texDepth.read(gid).r;

    float3 finalColor = totalLight * albedo; 

    finalColor *= 1.5f; 

    float fogAmount = clamp(1.0f - exp(-depth * 0.0004f), 0.0f, 1.0f);
    float3 fogColor = float3(0.95f, 0.95f, 1.0f);
    
    if (depth > 50000.0f) {
        texFinal.write(float4(totalLight, 1.0f), gid);
    } else {
        finalColor = mix(finalColor, fogColor, fogAmount);
        texFinal.write(float4(finalColor, 1.0f), gid);
    }
}


kernel void distApproximationKernel( //1,5 gigaray per second
    texture2d<float, access::write> distTex [[texture(0)]],
    constant const CameraData& camera [[buffer(0)]],
    constant const FrameData& frame [[buffer(1)]],
    
    texture3d<uint, access::read> bitsTex [[texture(2)]],
    texture3d<float, access::sample> csdf [[texture(3)]], 
    uint2 gid [[thread_position_in_grid]])
{
    uint width = distTex.get_width();
    uint height = distTex.get_height();
    
    if (gid.x >= width || gid.y >= height) return;

    float2 uv = (float2(gid) + 0.5f) / float2(width, height);
    float2 ndc = uv * 2.0f - 1.0f; 
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    // Move to start distance
    float3 currentPos = camera.position;

    // Precompute DDA constants (invariant for the ray)
    const float3 deltaDist = make_float3(
        abs(dir.x) > 1e-5f ? abs(1.0f / dir.x) : 1.0e30f,
        abs(dir.y) > 1e-5f ? abs(1.0f / dir.y) : 1.0e30f,
        abs(dir.z) > 1e-5f ? abs(1.0f / dir.z) : 1.0e30f
    );

    const int3 step = make_int3(
        dir.x > 0.0f ? 1 : -1,
        dir.y > 0.0f ? 1 : -1,
        dir.z > 0.0f ? 1 : -1
    );

    for (int majorIteration = 0; majorIteration < 10; majorIteration++)
    {
        currentPos = approximateCSDF(currentPos, dir, csdf);

        float3 fpos = floor3(currentPos);
        int3 ipos = to_int3(currentPos);
        
        float3 tMax;
        tMax.x = ((step.x > 0) ? (fpos.x + 1.0f - currentPos.x) : (currentPos.x - fpos.x)) * deltaDist.x;
        tMax.y = ((step.y > 0) ? (fpos.y + 1.0f - currentPos.y) : (currentPos.y - fpos.y)) * deltaDist.y;
        tMax.z = ((step.z > 0) ? (fpos.z + 1.0f - currentPos.z) : (currentPos.z - fpos.z)) * deltaDist.z;

        int mask = -1;
        float distTraveledInDDA = 0.0f;
        bool hitFound = false;

        for (int i = 0; i < 8; i++) 
        {            
            if (ipos.x < 0 || ipos.y < 0 || ipos.z < 0 || 
                ipos.x >= (int)SIZEX || ipos.y >= (int)SIZEY || ipos.z >= (int)SIZEZ) {
                majorIteration = 10;
                i = 10;
                break;
            }

            if (IsSolid(ipos, bitsTex)) 
            {
                float tVal = 0.0f;
                if (mask == 0) {
                    tVal = tMax.x - deltaDist.x;
                } else if (mask == 1) {
                    tVal = tMax.y - deltaDist.y;
                } else {
                    tVal = tMax.z - deltaDist.z;
                }
                distTex.write(float4(length(currentPos + dir * tVal - camera.position), 0, 0, 0), gid);
                return;
            }
            
            if (tMax.x < tMax.y) {
                if (tMax.x < tMax.z) { 
                    distTraveledInDDA = tMax.x;
                    tMax.x += deltaDist.x; ipos.x += step.x; mask = 0; 
                } else { 
                    distTraveledInDDA = tMax.z;
                    tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; 
                }
            } else {
                if (tMax.y < tMax.z) { 
                    distTraveledInDDA = tMax.y;
                    tMax.y += deltaDist.y; ipos.y += step.y; mask = 1; 
                } else { 
                    distTraveledInDDA = tMax.z;
                    tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; 
                }
            }
        }
        currentPos += dir * (distTraveledInDDA + 0.0001f);
    }        
    distTex.write(float4(HALF_MAX, 0, 0, 1), gid);
}