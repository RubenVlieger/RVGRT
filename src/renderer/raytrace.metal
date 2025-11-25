#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
using namespace metal;


struct GBufferData {
    float depth; 
    
    // xyz = normal, w = shadow value
    half4 normal_shadow; 
};
inline float3 reconstructPos(float3 camPos, float3 rayDir, float depth) {
    return camPos + rayDir * depth;
}


inline half2 reconstructUV(float3 pos, half3 normal) {
    float3 fpos = floor(pos);
    half2 uv;

    if (abs(normal.x) > 0.5h) {
        uv = half2(pos.y - fpos.y, pos.z - fpos.z);
    } else if (abs(normal.y) > 0.5h) {
        uv = half2(pos.x - fpos.x, pos.z - fpos.z);
    } else {
        uv = half2(pos.x - fpos.x, pos.y - fpos.y);
    }
    return uv;
}



kernel void tiledDeferredRaytraceKernel(
    // --- Outputs ---
    texture2d<float, access::write> outputTexture [[texture(0)]],

    // --- Inputs ---
    constant const CameraData& camera [[buffer(0)]],
    constant const FrameData& frame   [[buffer(1)]],
    
    texture3d<uint, access::read>     bitsTex     [[texture(2)]],
    texture3d<float, access::sample>  csdf        [[texture(3)]],
    texture2d<float, access::sample>  textureAtlas[[texture(5)]],
    texture2d<float, access::sample>  halfDistTex [[texture(6)]],

    // --- Threading System ---
    ushort2 gid [[thread_position_in_grid]],
    ushort2 tid [[thread_position_in_threadgroup]])
{
    // ALLOCATE TILE MEMORY (L1)
    // Optimized struct size significantly improves SIMD occupancy
    threadgroup GBufferData tileData[16][16];

    bool valid = (gid.x < outputTexture.get_width() && gid.y < outputTexture.get_height());
    
    // Pre-calculate Ray Direction (Used in both stages)
    float2 uv = (float2(gid) + 0.5f) / float2(outputTexture.get_width(), outputTexture.get_height());
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    // =========================================================================
    // STAGE 1: TRACE & COMPACT WRITE
    // =========================================================================
    if (valid) 
    {
        constexpr sampler sLinear(filter::linear);
        float startDist = halfDistTex.sample(sLinear, uv).r;

        // Trace Primary Ray
        hitInfo hit = trace(camera.position, dir, startDist, bitsTex, csdf);

        float depth = 100000.0f;
        half4 normal_shadow = half4(0.0h, 0.0h, 0.0h, 1.0h); // w=1.0 (Lit)

        if (hit.hit) {
            depth = length(hit.pos - camera.position);
            
            // Shadow Trace (Inline)
            // Optimization: Reduced shadow trace distance to 500.0f. 
            // Most occlusions happen near the object.
            bool isShadowed = traceShadowAnyHit(hit.pos + (float3)(hit.normal * 1e-3h),
                                               frame.sunDirection, 500.0f, bitsTex, csdf);
            
            normal_shadow = half4((half3)hit.normal, isShadowed ? 0.1h : 1.0h);
        }

        // Write Compact Data
        tileData[tid.y][tid.x].depth = depth;
        tileData[tid.y][tid.x].normal_shadow = normal_shadow;
    }

    // Barrier to sync tile
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // STAGE 2: READ & SHADE
    // =========================================================================
    if (!valid) return;

    // 1. Read Compact Data
    GBufferData inData = tileData[tid.y][tid.x];
    float depth = inData.depth;

    // Early exit for Sky
    if (depth > 50000.0f) { 
        half3 sky = sampleSky(dir, frame.sunDirection);
        outputTexture.write(float4((float3)sky, 1.0f), gid);
        return;
    }

    // Reconstruct Position from Depth (Cheaper than storing float4)
    float3 pos = reconstructPos(camera.position, dir, depth);
    
    // Unpack Normal/Shadow
    half3 normal = inData.normal_shadow.xyz;
    half shadowVal = inData.normal_shadow.w;

    // 2. Shading
    half2 localUV = reconstructUV(pos, normal);
    half3 baseColor = sampleTexture(localUV, pos, textureAtlas);

    half diffuse = max(dot(normal, (half3)frame.sunDirection), 0.05h);
    
    // Use FMAD (fused multiply-add) where possible implicitly
    half3 directLight = baseColor * c_sunColor * diffuse * shadowVal;
    half3 ambient = sampleSky((float3)normal, frame.sunDirection) * 0.1h * baseColor;
    half3 finalColor = directLight + ambient;

    // 3. Water Reflection (Branch)
    // Optimization: Use half-precision for Y check
    if (pos.y < 31.001f && normal.y > 0.8h) 
    {
        float nx_wave = fbm3D(pos.x, pos.z, frame.time, 3, 0.06f, 2.0f, 0.6f);
        float ny_wave = fbm3D(pos.z, pos.x, frame.time + 112.0f, 3, 0.06f, 2.0f, 0.6f);
        half3 distortedNormal = normalize(normal + half3(half(nx_wave) * 0.1h, 0.0h, half(ny_wave) * 0.1h));

        float3 reflDir = reflect(dir, (float3)distortedNormal);
        // Optimization: Offset origin slightly more to prevent self-intersection noise
        float3 reflOrigin = pos + (float3)normal * 0.02f; 

        // Trace Reflection
        // Optimization: Max dist reduced to 600.0. Reflection precision at distance matters less.
        hitInfo reflHit = trace(reflOrigin, reflDir, 600.0f, bitsTex, csdf);

        half3 reflectionColor;
        if (reflHit.hit) {
            half3 reflAlbedo = sampleTexture(reflHit.uv, reflHit.pos, textureAtlas);
            half reflDiffuse = max(dot(reflHit.normal, (half3)frame.sunDirection), 0.2h);        
            
            // Optimization: Simplified reflection shadow (short distance check)
            bool reflShadowed = traceShadowAnyHit(reflHit.pos + (float3)(reflHit.normal * 1e-3h),
                                                frame.sunDirection, 50.0f, bitsTex, csdf);
                                                
            reflectionColor = reflAlbedo * c_sunColor * reflDiffuse * (reflShadowed ? 0.1h : 1.0h);
        } else {
            reflectionColor = sampleSky(reflDir, frame.sunDirection);
        }

        half NdotV = max(dot(distortedNormal, -(half3)dir), 0.0h);
        half fresnel = 0.02h + (0.98h) * pow(1.0h - NdotV, 5.0h);

        finalColor = lerp(make_half3(0.0h, 0.1h, 0.3h), reflectionColor, fresnel);
    }

    // 4. Fog (Calculated on half precision depth)
    half fogAmount = clamp(1.0h - exp(-half(depth) * 0.0004h), 0.0h, 1.0h);
    half3 fogColor = make_half3(0.95h, 0.95h, 1.0h);
    
    finalColor = lerp(finalColor, fogColor, fogAmount);

    outputTexture.write(float4((float3)finalColor, 1.0f), gid);
}

kernel void gbuffer_kernel( // 1,5ms -> 4,8ms (1,3 -> 3,5ms)
    // WRITABLE TEXTURES (The G-Buffer)
    texture2d<float, access::write> gPos [[texture(0)]],
    texture2d<float, access::write> gNorm [[texture(1)]],
    
    // INPUTS
    constant const CameraData& camera [[buffer(0)]],
    texture3d<uint, access::read> bitsTex [[texture(2)]],
    texture3d<float, access::sample> csdf [[texture(3)]], 
    texture2d<float, access::sample> halfDistTex [[texture(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = gPos.get_width();
    uint height = gPos.get_height();
    if (gid.x >= width || gid.y >= height) return;

    float2 uv = (float2(gid) + 0.5f) / float2(width, height);
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);
    
    constexpr sampler sLinear(filter::linear);
    float startDist = halfDistTex.sample(sLinear, uv).r;

    hitInfo hit = trace(camera.position, dir, startDist, bitsTex, csdf);

    if (hit.hit) {
        gPos.write(float4(hit.pos, 1.0), gid);
        gNorm.write(float4((float3)hit.normal, 0.0), gid);
    } else {
        gPos.write(float4(0,0,0,0), gid); 
        gNorm.write(float4(0,0,0,0), gid);
    }
}

kernel void shading_kernel(
    // OUTPUT
    texture2d<float, access::write> outputTexture [[texture(0)]],
    
    // INPUT BUFFERS (G-Buffer & Light Buffers)
    texture2d<float, access::read> gPos [[texture(1)]],
    texture2d<float, access::read> gNorm [[texture(2)]],
    texture2d<float, access::read> shadowMask [[texture(3)]],
    texture2d<float, access::read> reflectionTex [[texture(4)]],
    
    // ASSETS
    texture2d<float, access::sample> textureAtlas [[texture(5)]],
    
    // CONSTANTS
    constant const CameraData& camera [[buffer(0)]],
    constant const FrameData& frame [[buffer(1)]],
    
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) return;

    // 1. Read G-Buffer Position
    float4 posData = gPos.read(gid);
    
    // If w == 0, it's a miss (Sky)
    if (posData.w < 0.5f) {
        float2 uv = (float2(gid) + 0.5f) / float2(outputTexture.get_width(), outputTexture.get_height());
        float2 ndc = uv * 2.0f - 1.0f;
        float3 viewDir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);
        half3 sky = sampleSky(viewDir, frame.sunDirection);
        outputTexture.write(float4((float3)sky, 1.0f), gid);
        return;
    }

    float3 pos = posData.xyz;
    half3 normal = (half3)gNorm.read(gid).xyz;

    half shadowVal = (half)shadowMask.read(gid).r;

    half2 localUV = reconstructUV(pos, normal);
    half3 baseColor = sampleTexture(localUV, pos, textureAtlas);

    half diffuse = max(dot(normal, (half3)frame.sunDirection), 0.05h);
    half3 directLight = baseColor * diffuse * c_sunColor * shadowVal;
    
    half3 ambient = sampleSky((float3)normal, frame.sunDirection) * 0.1h * baseColor;
    
    half3 finalColor = directLight + ambient;
    float4 reflData = reflectionTex.read(gid);
    half fresnel = (half)reflData.w;

    if (fresnel > 0.0h) {
        half3 reflectionColor = (half3)reflData.xyz;
        half3 waterBase = make_half3(0.0h, 0.1h, 0.3h);
        
        finalColor = lerp(waterBase, reflectionColor, fresnel);
    }

    float dist = length(pos - camera.position);
    half fogAmount = clamp(1.0h - exp(-half(dist) * 0.0004h), 0.0h, 1.0h);
    half3 fogColor = make_half3(0.95h, 0.95h, 1.0h);
    
    finalColor = lerp(finalColor, fogColor, fogAmount);

    outputTexture.write(float4((float3)finalColor, 1.0f), gid);
}



inline float2 intersectAABB_Opt(float3 rayOrigin, float3 invDir, float3 boxMax) {
    float3 tMin = (0.0f - rayOrigin) * invDir;
    float3 tMax = (boxMax - rayOrigin) * invDir;
    float3 t1 = min(tMin, tMax);
    float3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return float2(tNear, tFar);
}

inline float2 intersectWorldAABB_Fast(float3 rayOrigin, float3 invRayDir) {
    float3 t0 = (float3(0.0f) - rayOrigin) * invRayDir;
    float3 t1 = (float3(SIZEX, SIZEY, SIZEZ) - rayOrigin) * invRayDir;
    float3 tMax = max(t0, t1);
    float3 tMin = min(t0, t1);
    return float2(max(max(tMin.x, tMin.y), tMin.z), min(min(tMax.x, tMax.y), tMax.z));
}





// kernel void distApproximationKernel(
//     texture2d<float, access::write> distTex [[texture(0)]],
//     constant const CameraData& camera [[buffer(0)]],
//     constant const FrameData& frame [[buffer(1)]],
//     texture3d<uint, access::read> bitsTex [[texture(2)]],
//     texture3d<float, access::sample> csdf [[texture(3)]], 
//     uint2 gid [[thread_position_in_grid]])
// {
//     const uint width = distTex.get_width();
//     const uint height = distTex.get_height();
//     if (gid.x >= width || gid.y >= height) return;

//     // 1. Setup
//     const float2 uv = (float2(gid) + 0.5f) / float2(width, height);
//     const float2 ndc = uv * 2.0f - 1.0f; 
//     const float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);
    
//     // 2. Optimization: AABB Intersection (Skip empty space)
//     // This is the only part of the "new" code we keep. It saves massive time for sky rays.
//     const float3 invDir = 1.0f / (select(sign(dir) * 1e-5f, dir, abs(dir) > 1e-5f));
//     float2 worldT = intersectAABB_Opt(camera.position, invDir, float3(SIZEX, SIZEY, SIZEZ));
    
//     // If ray misses world or is behind us, exit immediately.
//     if (worldT.x > worldT.y || worldT.y < 0.0f) {
//         distTex.write(float4(HALF_MAX, 0, 0, 1), gid);
//         return;
//     }

//     // Initialize position at the box entry point
//     // This allows us to remove the "if (ipos.x < 0)" check from the inner loop!
//     float t = max(0.0f, worldT.x);
//     const float tEnd = worldT.y;
//     float3 currentPos = camera.position + dir * t;

//     // 3. Precompute DDA Constants (Your original robust logic)
//     const float3 deltaDist = abs(invDir);
//     const int3 step = int3(sign(dir));
//     const float3 stepF = float3(step); // Used for tMax calc

//     // 4. Main Loop
//     for (int majorIteration = 0; majorIteration < 12; majorIteration++)
//     {
//         if (t >= tEnd) break;

//         // --- A. Sphere Trace Step ---
//         // Instead of calling 'approximateCSDF' (which might loop internally), 
//         // we do the logic inline to ensure we control the instruction count.
        
//         // We iterate sphere tracing until we are close enough to a block to need DDA.
//         for(int k=0; k<4; k++) {
//             // Note: approximateCSDF usually returns a new position. 
//             // If your function does texture lookups, ensure it uses normalized coords correctly.
//             // Assuming approximateCSDF handles the CSDF lookup and step:
//             float3 nextPos = approximateCSDF(currentPos, dir, csdf);
//             float distMoved = length(nextPos - currentPos);
            
//             t += distMoved;
//             currentPos = nextPos;
            
//             // If we moved less than a voxel diagonal, stop sphere tracing and start DDA
//             if(distMoved < 1.5f) break; 
//             if(t >= tEnd) goto EndKernel; // Break out of nested loops
//         }

//         // --- B. DDA Setup (Your robust logic) ---
//         int3 ipos = to_int3(floor3(currentPos));
//         float3 fpos = make_float3(ipos);
        
//         float3 tMax;
//         // Your original formula (Robust)
//         tMax.x = ((step.x > 0) ? (fpos.x + 1.0f - currentPos.x) : (currentPos.x - fpos.x)) * deltaDist.x;
//         tMax.y = ((step.y > 0) ? (fpos.y + 1.0f - currentPos.y) : (currentPos.y - fpos.y)) * deltaDist.y;
//         tMax.z = ((step.z > 0) ? (fpos.z + 1.0f - currentPos.z) : (currentPos.z - fpos.z)) * deltaDist.z;

//         int mask = -1;
//         float distTraveledInDDA = 0.0f;

//         // --- C. Inner DDA Loop ---
//         for (int i = 0; i < 8; i++) 
//         {
//             // Bounds check removed! (We rely on tEnd and worldT)
//             // However, to be safe against float drift, we clamp the read index.
//             uint3 superPos = uint3(uint(ipos.x) >> 2, uint(ipos.y) >> 2, uint(ipos.z) >> 1);
            
//             // Safe Read (Hardware handles out-of-bounds read by returning 0 usually, but explicit is safe)
//             // If your texture is POT, masking is faster: (ipos.x & 1023) etc.
//             // Here we assume valid range due to tEnd logic.
//             uint blockBits = bitsTex.read(superPos).r;
            
//             uint bitIndex = (ipos.x & 3) | ((ipos.y & 3) << 2) | ((ipos.z & 1) << 4);

//             if ((blockBits & (1u << bitIndex)) != 0) 
//             {
//                 float tVal = 0.0f;
                
//                 // Hit Calculation (Your original logic)
//                 if (mask == -1) {
//                     // Started inside block (rare but possible)
//                     float3 tBack = deltaDist - tMax;
//                     tVal = -max(max(tBack.x, tBack.y), tBack.z);
//                 } else if (mask == 0) {
//                     tVal = tMax.x - deltaDist.x;
//                 } else if (mask == 1) {
//                     tVal = tMax.y - deltaDist.y;
//                 } else {
//                     tVal = tMax.z - deltaDist.z;
//                 }
                
//                 // Write absolute distance
//                 distTex.write(float4(t + tVal, 0, 0, 0), gid);
//                 return;
//             }
            
//             // Step
//             if (tMax.x < tMax.y) {
//                 if (tMax.x < tMax.z) { 
//                     distTraveledInDDA = tMax.x;
//                     tMax.x += deltaDist.x; ipos.x += step.x; mask = 0; 
//                 } else { 
//                     distTraveledInDDA = tMax.z;
//                     tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; 
//                 }
//             } else {
//                 if (tMax.y < tMax.z) { 
//                     distTraveledInDDA = tMax.y;
//                     tMax.y += deltaDist.y; ipos.y += step.y; mask = 1; 
//                 } else { 
//                     distTraveledInDDA = tMax.z;
//                     tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; 
//                 }
//             }
//         }
        
//         // Advance position for the next sphere trace
//         // Adding epsilon to ensure we don't get stuck on the face we just hit
//         float advance = distTraveledInDDA + 0.001f;
//         t += advance;
//         currentPos += dir * advance;
//     }        

//     distTex.write(float4(HALF_MAX, 0, 0, 1), gid);
// }

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

//
//kernel void distApproximationKernel(
//    texture2d<float, access::write> distTex [[texture(0)]],
//    constant const CameraData& camera [[buffer(0)]],
//    constant const FrameData& frame [[buffer(1)]],
//    texture3d<uint, access::read> bitsTex [[texture(2)]],
//    texture3d<float, access::sample> csdf [[texture(3)]], 
//    uint2 gid [[thread_position_in_grid]])
//{
//    uint width = distTex.get_width();
//    uint height = distTex.get_height();
//    if (gid.x >= width || gid.y >= height) return;
//
//    float2 uv = (float2(gid) + 0.5f) / float2(width, height);
//    float2 ndc = uv * 2.0f - 1.0f; 
//    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);
//
//    float3 invDir = 1.0f / select(sign(dir) * 1e-5f, dir, abs(dir) > 1e-5f);
//    float2 boxT = intersectWorldAABB_Fast(camera.position, invDir);
//    
//    bool miss = (boxT.x > boxT.y) || (boxT.y < 0.0f);
//
//    if (simd_all(miss)) {
//        distTex.write(float4(HALF_MAX, 0, 0, 1), gid);
//        return;
//    }
//
//    if (miss) {
//        distTex.write(float4(HALF_MAX, 0, 0, 1), gid);
//        return;
//    }
//
//    float tCurrent = max(0.0f, boxT.x);
//    float tEnd = boxT.y;
//
//    float3 currentPos = camera.position + dir * tCurrent;
//
//    const float3 deltaDist = abs(invDir);
//    const int3 step = int3(sign(dir));
//
//    bool active = true;
//    bool hit = false;
//    float hitDist = HALF_MAX;
//
//    for (int majorIteration = 0; majorIteration < 10; majorIteration++)
//    {
//        if (simd_all(!active)) break;
//
//        if (active) 
//        {
//            currentPos = approximateCSDF(currentPos, dir, csdf);
//            
//            float distFromCam = length(currentPos - camera.position);
//            if (distFromCam > tEnd) {
//                active = false;
//                continue;
//            }
//
//            float3 fpos = floor3(currentPos);
//            int3 ipos = to_int3(currentPos);
//            
//            float3 tMax;
//            tMax.x = ((step.x > 0) ? (fpos.x + 1.0f - currentPos.x) : (currentPos.x - fpos.x)) * deltaDist.x;
//            tMax.y = ((step.y > 0) ? (fpos.y + 1.0f - currentPos.y) : (currentPos.y - fpos.y)) * deltaDist.y;
//            tMax.z = ((step.z > 0) ? (fpos.z + 1.0f - currentPos.z) : (currentPos.z - fpos.z)) * deltaDist.z;
//
//            float distTraveledInDDA = 0.0f;
//            int mask = -1;
//
//            for (int i = 0; i < 8; i++) 
//            {            
//                if (ipos.x < 0 || ipos.y < 0 || ipos.z < 0 || 
//                    ipos.x >= (int)SIZEX || ipos.y >= (int)SIZEY || ipos.z >= (int)SIZEZ) 
//                {
//                    active = false;
//                    break;
//                }
//
//                if (IsSolid(ipos, bitsTex)) 
//                {
//                    float tVal = 0.0f;
//                    if (mask == 0) tVal = tMax.x - deltaDist.x;
//                    else if (mask == 1) tVal = tMax.y - deltaDist.y;
//                    else tVal = tMax.z - deltaDist.z;
//                    
//                    hitDist = length((currentPos + dir * tVal) - camera.position);
//                    
//                    hit = true;
//                    active = false;
//                    break;
//                }
//                
//                if (tMax.x < tMax.y) {
//                    if (tMax.x < tMax.z) { 
//                        distTraveledInDDA = tMax.x;
//                        tMax.x += deltaDist.x; ipos.x += step.x; mask = 0; 
//                    } else { 
//                        distTraveledInDDA = tMax.z;
//                        tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; 
//                    }
//                } else {
//                    if (tMax.y < tMax.z) { 
//                        distTraveledInDDA = tMax.y;
//                        tMax.y += deltaDist.y; ipos.y += step.y; mask = 1; 
//                    } else { 
//                        distTraveledInDDA = tMax.z;
//                        tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; 
//                    }
//                }
//            }
//            
//            currentPos += dir * (distTraveledInDDA + 0.0001f);
//        }
//    }
//
//    if (hit) {
//        distTex.write(float4(max(0.0f, hitDist - 2.0f), 0, 0, 1), gid);
//    } else {
//        distTex.write(float4(HALF_MAX, 0, 0, 1), gid);
//    }
//
//}

//  kernel void distApproximationKernel( //1,3 gigaray per second
//      texture2d<float, access::write> distTex [[texture(0)]],
//      constant const CameraData& camera [[buffer(0)]],
//      constant const FrameData& frame [[buffer(1)]],
    
//      texture3d<uint, access::read> bitsTex [[texture(2)]],
//      texture3d<float, access::sample> csdf [[texture(3)]], 
//      uint2 gid [[thread_position_in_grid]])
//  {
//      constexpr sampler csdfSampler(coord::pixel, address::clamp_to_edge, filter::linear);

//      uint width = distTex.get_width();
//      uint height = distTex.get_height();
    
//      if (gid.x >= width || gid.y >= height) return;

//      float2 uv = (float2(gid) + 0.5f) / float2(width, height);
//      float2 ndc = uv * 2.0f - 1.0f; 
//      float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

//      // Move to start distance
//      float3 currentPos = camera.position;

//      // Precompute DDA constants (invariant for the ray)
//      const float3 deltaDist = make_float3(
//          abs(dir.x) > 1e-5f ? abs(1.0f / dir.x) : 1.0e30f,
//          abs(dir.y) > 1e-5f ? abs(1.0f / dir.y) : 1.0e30f,
//          abs(dir.z) > 1e-5f ? abs(1.0f / dir.z) : 1.0e30f
//      );

//      const int3 step = make_int3(
//          dir.x > 0.0f ? 1 : -1,
//          dir.y > 0.0f ? 1 : -1,
//          dir.z > 0.0f ? 1 : -1
//      );

//      for (int majorIteration = 0; majorIteration < 16; majorIteration++)
//      {
//          for(int i = 0; i < 8; ++i) {
//              float dist = ((csdf.sample(csdfSampler, currentPos * 0.5f).r - 0.5f)) * 2.0f;
//              if(dist < 1.25f) break;
//              currentPos += dir * dist;
//          }
//          float3 fpos = floor3(currentPos);
//          int3 ipos = to_int3(fpos);
        
//          float3 tMax;
//          tMax.x = ((step.x > 0) ? (fpos.x + 1.0f - currentPos.x) : (currentPos.x - fpos.x)) * deltaDist.x;
//          tMax.y = ((step.y > 0) ? (fpos.y + 1.0f - currentPos.y) : (currentPos.y - fpos.y)) * deltaDist.y;
//          tMax.z = ((step.z > 0) ? (fpos.z + 1.0f - currentPos.z) : (currentPos.z - fpos.z)) * deltaDist.z;

//          int mask = -1;
//          float distTraveledInDDA = 0.0f;

//          for (int i = 0; i < 8; i++) 
//          {            
//              if (ipos.x < 0 || ipos.y < 0 || ipos.z < 0 || 
//                  ipos.x >= (int)SIZEX || ipos.y >= (int)SIZEY || ipos.z >= (int)SIZEZ) {
//                  distTex.write(float4(0.0f, 0, 0, 0), gid);
//                  return;
//              }

//              uint32_t blockBits = bitsTex.read(uint3(ipos.x >> 2, ipos.y >> 2, ipos.z >> 1)).r;
//              if ((blockBits & (1u << (((ipos.x >> 2) & 3) | (ipos.y & 4) | ((ipos.z & 2) << 3)))) != 0) 
//              {
//                  float tVal = 0.0f;
//                  if (mask == 0) {
//                      distTex.write(float4(length(currentPos + dir * (tMax.x - deltaDist.x) - camera.position), 0, 0, 0), gid);
//                  } else if (mask == 1) {
//                      distTex.write(float4(length(currentPos + dir * (tMax.y - deltaDist.y) - camera.position), 0, 0, 0), gid);
//                  } else {
//                      distTex.write(float4(length(currentPos + dir * (tMax.z - deltaDist.z) - camera.position), 0, 0, 0), gid);
//                  }
//                  return;
//              }
            
//              if (tMax.x < tMax.y) {
//                  if (tMax.x < tMax.z) { 
//                      distTraveledInDDA = tMax.x;
//                      tMax.x += deltaDist.x; ipos.x += step.x; mask = 0; 
//                  } else { 
//                      distTraveledInDDA = tMax.z;
//                      tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; 
//                  }
//              } else {
//                  if (tMax.y < tMax.z) { 
//                      distTraveledInDDA = tMax.y;
//                      tMax.y += deltaDist.y; ipos.y += step.y; mask = 1; 
//                  } else { 
//                      distTraveledInDDA = tMax.z;
//                      tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; 
//                  }
//              }
//          }
//          currentPos += dir * (distTraveledInDDA + 0.0001f);
//      }        
//      distTex.write(float4(HALF_MAX, 0, 0, 1), gid);
//  }

// kernel void distApproximationKernel(
//     texture2d<float, access::write> distTex [[texture(0)]],
//     texture2d<float, access::write> shadowTex [[texture(1)]],
//     constant const CameraData& camera [[buffer(0)]],
//     constant const FrameData& frame [[buffer(1)]],

//     texture3d<uint, access::read> bitsTex [[texture(2)]],

//     texture3d<uint, access::read> csdf [[texture(3)]],
//     uint2 gid [[thread_position_in_grid]])
// {
//     uint width = distTex.get_width();
//     uint height = distTex.get_height();
    
//     if (gid.x >= width || gid.y >= height) return;

//     float2 uv = (float2(gid) + 0.5f) / float2(width, height);
    
//     float2 ndc = uv * 2.0f - 1.0f; 

//     float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

//     hitInfo hit = trace(camera.position, dir, 0.0f, bitsTex, csdf);
    
//     float dist = hit.hit ? length(hit.pos - camera.position) : 300.0f;
    
//     float shadowValue = 1.0f;
//     if(hit.hit)
//     {
//         hitInfo shadowHit = trace(hit.pos + (float3)(hit.normal * 0.1h), frame.sunDirection, 0.0f, bitsTex, csdf);
//         shadowValue = shadowHit.hit ? 0.2f : 1.0f;
//     }

//     distTex.write(float4(dist - 8.0f, 0, 0, 1), gid);
//     shadowTex.write(float4(shadowValue, 0, 0, 1), gid);
// }




// half3 old_deprecatedkernel(
//     float2 uv,
//     float startDist,
//     float shadowValue,
//     constant const CameraData& camera,
//     constant const FrameData& frame,
//     texture3d<uint, access::read> bitsTex,
//     texture3d<float, access::sample> csdf,
//     texture3d<uint, access::read> giData,
//     texture2d<float, access::sample> textureAtlas)
// {
//     float2 ndc = uv * 2.0f - 1.0f;
//     float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

//     hitInfo hit = trace(camera.position, dir, startDist, bitsTex, csdf);

//     half3 color = make_half3(0.0);

//     // logic for reflection code
//     if (hit.hit && hit.pos.y < 31.001f) 
//     {
//         half nx_wave = fbm3D_h(hit.pos.x, hit.pos.z, frame.time, 3, 0.06f, 2.0f, 0.6f);
//         half ny_wave = fbm3D_h(hit.pos.z, hit.pos.x, frame.time + 112.0f, 3, 0.06f, 2.0f, 0.6f);

//         half3 distortedNormal = normalize(hit.normal + half3(nx_wave * 0.1h, ny_wave * 0.1h, 0.h));

//         float3 reflDir = reflect(dir, (float3)distortedNormal);
//         float3 reflOrigin = hit.pos + (float3)hit.normal * 0.005f; 

//         hitInfo reflHit = trace(reflOrigin, reflDir, 0.0f, bitsTex, csdf);

//         half3 reflectionColor;
//         if (reflHit.hit) {
//             // A. Sample Texture
//             half3 reflAlbedo = (half3)sampleTexture(reflHit.uv, reflHit.pos, textureAtlas);
            

//             half diffuse = max(dot(reflHit.normal, (half3)frame.sunDirection), 0.0h);
            
//             reflectionColor = reflAlbedo * c_sunColor * diffuse * shadowValue;
//         } else {
//             // Hit Sky
//             reflectionColor = sampleSky(reflDir, frame.sunDirection);
//         }

//         // 4. Fresnel Effect
//         half NdotV = max(dot(hit.normal, -(half3)dir), 0.0h);
//         const half c_waterReflectivity = 0.02f; // Base reflectivity
//         half fresnel = c_waterReflectivity + (1.0h - c_waterReflectivity) * pow(1.0h - NdotV, 5.0h);

//         // 5. Mix Water Color and Reflection
//         half3 waterBaseColor = make_half3(0.0h, 0.1h, 0.3h);
//         color = lerp(waterBaseColor, reflectionColor, fresnel);
//     }
//     else if (hit.hit)
//     {
//         half3 baseColor = sampleTexture(hit.uv, hit.pos, textureAtlas);

//         // --- Direct Lighting (Sun) ---

//         half diffuse = max(dot(hit.normal, (half3)frame.sunDirection), 0.05h);
//         half3 directLight = baseColor * diffuse * c_sunColor * (half)shadowValue; 

//         // --- Global Illumination (Voxel Cone Tracing) ---
//         // float3 indirectLight = float3(0.0);
//         // float3 up = hit.normal;
//         // float3 right = normalize(cross(up, float3(0.577f, 0.577f, 0.577f))); // Jittered tangent
//         // float3 forward = cross(up, right);

//         // Trace 6 cones in a hemisphere around the normal
//         // indirectLight += traceCone(hit.pos, up, giData, csdf);
//         // indirectLight += traceCone(hit.pos, lerp(up, right, 0.5f), giData, csdf);
//         // indirectLight += traceCone(hit.pos, lerp(up, -right, 0.5f), giData, csdf);
//         // indirectLight += traceCone(hit.pos, lerp(up, forward, 0.5f), giData, csdf);
//         // indirectLight += traceCone(hit.pos, lerp(up, -forward, 0.5f), giData, csdf);
//         // indirectLight += traceCone(hit.pos, lerp(up, lerp(right, forward, 0.5f), 0.5f), giData, csdf);
        
//         // Average the cone results and apply albedo and an artistic strength factor
//         //indirectLight = (indirectLight / 6.0f) * baseColor * 0.6f;

//         // --- Final Color Composition ---
//         half3 ambient = sampleSky((float3)hit.normal, frame.sunDirection) * 0.05f * baseColor;
//         color = directLight + ambient; //+ indirectLight
//     }
//     else
//     {
//         color = sampleSky(dir, frame.sunDirection);
//     }

//     // --- 4. Apply Volumetric Fog ---
//     half dist = (half)(hit.hit ? length(hit.pos - camera.position) : 10000.0);
//     half fogAmount = clamp(1.0h - exp(-dist * 0.004h), 0.0h, 1.0h);
//     half3 fogColor = make_half3(0.95h, 0.95h, 1.0h);
    
//     return lerp(color, fogColor, fogAmount);
// }

