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
    texture3d<float, access::sample>  giData      [[texture(4)]], 
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

        hitInfo hit = trace(camera.position, dir, startDist, bitsTex, csdf);

        float depth = 100000.0f;
        half4 normal_shadow = half4(0.0h, 0.0h, 0.0h, 1.0h); // w=1.0 (Lit)

        if (hit.hit) {
            depth = length(hit.pos - camera.position);
            
            bool isShadowed = traceShadowAnyHit(hit.pos + (float3)(hit.normal * 1e-3h),
                                               frame.sunDirection, 2000.0f, bitsTex, csdf);
            
            normal_shadow = half4((half3)hit.normal, isShadowed ? 0.1h : 1.0h);
        }

        // Write Compact Data
        tileData[tid.y][tid.x].depth = depth;
        tileData[tid.y][tid.x].normal_shadow = normal_shadow;
    }
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
    
    half3 directLight = baseColor * c_sunColor * diffuse * shadowVal;
    half3 ambient = sampleSky((float3)normal, frame.sunDirection) * 0.1h * baseColor;


    //--- Global Illumination (Voxel Cone Tracing) ---
    half3 indirectLight = half3(0.0);
    half3 up = normal;
    half3 right = normalize(cross(up, half3(0.577h, 0.577h, 0.577h)));
    half3 forward = cross(up, right);

    //Trace 6 cones in a hemisphere around the normal
    indirectLight += traceCone(pos, up, giData, csdf);
    indirectLight += traceCone(pos, lerp(up, right, 0.5h), giData, csdf);
    indirectLight += traceCone(pos, lerp(up, -right, 0.5h), giData, csdf);
    indirectLight += traceCone(pos, lerp(up, forward, 0.5h), giData, csdf);
    indirectLight += traceCone(pos, lerp(up, -forward, 0.5h), giData, csdf);
    indirectLight += traceCone(pos, lerp(up, lerp(right, forward, 0.5h), 0.5h), giData, csdf);
    
    //Average the cone results and apply albedo and an artistic strength factor
    indirectLight = (indirectLight / 6.0h) * baseColor * 1.5h;

    half3 finalColor = directLight + ambient + indirectLight;

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

inline float2 intersectWorldAABB_Fast(float3 rayOrigin, float3 invRayDir) {
    float3 t0 = (float3(0.0f) - rayOrigin) * invRayDir;
    float3 t1 = (float3(SIZEX, SIZEY, SIZEZ) - rayOrigin) * invRayDir;
    float3 tMax = max(t0, t1);
    float3 tMin = min(t0, t1);
    return float2(max(max(tMin.x, tMin.y), tMin.z), min(min(tMax.x, tMax.y), tMax.z));
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