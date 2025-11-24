#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
using namespace metal;

float minDist(texture2d<float, access::sample> tex, float2 uv)
{
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest);
    
    float width = float(tex.get_width());
    float height = float(tex.get_height());
    
    float half_pixel_x = 1.0f / width;
    float half_pixel_y = 1.0f / height;

    float dist1 = tex.sample(s, uv).r;
    float dist2 = tex.sample(s, uv + float2(half_pixel_x, 0.0f)).r;
    float dist3 = tex.sample(s, uv + float2(0.0f, half_pixel_y)).r;
    float dist4 = tex.sample(s, uv + float2(half_pixel_x, half_pixel_y)).r;

    return min(min(dist1, dist2), min(dist3, dist4));
}


kernel void distApproximationKernel(
    texture2d<float, access::write> distTex [[texture(0)]],
    texture2d<float, access::write> shadowTex [[texture(1)]],
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

    // 1. Base coarse estimate
    float coarseDist = traceDistanceOnly(camera.position, dir, 2048.0f, csdf);
    
    float lodStart = 48.0f; 
    float lodEnd   = 96.0f; 
    
    bool needsHighQuality = (coarseDist < lodEnd);

    float finalDist = coarseDist;
    float shadowValue = 1.0f;

    if (simd_any(needsHighQuality)) 
    {        
        float lqShadow = coarseDist < 2048 ? traceShadowCSDF(camera.position + dir * coarseDist, frame.sunDirection, 2048.f, csdf) : 1.0f;
        
        hitInfo hit = trace(camera.position, dir, max(0.0f, coarseDist - 4.0f), bitsTex, csdf);
        
        if (hit.hit) {
            float hqDist = dot(hit.pos - camera.position, dir);            
            hitInfo shadowHit = trace(hit.pos, frame.sunDirection, 0.001f, bitsTex, csdf);
            float hqShadow = shadowHit.hit ? 0.1f : 1.0f;
            float t = saturate((hqDist - lodStart) / (lodEnd - lodStart));
            
            finalDist = mix(hqDist, coarseDist, t);
            shadowValue = mix(hqShadow, lqShadow, t);
        } 
        else {
            // DDA missed (maybe a hole in the voxel mesh that CSDF didn't see, or sky)
            finalDist = 2048.f;
            shadowValue = lqShadow;
        }
    }
    
    else 
    {
        // --- SIMD GROUP IS FAR (Optimization) ---
        shadowValue = coarseDist < 400.0f ? traceShadowCSDF(camera.position + dir * coarseDist, frame.sunDirection, 2048.f, csdf) : shadowValue;
    }

    distTex.write(float4(max(0.0f, finalDist - 2.0f), 0, 0, 1), gid);
    shadowTex.write(float4(shadowValue, 0, 0, 1), gid);
}

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




half3 computeColor(
    float2 uv,
    float startDist,
    float shadowValue,
    constant const CameraData& camera,
    constant const FrameData& frame,
    texture3d<uint, access::read> bitsTex,
    texture3d<float, access::sample> csdf,
    texture3d<uint, access::read> giData,
    texture2d<float, access::sample> textureAtlas)
{
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    hitInfo hit = trace(camera.position, dir, startDist, bitsTex, csdf);

    half3 color = make_half3(0.0);

    // logic for reflection code
    if (hit.hit && hit.pos.y < 31.001f) 
    {
        half nx_wave = fbm3D_h(hit.pos.x, hit.pos.z, frame.time, 3, 0.06f, 2.0f, 0.6f);
        half ny_wave = fbm3D_h(hit.pos.z, hit.pos.x, frame.time + 112.0f, 3, 0.06f, 2.0f, 0.6f);

        half3 distortedNormal = normalize(hit.normal + half3(nx_wave * 0.1h, ny_wave * 0.1h, 0.h));

        float3 reflDir = reflect(dir, (float3)distortedNormal);
        float3 reflOrigin = hit.pos + (float3)hit.normal * 0.005f; 

        hitInfo reflHit = trace(reflOrigin, reflDir, 0.0f, bitsTex, csdf);

        half3 reflectionColor;
        if (reflHit.hit) {
            // A. Sample Texture
            half3 reflAlbedo = (half3)sampleTexture(reflHit.uv, reflHit.pos, textureAtlas);
            

            half diffuse = max(dot(reflHit.normal, (half3)frame.sunDirection), 0.0h);
            
            reflectionColor = reflAlbedo * c_sunColor * diffuse * shadowValue;
        } else {
            // Hit Sky
            reflectionColor = sampleSky(reflDir, frame.sunDirection);
        }

        // 4. Fresnel Effect
        half NdotV = max(dot(hit.normal, -(half3)dir), 0.0h);
        const half c_waterReflectivity = 0.02f; // Base reflectivity
        half fresnel = c_waterReflectivity + (1.0h - c_waterReflectivity) * pow(1.0h - NdotV, 5.0h);

        // 5. Mix Water Color and Reflection
        half3 waterBaseColor = make_half3(0.0h, 0.1h, 0.3h);
        color = lerp(waterBaseColor, reflectionColor, fresnel);
    }
    else if (hit.hit)
    {
        half3 baseColor = sampleTexture(hit.uv, hit.pos, textureAtlas);

        // --- Direct Lighting (Sun) ---

        half diffuse = max(dot(hit.normal, (half3)frame.sunDirection), 0.05h);
        half3 directLight = baseColor * diffuse * c_sunColor * (half)shadowValue; 

        // --- Global Illumination (Voxel Cone Tracing) ---
        // float3 indirectLight = float3(0.0);
        // float3 up = hit.normal;
        // float3 right = normalize(cross(up, float3(0.577f, 0.577f, 0.577f))); // Jittered tangent
        // float3 forward = cross(up, right);

        // Trace 6 cones in a hemisphere around the normal
        // indirectLight += traceCone(hit.pos, up, giData, csdf);
        // indirectLight += traceCone(hit.pos, lerp(up, right, 0.5f), giData, csdf);
        // indirectLight += traceCone(hit.pos, lerp(up, -right, 0.5f), giData, csdf);
        // indirectLight += traceCone(hit.pos, lerp(up, forward, 0.5f), giData, csdf);
        // indirectLight += traceCone(hit.pos, lerp(up, -forward, 0.5f), giData, csdf);
        // indirectLight += traceCone(hit.pos, lerp(up, lerp(right, forward, 0.5f), 0.5f), giData, csdf);
        
        // Average the cone results and apply albedo and an artistic strength factor
        //indirectLight = (indirectLight / 6.0f) * baseColor * 0.6f;

        // --- Final Color Composition ---
        half3 ambient = sampleSky((float3)hit.normal, frame.sunDirection) * 0.05f * baseColor;
        color = directLight + ambient; //+ indirectLight
    }
    else
    {
        color = sampleSky(dir, frame.sunDirection);
    }

    // --- 4. Apply Volumetric Fog ---
    half dist = (half)(hit.hit ? length(hit.pos - camera.position) : 10000.0);
    half fogAmount = clamp(1.0h - exp(-dist * 0.004h), 0.0h, 1.0h);
    half3 fogColor = make_half3(0.95h, 0.95h, 1.0h);
    
    return lerp(color, fogColor, fogAmount);
}



kernel void raytrace_kernel(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    constant const CameraData& camera [[buffer(0)]],
    constant const FrameData& frame [[buffer(1)]],
    texture3d<uint, access::read> bitsTex [[texture(2)]],
    texture3d<float, access::sample> csdf [[texture(3)]], 
    texture3d<uint, access::read> giData [[texture(4)]],
    texture2d<float, access::sample> textureAtlas [[texture(5)]],
    texture2d<float, access::sample> halfDistTex [[texture(6)]],
    texture2d<float, access::sample> halfShadowTex [[texture(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = outputTexture.get_width();
    uint height = outputTexture.get_height();
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    float2 uv = (float2)gid / float2(width, height);


    constexpr sampler sLinear(filter::linear);
    float dist = minDist(halfDistTex, uv);
    float shadow = halfShadowTex.sample(sLinear, uv).r;

    
    // Call the main color computation function
    half3 final_color = computeColor(uv, dist, shadow, camera, frame, bitsTex, csdf, giData, textureAtlas);

    // Clamp final color to prevent illegal values and write to the output texture
    outputTexture.write(float4(saturate((float3)final_color), 1.0f), gid);
}
