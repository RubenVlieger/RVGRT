#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
using namespace metal;


struct GBufferData {
    float4 position; // xyz = world pos, w = hit (0 or 1)
    float4 normal;   // xyz = normal, w = unused
};

kernel void gbuffer_kernel(
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

kernel void shadow_kernel(
    texture2d<float, access::write> shadowMask [[texture(0)]],
    texture2d<float, access::read> gPos [[texture(1)]],
    texture3d<uint, access::read> bits [[texture(2)]],
    texture3d<float, access::sample> csdf [[texture(3)]],
    constant const FrameData& frame [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= shadowMask.get_width() || gid.y >= shadowMask.get_height()) return;

    float4 posData = gPos.read(gid);
    if (posData.w < 0.5f) {
        shadowMask.write(float4(1.0), gid); // Sky is lit
        return;
    }

    bool blocked = traceShadowAnyHit(posData.xyz + frame.sunDirection * 0.1f, 
                                     frame.sunDirection, 2048.0f, bits, csdf);
    
    shadowMask.write(float4(blocked ? 0.1 : 1.0), gid);
}


kernel void reflection_kernel(
    // OUTPUT: RGB = Reflected Color, A = Fresnel Intensity
    texture2d<float, access::write> reflectionOutput [[texture(0)]],
    
    // INPUT G-BUFFER
    texture2d<float, access::read> gPos [[texture(1)]],
    texture2d<float, access::read> gNorm [[texture(2)]],
    
    // SCENE DATA
    texture3d<uint, access::read> bitsTex [[texture(3)]],
    texture3d<float, access::sample> csdf [[texture(4)]], 
    texture2d<float, access::sample> textureAtlas [[texture(5)]],
    
    // CONSTANTS
    constant const CameraData& camera [[buffer(0)]],
    constant const FrameData& frame [[buffer(1)]],
    
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= reflectionOutput.get_width() || gid.y >= reflectionOutput.get_height()) return;

    float4 posData = gPos.read(gid);
    half3 normal = (half3)gNorm.read(gid).xyz;
    
    if (posData.w < 0.5f || posData.y >= 31.001f || normal.y < 0.8h) {
        reflectionOutput.write(float4(0.0f), gid);
        return;
    }

    float3 pos = posData.xyz;
    float nx_wave = fbm3D(pos.x, pos.z, frame.time, 3, 0.06f, 2.0f, 0.6f);
    float ny_wave = fbm3D(pos.z, pos.x, frame.time + 112.0f, 3, 0.06f, 2.0f, 0.6f);
    
    half3 distortedNormal = normalize(normal + half3(half(nx_wave) * 0.1h, 0.0h, half(ny_wave) * 0.1h));

    float2 uv = (float2(gid) + 0.5f) / float2(reflectionOutput.get_width(), reflectionOutput.get_height());
    float2 ndc = uv * 2.0f - 1.0f;
    float3 viewDir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    float3 reflDir = reflect(viewDir, (float3)distortedNormal);
    float3 reflOrigin = pos + (float3)normal * 0.01f;

    hitInfo reflHit = trace(reflOrigin, reflDir, 1000.0f, bitsTex, csdf);

    half3 reflectionColor;
    if (reflHit.hit) {
        half3 reflAlbedo = sampleTexture(reflHit.uv, reflHit.pos, textureAtlas);
        
        half reflDiffuse = max(dot(reflHit.normal, (half3)frame.sunDirection), 0.2h);        
        reflectionColor = reflAlbedo * c_sunColor * reflDiffuse;
    } else {
        reflectionColor = sampleSky(reflDir, frame.sunDirection);
    }

    half NdotV = max(dot(distortedNormal, -(half3)viewDir), 0.0h);
    const half c_waterReflectivity = 0.02h;
    half fresnel = c_waterReflectivity + (1.0h - c_waterReflectivity) * pow(1.0h - NdotV, 5.0h);

    reflectionOutput.write(float4((float3)reflectionColor, (float)fresnel), gid);
}


kernel void distApproximationKernel(
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

    hitInfo hit = trace(camera.position, dir, 0.0f, bitsTex, csdf);
    float dist = hit.hit ? length(hit.pos - camera.position) : 1000000.0f;

    distTex.write(float4(dist, 0, 0, 1), gid);
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



// kernel void raytrace_kernel(
//     texture2d<float, access::write> outputTexture [[texture(0)]],
//     constant const CameraData& camera [[buffer(0)]],
//     constant const FrameData& frame [[buffer(1)]],
//     texture3d<uint, access::read> bitsTex [[texture(2)]],
//     texture3d<float, access::sample> csdf [[texture(3)]], 
//     texture3d<uint, access::read> giData [[texture(4)]],
//     texture2d<float, access::sample> textureAtlas [[texture(5)]],
//     texture2d<float, access::sample> halfDistTex [[texture(6)]],
//     texture2d<float, access::sample> halfShadowTex [[texture(7)]],
//     uint2 gid [[thread_position_in_grid]])
// {
//     uint width = outputTexture.get_width();
//     uint height = outputTexture.get_height();
//     if (gid.x >= width || gid.y >= height) {
//         return;
//     }
//     float2 uv = (float2)gid / float2(width, height);


//     constexpr sampler sLinear(filter::linear);
//     float dist = minDist(halfDistTex, uv);
//     float shadow = halfShadowTex.sample(sLinear, uv).r;

    
//     // Call the main color computation function
//     half3 final_color = computeColor(uv, dist, shadow, camera, frame, bitsTex, csdf, giData, textureAtlas);

//     // Clamp final color to prevent illegal values and write to the output texture
//     outputTexture.write(float4(saturate((float3)final_color), 1.0f), gid);
// }
