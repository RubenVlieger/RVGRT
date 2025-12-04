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

inline uint hash3_to_1(int3 p) {
    uint3 u = uint3(p);
    u = ((u >> 8U) ^ u.yzx) * 0x45D9F3BU;
    u = ((u >> 8U) ^ u.yzx) * 0x45D9F3BU;
    u = ((u >> 8U) ^ u.yzx) * 0x45D9F3BU;
    return u.x ^ u.y ^ u.z;
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

// Standard ACES fitted tone mapper (Unreal Engine 4 version)
// Compresses High Dynamic Range (e.g., 0 to 100) to LDR (0 to 1).
inline float3 ACESFilm(float3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

inline float3 LinearToSRGB(float3 color) {
    // Approx pow(x, 1.0/2.2)
    return select(1.055f * pow(color, 1.0f / 2.4f) - 0.055f,
                  12.92f * color,
                  color <= 0.0031308f);
}

// Schlick's approximation for Fresnel
inline float3 F_Schlick(float cosTheta, float3 F0) {
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

inline float3 applyContrast(float3 color, float contrast) {
    return max(float3(0.0f), (color - 0.5f) * contrast + 0.5f);
}

// Saturation boost (Luma-based)
inline float3 applySaturation(float3 color, float saturation) {
    // Standard Luma coefficients (Rec. 709)
    float luma = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    return mix(float3(luma), color, saturation);
}

// =================================================================================
// KERNEL 1: G-BUFFER & DIRECT LIGHTING (Physically Based Update)
// =================================================================================
kernel void GBufferAndDirectLight(
    // --- Outputs ---
    texture2d<float, access::write> texDirectLight [[texture(0)]], // RGB = Incoming Light Intensity (Irradiance)
    texture2d<float, access::write> texAlbedo      [[texture(1)]], // RGB = Surface Color (Material)
    texture2d<float, access::write> texNormal      [[texture(2)]], // RGB = Encoded Normal
    texture2d<float, access::write> texMotion      [[texture(3)]], // RG = Motion Vector
    texture2d<float, access::write> texDepth       [[texture(4)]], // R = Depth

    // --- Inputs ---
    constant const CameraData& camera [[buffer(0)]],
    constant const FrameData& frame   [[buffer(1)]],
    
    texture3d<uint, access::read>     bitsTex     [[texture(5)]],
    texture3d<float, access::sample>  csdf        [[texture(6)]],
    texture2d_array<float, access::sample>  textureAtlas[[texture(7)]],
    texture2d<float, access::sample>  halfDistTex [[texture(8)]],

    uint2 gid [[thread_position_in_grid]])
{
    // Bounds Check
    if (gid.x >= texDirectLight.get_width() || gid.y >= texDirectLight.get_height()) return;

    // 1. Ray Generation (Standard Pinhole with TAA Jitter)
    float2 pixelCenter = float2(gid) + 0.5f;
    float2 jitteredCoord = pixelCenter + camera.jitter; 
    
    float2 uv = jitteredCoord / float2(texDirectLight.get_width(), texDirectLight.get_height());
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    // 2. Accelerator Read (Distance Estimation from Pre-Pass)
    constexpr sampler sLinear(filter::linear);
    float startDist = halfDistTex.sample(sLinear, uv).r;
    
    // 3. Primary Ray Trace
    hitInfo hit = trace(camera.position, dir, startDist, bitsTex, csdf);

    // Initialize Outputs
    float depth = 100000.0f;
    half3 irradiance = half3(0.0h); // Incoming Light
    half3 albedo = half3(0.0h);     // Material Color
    half3 normal = half3(0.0h);

    if (hit.hit) 
    {
        depth = length(hit.pos - camera.position);
        normal = hit.normal;
        
        // --- Motion Vector Calculation ---
        float2 motionVector = float2(0.0f);
        if (depth < 50000.0f) {
            float4 currentClipPos = camera.unjitteredViewProjection * float4(hit.pos, 1.0f);
            float4 previousClipPos = camera.prevUnjitteredViewProjection * float4(hit.pos, 1.0f);
            if (previousClipPos.w > 0.0f && currentClipPos.w > 0.0f) {
                float2 prevNDC = previousClipPos.xy / previousClipPos.w;
                float2 currNDC = currentClipPos.xy / currentClipPos.w;
                motionVector = 0.5f * (currNDC - prevNDC);
                motionVector.y = -motionVector.y; // Flip Y for texture coords
            }
        }
        texMotion.write(float4(motionVector.x, motionVector.y, 0, 0), gid);

        // --- Material Logic ---
        bool isWater = (hit.pos.y < 31.001f && normal.y > 0.8h);
        if (isWater) 
        {
            // === WATER FIX ===
            
            // 1. Dark Albedo
            // Water is dark. This allows the reflection to sit "on top".
            // We use a slight blue tint for the "deep water" color.
            albedo = half3(0.04h, 0.1h, 0.25h); 

            // 2. Waves
            float nx_wave = fbm3D(hit.pos.x, hit.pos.z, frame.time, 3, 0.06f, 2.0f, 0.6f);
            float ny_wave = fbm3D(hit.pos.z, hit.pos.x, frame.time + 112.0f, 3, 0.06f, 2.0f, 0.6f);
            half3 distortedNormal = normalize(normal + half3(half(nx_wave) * 0.1h, 0.0h, half(ny_wave) * 0.1h));
            
            // 3. Reflection
            float3 reflDir = reflect(dir, (float3)distortedNormal);
            hitInfo reflHit = trace(hit.pos, reflDir, 0.05f, bitsTex, csdf);
            
            half3 reflectColor;
            if (reflHit.hit) {
                // Hitting geometry
                float3 camToWater = hit.pos - camera.position;
                float3 waterToRefl = reflHit.pos - hit.pos;
                float distSq = dot(camToWater, camToWater) + dot(waterToRefl, waterToRefl);
                half3 rAlbedo = sampleTexture(reflHit.uv, reflHit.pos, textureAtlas, distSq);
                
                // Shadow check for the reflected object
                bool rShadow = traceShadowAnyHitSlow(reflHit.pos + (float3)reflHit.normal * 0.01f, frame.sunDirection, 1000.0f, bitsTex, csdf);
                
                half3 litVal = c_sunColor;
                half3 shadowVal = half3(0.05h); // Neutral dark grey ambient

                reflectColor = rAlbedo * (rShadow ? shadowVal : litVal);
            } else {
                // Hitting Sky
                reflectColor = sampleSky(reflDir, frame.sunDirection);
            }
            
            // 4. Specular (Sun Highlight on water)
            float3 viewDir = -dir;
            float3 halfVec = normalize(viewDir + frame.sunDirection);
            float NdotH = max(dot((float3)distortedNormal, halfVec), 0.0f);
            half specular = pow(NdotH, 512.0f) * 4.0f; 

            // 5. Fresnel
            half NdotV = max(dot(distortedNormal, (half3)viewDir), 0.0h);
            // F0=0.02 (Water). at 90 degrees (grazing), it becomes 1.0 reflection.
            half fresnel = 0.02h + (0.98h) * pow(1.0h - NdotV, 5.0h);
            
            bool waterShadow = traceShadowAnyHitSlow(hit.pos, frame.sunDirection, 1000.0f, bitsTex, csdf);
            half shadowVal = waterShadow ? 0.0h : 1.0h;

            // 6. Combine
            // Total light coming from surface = (Reflection * Fresnel) + (SunSpec * Shadow)
            half3 totalReflection = (reflectColor * fresnel) + (c_sunColor * specular * shadowVal);

            // 7. Store using the Math Hack
            
            // Composite Logic: Final = (StoredDirect + Indirect) * Albedo
            // StoredDirect = TotalReflection / Albedo
            // We add a tiny epsilon to albedo to avoid divide-by-zero
            irradiance = totalReflection / (albedo + 0.001h);
        } else 
        {
            // === SOLID BLOCK SHADING ===
            
            // 1. Texture Sampling
            half2 localUV = reconstructUV(hit.pos, normal);
            // PBR Rule: This is purely color. Do NOT multiply by sun here.
            albedo = sampleTexture(localUV, hit.pos, textureAtlas, depth * depth);
            
            // 2. Shadow Trace
            // Offset start pos slightly to avoid acne
            bool isShadowed = traceShadowAnyHitSlow(hit.pos + (float3)normal * 0.005f, frame.sunDirection, 2000.0f, bitsTex, csdf);
            half shadowFactor = isShadowed ? 0.02h : 1.0h;
            
            // 3. Lambertian Diffuse Lighting
            // Intensity = LightColor * dot(N, L) * Shadow
            half NdotL = max(dot(normal, (half3)frame.sunDirection), 0.0h);
            
            irradiance = c_sunColor * NdotL * shadowFactor;
        }
    } 
    else 
    {
        // === SKYBOX LOGIC ===
        
        // 1. Sky Motion Vectors
        float3 fakePos = camera.position + dir * 1000.0f;
        float4 currentClipPos = camera.unjitteredViewProjection * float4(fakePos, 1.0f);
        float4 previousClipPos = camera.prevUnjitteredViewProjection * float4(fakePos, 1.0f);
        
        float2 mv = float2(0.0);
        if (previousClipPos.w > 0.0f && currentClipPos.w > 0.0f) {
            float2 prevNDC = previousClipPos.xy / previousClipPos.w;
            float2 currNDC = currentClipPos.xy / currentClipPos.w;
            mv = 0.5f * (currNDC - prevNDC);
            mv.y = -mv.y;
        }
        texMotion.write(float4(mv.x, mv.y, 0, 0), gid);

        // 2. Sky Lighting
        // For the sky, Light * Albedo must equal SkyColor.
        irradiance = sampleSky(dir, frame.sunDirection);
        albedo = half3(1.0h); 
    }

    // --- Write Outputs ---
    texDirectLight.write(float4((float3)irradiance, 1.0f), gid);
    texAlbedo.write(float4((float3)albedo, 1.0f), gid);
    texNormal.write(float4(((float3)normal), 1.0f), gid); 
    texDepth.write(float4(depth), gid);
}


// =================================================================================
// KERNEL 2: INDIRECT BOUNCE (Physically Based - 1 Bounce)
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
    texture2d_array<float, access::sample> textureAtlas[[texture(5)]],
    
    uint2 gid [[thread_position_in_grid]])
{
    // 1. Bounds Check
    if (gid.x >= texRawIndirect.get_width() || gid.y >= texRawIndirect.get_height()) return;
    
    // 2. Read G-Buffer
    float depth = texDepth.read(gid).r;
    
    // Sky Optimization: If depth is infinite (sky), there is no surface to receive indirect light.
    if (depth > 50000.0f) {
        texRawIndirect.write(float4(0,0,0,0), gid);
        return;
    }

    half3 normal = (half3)texNormal.read(gid).rgb;
    
    // 3. Reconstruct World Position
    float2 uv = (float2(gid) + 0.5f) / float2(texRawIndirect.get_width(), texRawIndirect.get_height());
    float3 pos = reconstructPos(depth, uv, camera);

    // 4. Initialize Random Number Generator (PCG Hash)
    // We use position + time to get a stable but animated noise pattern
    uint voxelHash = hash3_to_1(int3(pos * 1024.f));
    uint seed = voxelHash + uint(frame.time * 123456.0f); // Time dependent for accumulation
    
    // 5. Create Orthonormal Basis (Tangent Space)
    float3 N = (float3)normal;
    // Duff's method or simple helper to find perpendicular vector
    float3 helper = abs(N.x) > 0.99f ? float3(0,0,1) : float3(1,0,0);
    float3 Tangent = normalize(cross(N, helper));
    float3 Bitangent = cross(N, Tangent);

    // 6. Cosine-Weighted Hemisphere Sampling
    // PBR requirement: Diffuse surfaces reflect light in a cosine-weighted lobe.
    float r1 = rand_float(seed);
    float r2 = rand_float(seed);
    
    // Map square random numbers to hemisphere
    float phi = 2.0f * 3.14159f * r1;
    float cosTheta = sqrt(1.0f - r2);
    float sinTheta = sqrt(r2); 
    
    float3 localDir = float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));

    // Transform to World Space
    // Note: localDir.y corresponds to the Up vector (Normal)
    float3 rayDir = localDir.x * Tangent + localDir.y * N + localDir.z * Bitangent;
    rayDir = normalize(rayDir);

    // 7. Trace the Bounce Ray
    hitInfo hit = trace(pos , rayDir, 0.05f, bitsTex, csdf);
    
    half3 incomingLight = half3(0.0h);

    if (hit.hit) {
        // --- NEXT EVENT ESTIMATION (Lighting at the hit point) ---
        
        // A. Shadow Check (Is the bounced surface lit by the sun?)
        bool isShadowed = traceShadowAnyHitFast(hit.pos + (float3)hit.normal * 0.01f, frame.sunDirection, 1000.0f, bitsTex, csdf);
        
        // B. Get Material of the bounced surface
        half2 hitUV = reconstructUV(hit.pos, hit.normal);
        float3 bounceVec = hit.pos - pos;
        float totalDistSq = (depth * depth) + dot(bounceVec, bounceVec); 
        
        half3 bouncedAlbedo = sampleTexture(hitUV, hit.pos, textureAtlas, totalDistSq);
        
        // C. Calculate Radiance
        half NdotL = max(dot(hit.normal, (half3)frame.sunDirection), 0.0h);
        
        // This restores color bleeding (e.g. Gold reflecting yellow light)
        half3 directLightAtHit = c_sunColor * NdotL * (isShadowed ? 0.0h : 1.0h); 
        
        // Add a tiny bit of bounce ambient to prevent pitch black corners, 
        // but keep it the color of the material
        half3 bounceAmbient = bouncedAlbedo * 0.05h; 

        incomingLight = (directLightAtHit * bouncedAlbedo) + bounceAmbient;

    } else {
        // We hit the sky
        half3 skyLight = sampleSky(rayDir, frame.sunDirection);
        float luma = dot((float3)skyLight, float3(0.3f, 0.59f, 0.11f));
        half3 desaturatedSky = mix(skyLight, half3(luma), 0.6h); 
        
        incomingLight = desaturatedSky * 0.25h; 
    }
    
    // Note on Division by PI / Cosine Term:
    // Since we used Cosine-Weighted Sampling for the ray direction, the PDF (Probability Density Function)
    // cancels out the cosine term in the rendering equation. We usually divide by PI, but often
    // sun intensity is calibrated without it. For now, this is statistically unbiased.
    
    texRawIndirect.write(float4((float3)incomingLight, 1.0f), gid);
}

// =================================================================================
// HELPER: RGB <-> YCoCg Conversions
// =================================================================================
inline float3 RGBToYCoCg(float3 rgb) {
    float Y  = dot(rgb, float3(0.25f, 0.50f, 0.25f));
    float Co = dot(rgb, float3(0.50f, 0.00f, -0.50f));
    float Cg = dot(rgb, float3(-0.25f, 0.50f, -0.25f));
    return float3(Y, Co, Cg);
}

inline float3 YCoCgToRGB(float3 ycocg) {
    float Y  = ycocg.x;
    float Co = ycocg.y;
    float Cg = ycocg.z;
    return float3(Y + Co - Cg, Y + Cg, Y - Co - Cg);
}

// =================================================================================
// KERNEL 3: ADVANCED TEMPORAL ACCUMULATION
// =================================================================================
kernel void TemporalAccumulation(
    texture2d<float, access::write> texAccum      [[texture(0)]],
    texture2d<float, access::read>  texRawIndirect[[texture(1)]],
    texture2d<float, access::sample> texHistory   [[texture(2)]], 
    texture2d<float, access::read>  texMotion     [[texture(3)]],
    texture2d<float, access::read>  texDepth      [[texture(4)]],
    texture2d<float, access::read>  texPrevDepth  [[texture(5)]], 
    texture2d<float, access::read>  texDirect     [[texture(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texAccum.get_width() || gid.y >= texAccum.get_height()) return;

    // 1. Read Current Frame Color (Direct + Indirect)
    float3 currentRGB = texDirect.read(gid).rgb + texRawIndirect.read(gid).rgb;
    
    // 2. Motion and UVs
    float2 motion = texMotion.read(gid).xy;
    float velMag = length(motion);
    float movementFactor = saturate(velMag * 200.0f); 
    

    float2 uv = (float2(gid) + 0.5f) / float2(texAccum.get_width(), texAccum.get_height());
    float2 prevUV = uv - motion;

    // 3. Neighborhood Statistics (Variance Calculation)
    float3 m1 = float3(0.0f); // First moment (Mean)
    float3 m2 = float3(0.0f); // Second moment (Variance)
    
    // We sample a 3x3 neighborhood
    for(int y = -1; y <= 1; ++y) {
        for(int x = -1; x <= 1; ++x) {
            uint2 tapCoord = uint2(gid.x + x, gid.y + y);
            
            // Boundary checks (clamp to edge)
            tapCoord.x = clamp(tapCoord.x, 0u, texAccum.get_width() - 1);
            tapCoord.y = clamp(tapCoord.y, 0u, texAccum.get_height() - 1);

            float3 neighborRGB = texDirect.read(tapCoord).rgb + texRawIndirect.read(tapCoord).rgb;
            float3 neighborYCoCg = RGBToYCoCg(neighborRGB);

            m1 += neighborYCoCg;
            m2 += neighborYCoCg * neighborYCoCg;
        }
    }

    float3 mu = m1 / 9.0f;
    float3 sigma = sqrt(abs(m2 / 9.0f - mu * mu));


    float gamma = mix(10.0f, 0.75f, movementFactor); 
    float3 minColor = mu - gamma * sigma;
    float3 maxColor = mu + gamma * sigma;

    // 5. Sample History
    constexpr sampler sLinear(filter::linear);
    float3 historyRGB = texHistory.sample(sLinear, prevUV).rgb;
    float3 historyYCoCg = RGBToYCoCg(historyRGB);

    // 6. CLIP History to Box
    // Instead of hard clamp, we clip the vector towards the center (better color stability)
    // But for performance/simplicity, hard clamping in YCoCg is usually sufficient.
    float3 clampedHistoryYCoCg = clamp(historyYCoCg, minColor, maxColor);
    float3 clampedHistoryRGB = YCoCgToRGB(clampedHistoryYCoCg);


    float blendWeight = mix(0.98f, 0.9f, movementFactor);
    
    // 8. Depth Rejection (Disocclusion Check)
    bool validHistory = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f);
    if (validHistory) {
        uint2 prevCoords = uint2(prevUV.x * texPrevDepth.get_width(), prevUV.y * texPrevDepth.get_height());
        float currentDepth = texDepth.read(gid).r;
        float prevDepth = texPrevDepth.read(prevCoords).r;
        
        // Use relative difference
        float diff = abs(currentDepth - prevDepth) / (currentDepth + 1e-5f);
        if (diff > 0.05f) { // Stricter threshold
            blendWeight = 0.0f; // Reset
        }
    } else {
        blendWeight = 0.0f;
    }

    // 8. Blend
    float3 result = mix(currentRGB, clampedHistoryRGB, blendWeight);
    texAccum.write(float4(result, 1.0f), gid);
}

// =================================================================================
// KERNEL 4: A-TROUS EDGE-AVOIDING FILTER
// =================================================================================
kernel void BilateralDenoise(
    texture2d<float, access::write> texDenoised [[texture(0)]],
    texture2d<float, access::read>  texAccum    [[texture(1)]],
    texture2d<float, access::read>  texNormal   [[texture(2)]],
    texture2d<float, access::read>  texDepth    [[texture(3)]],
    constant int& step_width        [[buffer(0)]], 
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texDenoised.get_width() || gid.y >= texDenoised.get_height()) return;

    // 1. Center Tap Data
    float3 centerC = texAccum.read(gid).rgb;
    float3 centerN = texNormal.read(gid).rgb;
    float centerD  = texDepth.read(gid).r;

    // Gaussian-approximate weights for 3x3
    const float kernelWeights[3] = { 1.0f, 2.0f / 1.0f, 4.0f / 1.0f };

    float3 sumColor = float3(0.0f);
    float sumWeight = 0.0f;

    // 3. Iteration (3x3 grid with holes)
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            
            // Offset coordinate by step_width
            int2 offset = int2(x, y) * step_width;
            uint2 tapCoord = uint2(gid.x + offset.x, gid.y + offset.y);

            // Bounds check
            if(tapCoord.x >= texDenoised.get_width() || tapCoord.y >= texDenoised.get_height()) {
                tapCoord = gid;
            }

            float3 tapC = texAccum.read(tapCoord).rgb;
            float3 tapN = texNormal.read(tapCoord).rgb;
            float tapD  = texDepth.read(tapCoord).r;

            // --- A. Normal Weight
            float dotN = max(dot(centerN, tapN), 0.0f);
            float wNormal = pow(dotN, 16.0f); // High power ensures we don't bleed colors around voxel corners

            // --- B. Depth Weight (Plane Distance) ---
            // 1.0 = Allow 1 unit (1 block) of depth deviation before rejecting
            float wDepth = (abs(centerD - tapD) < 1.5f) ? 1.0f : 0.0f;
            

            // Calculate Kernel Weight (Gaussian)
            float kWeight = kernelWeights[abs(x)] * kernelWeights[abs(y)];

            // Combine
            float w = wNormal * wDepth * kWeight;

            sumColor  += tapC * w;
            sumWeight += w;
        }
    }

    if (sumWeight < 1e-4f) {
        sumColor = centerC;
        sumWeight = 1.0f;
    }

    texDenoised.write(float4(sumColor / sumWeight, 1.0f), gid);
}

// =================================================================================
// KERNEL 5: COMPOSITE (Color Grading & Fog Fix)
// =================================================================================
kernel void Composite(
    texture2d<float, access::write> texFinal   [[texture(0)]],
    texture2d<float, access::read>  texDirect  [[texture(1)]],
    texture2d<float, access::read>  texAccum   [[texture(2)]], 
    texture2d<float, access::read>  texAlbedo  [[texture(3)]],
    texture2d<float, access::read>  texDepth   [[texture(4)]],

    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= texFinal.get_width() || gid.y >= texFinal.get_height()) return;

    // 1. Gather Data
    float3 directLight   = texDirect.read(gid).rgb;
    float3 indirectLight = texAccum.read(gid).rgb;
    float3 albedo        = texAlbedo.read(gid).rgb;
    float depth          = texDepth.read(gid).r;

    // 3. Apply Material (Linear Space)
    float3 totalIrradiance = directLight + indirectLight;
    float3 linearColor = totalIrradiance * albedo;

    // 4. BETTER FOG LOGIC
    if (depth < 50000.0f) 
    {
        const float fogStart = 60.0f;  // Fog starts 60 blocks away (keeps foreground clear)
        const float fogDensity = 0.0002f; 
        
        // Calculate factor
        float dist = max(depth - fogStart, 0.0f);
        float fogFactor = 1.0f - exp(-dist * fogDensity);

        float3 fogColor = float3(0.5f, 0.7f, 0.9f); 
        
        linearColor = mix(linearColor, fogColor, fogFactor);
    }

    // 5. COLOR GRADING 
    
    // A. Exposure Compensation (Brighten the image up)
    linearColor *= 0.8f; 

    // B. Saturation Boost 
    linearColor = applySaturation(linearColor, (depth > 50000.0f) ? 1.05f : 1.4f); 

    // C. Contrast S-Curve (make darks darker, brights brighter)
    //linearColor = applyContrast(linearColor, 1.03f);

    // 6. Tone Mapping (ACES)
    float3 toneMapped = ACESFilm(linearColor);

    // 7. Gamma Correction (Linear -> sRGB)
    float3 finalColor = LinearToSRGB(toneMapped);

    texFinal.write(float4(finalColor, 1.0f), gid);
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