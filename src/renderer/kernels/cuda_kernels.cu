#ifdef _WIN32
#include "cuda_kernels.cuh"
#include "cumath.h"
#include "renderer/intersections.h"
#include "TerrainGeneration.h"
#include "raytracing_functions.h"

__constant__ CameraData c_camera;
__constant__ FrameData c_frame;
__constant__ CharacterGPUData c_charData;

void update_constant_memory(const CameraData& camData, const FrameData& frameData, const void* charData, size_t charSize) {
    cudaMemcpyToSymbol(c_camera, &camData, sizeof(CameraData));
    cudaMemcpyToSymbol(c_frame, &frameData, sizeof(FrameData));
    if (charData) cudaMemcpyToSymbol(c_charData, charData, charSize);
}

// ---------------------------------------------------------------------------------------------------------
// Helper: float4 to uchar4
__device__ __forceinline__ uchar4 float4_to_uchar4(float4 v) {
    return make_uchar4(
        (unsigned char)(fminf(fmaxf(v.x, 0.0f), 1.0f) * 255.0f),
        (unsigned char)(fminf(fmaxf(v.y, 0.0f), 1.0f) * 255.0f),
        (unsigned char)(fminf(fmaxf(v.z, 0.0f), 1.0f) * 255.0f),
        (unsigned char)(fminf(fmaxf(v.w, 0.0f), 1.0f) * 255.0f)
    );
}

__device__ __forceinline__ char4 float4_to_char4(float4 v) {
    return make_char4(
        (char)(fminf(fmaxf(v.x, -1.0f), 1.0f) * 127.0f),
        (char)(fminf(fmaxf(v.y, -1.0f), 1.0f) * 127.0f),
        (char)(fminf(fmaxf(v.z, -1.0f), 1.0f) * 127.0f),
        (char)(fminf(fmaxf(v.w, -1.0f), 1.0f) * 127.0f)
    );
}

__device__ __forceinline__ float4 uchar4_to_float4(uchar4 v) {
    return make_float4(v.x / 255.0f, v.y / 255.0f, v.z / 255.0f, v.w / 255.0f);
}

__device__ __forceinline__ float4 char4_to_float4(char4 v) {
    return make_float4(v.x / 127.0f, v.y / 127.0f, v.z / 127.0f, v.w / 127.0f);
}

// ---------------------------------------------------------------------------------------------------------
// Pass 0: distApproximationKernel
__global__ void distApproximationKernel(uint32_t width, uint32_t height, cudaSurfaceObject_t distTex, const int3 worldOrigin, const uint32_t* indirection, const void* sectors, const uint64_t* occupancy, const uint8_t* data, const uint64_t* sectorMasks) {
    uint2 gid = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (gid.x >= width || gid.y >= height) return;

    float2 uv = (make_float2(gid.x, gid.y) + 0.5f) / make_float2(width, height);
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(c_camera.forward + ndc.x * c_camera.right + ndc.y * c_camera.up);

    hitInfo hit = trace(c_camera.position, dir, indirection, (SectorInfo*)sectors, (uint64_t*)occupancy, (uint8_t*)data, (uint64_t*)sectorMasks, worldOrigin, &c_charData);

    float dist = hit.hit ? length(hit.pos - c_camera.position) : 5000.0f;
    dist = fmaxf(0.0f, dist - 8.0f); // Safety padding

    surf2Dwrite(dist, distTex, gid.x * sizeof(float), gid.y);
}

void launch_distApproximationKernel(cudaStream_t stream, uint32_t width, uint32_t height, cudaSurfaceObject_t halfDistOut, const int3& worldOrigin, const uint32_t* indirection, const void* sectors, const uint64_t* occupancy, const uint8_t* data, const uint64_t* sectorMasks) {
    dim3 threads(8, 8);
    dim3 blocks((width + 7) / 8, (height + 7) / 8);
    distApproximationKernel<<<blocks, threads, 0, stream>>>(width, height, halfDistOut, worldOrigin, indirection, sectors, occupancy, data, sectorMasks);
}

// ---------------------------------------------------------------------------------------------------------
// Pass 1: GBufferAndDirectLight
__global__ void GBufferAndDirectLightKernel(uint32_t width, uint32_t height, cudaTextureObject_t halfDistTex, cudaSurfaceObject_t texDirectLight, cudaSurfaceObject_t texAlbedo, cudaSurfaceObject_t texNormal, cudaSurfaceObject_t texMotion, cudaSurfaceObject_t texDepth, const int3 worldOrigin, const uint32_t* indirection, const void* sectors, const uint64_t* occupancy, const uint8_t* data, const uint64_t* sectorMasks, cudaTextureObject_t textureAtlas) {
    uint2 gid = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (gid.x >= width || gid.y >= height) return;

    float2 pixelCenter = make_float2(gid.x, gid.y) + 0.5f;
    float2 jitteredCoord = pixelCenter + c_camera.jitter;
    float2 uv = jitteredCoord / make_float2(width, height);
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(c_camera.forward + ndc.x * c_camera.right + ndc.y * c_camera.up);

    float startDist = tex2D<float>(halfDistTex, uv.x, uv.y);
    hitInfo hit = trace(c_camera.position + startDist * dir, dir, indirection, (SectorInfo*)sectors, (uint64_t*)occupancy, (uint8_t*)data, (uint64_t*)sectorMasks, worldOrigin, &c_charData);

    float depth = 100000.0f;
    float3 irradiance = make_float3(0.0f);
    float3 albedo = make_float3(0.0f);
    float3 normal = make_float3(0.0f);
    float2 motionVector = make_float2(0.0f);

    float3 c_sunColor = make_float3(1.0f, 0.95f, 0.9f) * 12.0f;

    if (hit.hit) {
        depth = length(hit.pos - c_camera.position);
        normal = make_float3(hit.normal.x, hit.normal.y, hit.normal.z);

        if (depth < 50000.0f) {
            float4 currentClipPos = c_camera.unjitteredViewProjection * make_float4(hit.pos.x, hit.pos.y, hit.pos.z, 1.0f);
            float4 previousClipPos = c_camera.prevUnjitteredViewProjection * make_float4(hit.pos.x, hit.pos.y, hit.pos.z, 1.0f);
            if (previousClipPos.w > 0.0f && currentClipPos.w > 0.0f) {
                float2 prevNDC = make_float2(previousClipPos.x / previousClipPos.w, previousClipPos.y / previousClipPos.w);
                float2 currNDC = make_float2(currentClipPos.x / currentClipPos.w, currentClipPos.y / currentClipPos.w);
                motionVector = 0.5f * (currNDC - prevNDC);
                motionVector.y = -motionVector.y;
            }
        }

        bool isWater = (hit.pos.y <= 3.001f && normal.y > 0.8f);
        if (isWater) {
            albedo = sampleTexture(hit.uv, hit.matID, hit.normal, textureAtlas, depth * depth);
            float nx = fbm3D(hit.pos.x, hit.pos.z, c_frame.time, 3, 0.06f, 2.0f, 0.6f);
            float ny = fbm3D(hit.pos.z, hit.pos.x, c_frame.time + 112.0f, 3, 0.06f, 2.0f, 0.6f);
            float3 distNormal = normalize(normal + make_float3(nx * 0.1f, 0.0f, ny * 0.1f));
            float3 reflDir = reflect(dir, distNormal);
            
            hitInfo reflHit = trace(hit.pos, reflDir, indirection, (SectorInfo*)sectors, (uint64_t*)occupancy, (uint8_t*)data, (uint64_t*)sectorMasks, worldOrigin, &c_charData);
            float3 reflectColor = make_float3(0.0f);

            if (reflHit.hit) {
                float distSq = dot(reflHit.pos - hit.pos, reflHit.pos - hit.pos);
                float3 rAlbedo = sampleTexture(reflHit.uv, reflHit.matID, reflHit.normal, textureAtlas, distSq);
                bool rShadow = traceShadow(reflHit.pos + make_float3(reflHit.normal.x, reflHit.normal.y, reflHit.normal.z) * 0.01f, c_frame.sunDirection, 200.0f, 128, indirection, (SectorInfo*)sectors, (uint64_t*)occupancy, (uint8_t*)data, (uint64_t*)sectorMasks, worldOrigin, &c_charData);
                reflectColor = rAlbedo * (rShadow ? make_float3(0.05f) : c_sunColor);
            } else {
                reflectColor = sampleSky(reflDir, c_frame.sunDirection);
            }

            float3 viewDir = -dir;
            float3 halfVec = normalize(viewDir + c_frame.sunDirection);
            float NdotH = fmaxf(dot(distNormal, halfVec), 0.0f);
            float specular = powf(NdotH, 512.0f) * 4.0f;
            float NdotV = fmaxf(dot(distNormal, viewDir), 0.0f);
            float fresnel = 0.02f + 0.98f * powf(1.0f - NdotV, 5.0f);

            bool waterShadow = traceShadow(reflHit.pos, c_frame.sunDirection, 200.0f, 128, indirection, (SectorInfo*)sectors, (uint64_t*)occupancy, (uint8_t*)data, (uint64_t*)sectorMasks, worldOrigin, &c_charData);
            irradiance = (reflectColor * fresnel) + (c_sunColor * specular * (waterShadow ? 0.0f : 1.0f));
            irradiance /= (albedo + make_float3(0.001f));
        } else {
            albedo = sampleTexture(hit.uv, hit.matID, hit.normal, textureAtlas, depth * depth);
            bool isShadowed = traceShadow(hit.pos + normal * 0.005f, c_frame.sunDirection, 200.0f, 128, indirection, (SectorInfo*)sectors, (uint64_t*)occupancy, (uint8_t*)data, (uint64_t*)sectorMasks, worldOrigin, &c_charData);
            float NdotL = fmaxf(dot(normal, c_frame.sunDirection), 0.0f);
            irradiance = c_sunColor * NdotL * (isShadowed ? 0.02f : 1.0f);
        }
    } else {
        irradiance = sampleSky(dir, c_frame.sunDirection);
        albedo = make_float3(1.0f);
        
        float3 fakePos = c_camera.position + dir * 1000.0f;
        float4 currentClipPos = c_camera.unjitteredViewProjection * make_float4(fakePos.x, fakePos.y, fakePos.z, 1.0f);
        float4 previousClipPos = c_camera.prevUnjitteredViewProjection * make_float4(fakePos.x, fakePos.y, fakePos.z, 1.0f);
        if (previousClipPos.w > 0.0f && currentClipPos.w > 0.0f) {
            float2 prevNDC = make_float2(previousClipPos.x / previousClipPos.w, previousClipPos.y / previousClipPos.w);
            float2 currNDC = make_float2(currentClipPos.x / currentClipPos.w, currentClipPos.y / currentClipPos.w);
            motionVector = 0.5f * (currNDC - prevNDC);
            motionVector.y = -motionVector.y;
        }
    }

    surf2Dwrite(make_half4(irradiance.x, irradiance.y, irradiance.z, 1.0f), texDirectLight, gid.x * sizeof(half4), gid.y);
    surf2Dwrite(float4_to_uchar4(make_float4(albedo.x, albedo.y, albedo.z, 1.0f)), texAlbedo, gid.x * sizeof(uchar4), gid.y);
    surf2Dwrite(float4_to_char4(make_float4(normal.x, normal.y, normal.z, 1.0f)), texNormal, gid.x * sizeof(char4), gid.y);
    surf2Dwrite(make_half2(motionVector.x, motionVector.y), texMotion, gid.x * sizeof(half2), gid.y);
    surf2Dwrite(depth, texDepth, gid.x * sizeof(float), gid.y);
}

void launch_GBufferAndDirectLight(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t halfDistIn, cudaSurfaceObject_t directLightOut, cudaSurfaceObject_t albedoOut, cudaSurfaceObject_t normalOut, cudaSurfaceObject_t motionOut, cudaSurfaceObject_t depthOut, const int3& worldOrigin, const uint32_t* indirection, const void* sectors, const uint64_t* occupancy, const uint8_t* data, const uint64_t* sectorMasks, cudaTextureObject_t texObj) {
    dim3 threads(8, 8);
    dim3 blocks((width + 7) / 8, (height + 7) / 8);
    GBufferAndDirectLightKernel<<<blocks, threads, 0, stream>>>(width, height, halfDistIn, directLightOut, albedoOut, normalOut, motionOut, depthOut, worldOrigin, indirection, sectors, occupancy, data, sectorMasks, texObj);
}

// ---------------------------------------------------------------------------------------------------------
// Pass 2: IndirectBounce
__global__ void IndirectBounceKernel(uint32_t width, uint32_t height, cudaTextureObject_t texNormal, cudaTextureObject_t texDepth, cudaSurfaceObject_t rawIndirectOut, const int3 worldOrigin, const uint32_t* indirection, const void* sectors, const uint64_t* occupancy, const uint8_t* data, const uint64_t* sectorMasks, cudaTextureObject_t textureAtlas) {
    uint2 gid = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (gid.x >= width || gid.y >= height) return;

    float depth = tex2D<float>(texDepth, gid.x, gid.y);
    if (depth > 50000.0f) {
        surf2Dwrite(make_half4(make_half(0), make_half(0), make_half(0), make_half(0)), rawIndirectOut, gid.x * sizeof(half4), gid.y);
        return;
    }

    char4 normalChar = tex2D<char4>(texNormal, gid.x, gid.y);
    float3 normal = make_float3(normalChar.x / 127.0f, normalChar.y / 127.0f, normalChar.z / 127.0f);
    float2 uv = (make_float2(gid.x, gid.y) + 0.5f) / make_float2(width, height);
    float3 pos = reconstructPos(depth, uv, c_camera);

    uint voxelHash = hash3_to_1(make_int3((int)floorf(pos.x * 1024.0f), (int)floorf(pos.y * 1024.0f), (int)floorf(pos.z * 1024.0f)));
    uint seed = voxelHash + uint(c_frame.time * 123456.0f);

    float3 N = normal;
    if (dot(N, N) < 0.5f) {
        surf2Dwrite(make_half4(make_half(0.02f), make_half(0.02f), make_half(0.02f), make_half(1.0f)), rawIndirectOut, gid.x * sizeof(half4), gid.y);
        return;
    }

    float3 helper = fabsf(N.x) > 0.99f ? make_float3(0,0,1) : make_float3(1,0,0);
    float3 Tangent = normalize(cross(N, helper));
    float3 Bitangent = cross(N, Tangent);

    float r1 = rand_float(&seed);
    float r2 = rand_float(&seed);
    float phi = 2.0f * PI * r1;
    float cosTheta = sqrtf(fmaxf(1.0f - r2, 0.0f));
    float sinTheta = sqrtf(r2);
    float3 localDir = make_float3(sinTheta * cosf(phi), cosTheta, sinTheta * sinf(phi));
    float3 rayDir = normalize(localDir.x * Tangent + localDir.y * N + localDir.z * Bitangent);

    hitInfo hit = trace(pos + normal * 0.01f, rayDir, indirection, (SectorInfo*)sectors, (uint64_t*)occupancy, (uint8_t*)data, (uint64_t*)sectorMasks, worldOrigin, &c_charData);

    float3 incomingLight = make_float3(0.0f);
    float3 c_sunColor = make_float3(1.0f, 0.95f, 0.9f) * 12.0f;

    if (hit.hit) {
        bool isShadowed = traceShadow(hit.pos + hit.normal * 0.01f, c_frame.sunDirection, 80.0f, 64, indirection, (SectorInfo*)sectors, (uint64_t*)occupancy, (uint8_t*)data, (uint64_t*)sectorMasks, worldOrigin, &c_charData);
        float distSq = (depth * depth) + dot(hit.pos - pos, hit.pos - pos);
        float3 bouncedAlbedo = sampleTexture(hit.uv, hit.matID, hit.normal, textureAtlas, distSq);
        float NdotL = fmaxf(dot(hit.normal, c_frame.sunDirection), 0.0f);
        float3 directLightAtHit = c_sunColor * NdotL * (isShadowed ? 0.0f : 1.0f);
        incomingLight = (directLightAtHit * bouncedAlbedo) + (bouncedAlbedo * 0.05f);
    } else {
        float3 skyLight = sampleSky(rayDir, c_frame.sunDirection);
        float luma = dot(skyLight, make_float3(0.3f, 0.59f, 0.11f));
        incomingLight = lerp(skyLight, make_float3(luma), 0.6f) * 0.25f;
    }

    surf2Dwrite(make_half4(incomingLight.x, incomingLight.y, incomingLight.z, 1.0f), rawIndirectOut, gid.x * sizeof(half4), gid.y);
}

void launch_IndirectBounce(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t normalIn, cudaTextureObject_t depthIn, cudaSurfaceObject_t rawIndirectOut, const int3& worldOrigin, const uint32_t* indirection, const void* sectors, const uint64_t* occupancy, const uint8_t* data, const uint64_t* sectorMasks, cudaTextureObject_t texObj) {
    dim3 threads(8, 8);
    dim3 blocks((width + 7) / 8, (height + 7) / 8);
    IndirectBounceKernel<<<blocks, threads, 0, stream>>>(width, height, normalIn, depthIn, rawIndirectOut, worldOrigin, indirection, sectors, occupancy, data, sectorMasks, texObj);
}

// ---------------------------------------------------------------------------------------------------------
// Pass 3: Temporal Accumulation
__device__ __forceinline__ float3 RGBToYCoCg(float3 rgb) {
    float Y  = dot(rgb, make_float3(0.25f, 0.50f, 0.25f));
    float Co = dot(rgb, make_float3(0.50f, 0.00f, -0.50f));
    float Cg = dot(rgb, make_float3(-0.25f, 0.50f, -0.25f));
    return make_float3(Y, Co, Cg);
}

__device__ __forceinline__ float3 YCoCgToRGB(float3 ycocg) {
    float Y  = ycocg.x, Co = ycocg.y, Cg = ycocg.z;
    return make_float3(Y + Co - Cg, Y + Cg, Y - Co - Cg);
}

__global__ void TemporalAccumulationKernel(uint32_t width, uint32_t height, cudaTextureObject_t rawIndirectIn, cudaTextureObject_t directIn, cudaTextureObject_t motionIn, cudaTextureObject_t depthIn, cudaTextureObject_t prevDepthIn, cudaTextureObject_t prevAccumIn, cudaSurfaceObject_t accumOut, bool resetHistory) {
    uint2 gid = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (gid.x >= width || gid.y >= height) return;

    half4 dL = tex2D<half4>(directIn, gid.x, gid.y);
    half4 ind = tex2D<half4>(rawIndirectIn, gid.x, gid.y);
    float3 currentDirect = make_float3(dL.x, dL.y, dL.z);
    float3 currentIndirect = make_float3(ind.x, ind.y, ind.z);
    float3 currentRGB = currentDirect + currentIndirect;

    if (isnan(currentRGB.x) || isinf(currentRGB.x)) {
        surf2Dwrite(make_half4(make_half(1.0f), make_half(0.0f), make_half(1.0f), make_half(1.0f)), accumOut, gid.x * sizeof(half4), gid.y);
        return;
    }

    half2 mv = tex2D<half2>(motionIn, gid.x, gid.y);
    float2 motion = make_float2((float)mv.x, (float)mv.y);
    float velMag = length(motion);
    float movementFactor = fminf(fmaxf(velMag * 200.0f, 0.0f), 1.0f);

    float2 uv = (make_float2(gid.x, gid.y) + 0.5f) / make_float2(width, height);
    float2 prevUV = uv - motion;

    float3 m1 = make_float3(0.0f);
    float3 m2 = make_float3(0.0f);

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            uint tx = min(max(gid.x + x, 0u), width - 1);
            uint ty = min(max(gid.y + y, 0u), height - 1);
            half4 ntD = tex2D<half4>(directIn, tx, ty);
            half4 ntI = tex2D<half4>(rawIndirectIn, tx, ty);
            float3 neighborRGB = make_float3(ntD.x, ntD.y, ntD.z) + make_float3(ntI.x, ntI.y, ntI.z);
            float3 neighborYCoCg = RGBToYCoCg(neighborRGB);
            m1 += neighborYCoCg;
            m2 += neighborYCoCg * neighborYCoCg;
        }
    }

    float3 mu = m1 / 9.0f;
    float3 sigma_sq = m2 / 9.0f - mu * mu;
    float3 sigma = make_float3(sqrtf(fmaxf(sigma_sq.x, 0.0f)), sqrtf(fmaxf(sigma_sq.y, 0.0f)), sqrtf(fmaxf(sigma_sq.z, 0.0f)));

    float gamma = lerp(10.0f, 0.75f, movementFactor);
    float3 minColor = mu - gamma * sigma;
    float3 maxColor = mu + gamma * sigma;

    float3 historyRGB = currentRGB;
    if (!resetHistory) {
        half4 tempH = tex2D<half4>(prevAccumIn, prevUV.x, prevUV.y);
        historyRGB = make_float3(tempH.x, tempH.y, tempH.z);
    }

    if (isnan(historyRGB.x)) historyRGB = currentRGB;

    float3 historyYCoCg = RGBToYCoCg(historyRGB);
    float3 clampedHistoryYCoCg = make_float3(fmaxf(fminf(historyYCoCg.x, maxColor.x), minColor.x), fmaxf(fminf(historyYCoCg.y, maxColor.y), minColor.y), fmaxf(fminf(historyYCoCg.z, maxColor.z), minColor.z));
    float3 clampedHistoryRGB = YCoCgToRGB(clampedHistoryYCoCg);

    float blendWeight = lerp(0.98f, 0.9f, movementFactor);
    if (resetHistory) blendWeight = 0.0f;

    bool validHistory = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f);
    if (validHistory) {
        float currentDepth = tex2D<float>(depthIn, gid.x, gid.y);
        float prevDepth = tex2D<float>(prevDepthIn, prevUV.x * width, prevUV.y * height);
        float diff = fabsf(currentDepth - prevDepth) / (currentDepth + 1e-5f);
        if (diff > 0.05f) blendWeight = 0.0f;
    } else {
        blendWeight = 0.0f;
    }

    float3 result = lerp(currentRGB, clampedHistoryRGB, blendWeight);
    surf2Dwrite(make_half4(make_half(result.x), make_half(result.y), make_half(result.z), make_half(1.0f)), accumOut, gid.x * sizeof(half4), gid.y);
}

void launch_TemporalAccumulation(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t rawIndirectIn, cudaTextureObject_t directIn, cudaTextureObject_t motionIn, cudaTextureObject_t depthIn, cudaTextureObject_t prevDepthIn, cudaTextureObject_t prevAccumIn, cudaSurfaceObject_t accumOut, bool resetHistory) {
    dim3 threads(8, 8);
    dim3 blocks((width + 7) / 8, (height + 7) / 8);
    TemporalAccumulationKernel<<<blocks, threads, 0, stream>>>(width, height, rawIndirectIn, directIn, motionIn, depthIn, prevDepthIn, prevAccumIn, accumOut, resetHistory);
}

// ---------------------------------------------------------------------------------------------------------
// Pass 4: Bilateral Denoise
__global__ void BilateralDenoiseKernel(uint32_t width, uint32_t height, cudaTextureObject_t accumIn, cudaTextureObject_t normalIn, cudaTextureObject_t depthIn, cudaSurfaceObject_t denoisedOut, float step_width) {
    uint2 gid = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (gid.x >= width || gid.y >= height) return;

    half4 cC = tex2D<half4>(accumIn, gid.x, gid.y);
    char4 cN = tex2D<char4>(normalIn, gid.x, gid.y);
    float3 centerC = make_float3(cC.x, cC.y, cC.z);
    float3 centerN = make_float3(cN.x / 127.0f, cN.y / 127.0f, cN.z / 127.0f);
    float centerD = tex2D<float>(depthIn, gid.x, gid.y);

    const float kernelWeights[3] = { 1.0f, 2.0f, 4.0f }; // Metal kernel is 1.0f, 2.0f/1.0f, 4.0f/1.0f
    float3 sumColor = make_float3(0.0f);
    float sumWeight = 0.0f;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            int2 offset = make_int2(x * step_width, y * step_width);
            uint2 tapCoord = make_uint2(gid.x + offset.x, gid.y + offset.y);
            
            if (tapCoord.x >= width || tapCoord.y >= height) tapCoord = gid;

            half4 tC = tex2D<half4>(accumIn, tapCoord.x, tapCoord.y);
            char4 tN = tex2D<char4>(normalIn, tapCoord.x, tapCoord.y);
            float3 tapC = make_float3(tC.x, tC.y, tC.z);
            float3 tapN = make_float3(tN.x / 127.0f, tN.y / 127.0f, tN.z / 127.0f);
            float tapD = tex2D<float>(depthIn, tapCoord.x, tapCoord.y);

            float dotN = fmaxf(dot(centerN, tapN), 0.0f);
            float wNormal = powf(dotN, 16.0f);
            float wDepth = (fabsf(centerD - tapD) < 1.5f) ? 1.0f : 0.0f;
            float kWeight = kernelWeights[abs(x)] * kernelWeights[abs(y)];
            float w = wNormal * wDepth * kWeight;

            sumColor += tapC * w;
            sumWeight += w;
        }
    }

    if (sumWeight < 1e-4f) {
        sumColor = centerC;
        sumWeight = 1.0f;
    }
    float3 res = sumColor / sumWeight;
    surf2Dwrite(make_half4(make_half(res.x), make_half(res.y), make_half(res.z), make_half(1.0f)), denoisedOut, gid.x * sizeof(half4), gid.y);
}

void launch_BilateralDenoise(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t accumIn, cudaTextureObject_t normalIn, cudaTextureObject_t depthIn, cudaSurfaceObject_t denoisedOut, float stepWidth) {
    dim3 threads(8, 8);
    dim3 blocks((width + 7) / 8, (height + 7) / 8);
    BilateralDenoiseKernel<<<blocks, threads, 0, stream>>>(width, height, accumIn, normalIn, depthIn, denoisedOut, stepWidth);
}

// ---------------------------------------------------------------------------------------------------------
// Pass 5: Volumetric Fog
__device__ __forceinline__ float InterleavedGradientNoise(float2 pos) {
    float mx = pos.x * 0.06711056f + pos.y * 0.00583715f;
    float mxf = mx - floorf(mx);
    float mx2 = 52.9829189f * mxf;
    return mx2 - floorf(mx2);
}

__device__ __forceinline__ float phaseFunction(float3 viewDir, float3 lightDir, float g) {
    float cosTheta = dot(viewDir, lightDir);
    float denom = 1.0f + g * g - 2.0f * g * cosTheta;
    return (1.0f - g * g) / (4.0f * 3.14159f * powf(denom, 1.5f));
}

__global__ void VolumetricFogKernel(uint32_t width, uint32_t height, cudaTextureObject_t texDepth, cudaTextureObject_t texHistory, cudaSurfaceObject_t texVolumetric, const int3 worldOrigin, const uint32_t* indirection, const void* sectors, const uint64_t* occupancy, const uint64_t* sectorMasks) {
    uint2 gid = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (gid.x >= width || gid.y >= height) return;

    uint2 fullResCoord = make_uint2(gid.x * 2, gid.y * 2);
    // Wait, width and height passed are for the half-res texture, so texDepth is 2x larger
    if (fullResCoord.x >= width * 2) fullResCoord.x = width * 2 - 1;
    if (fullResCoord.y >= height * 2) fullResCoord.y = height * 2 - 1;

    float depth = tex2D<float>(texDepth, fullResCoord.x, fullResCoord.y);
    float2 uv = (make_float2(gid.x, gid.y) + 0.5f) / make_float2(width, height);
    float clampedDepth = fminf(depth, 180.0f); // hardcoded maxdist 180

    float3 endPos = reconstructPos(clampedDepth, uv, c_camera);
    float3 startPos = c_camera.position;
    
    float3 rayVec = endPos - startPos;
    float rayLength = length(rayVec);
    float3 rayDir = rayLength > 1e-4f ? rayVec / rayLength : make_float3(0,1,0);

    float dither = InterleavedGradientNoise(make_float2(gid.x, gid.y) + make_float2(c_frame.time * 5.588f, c_frame.time * 5.588f));
    int STEPS = 32;
    float stepSize = rayLength / float(STEPS);
    float currentT = stepSize * dither;
    
    float3 accumulatedLight = make_float3(0.0f);
    float accumulatedTransmittance = 1.0f;
    
    float density = 0.004f;
    float3 fogColor = make_float3(0.6f, 0.7f, 0.8f);
    float anisotropy = 0.5f;
    float phase = phaseFunction(rayDir, c_frame.sunDirection, anisotropy);
    float3 sunColor = make_float3(1.0f, 0.95f, 0.9f) * 12.0f;

    for (int i = 0; i < STEPS; i++) {
        float3 pos = startPos + rayDir * currentT;
        bool isShadowed = false;
        if (currentT > 2.0f && currentT < 200.0f && phase > 0.04f) {
            isShadowed = traceShadow(pos, c_frame.sunDirection, 80.0f, 64, indirection, (SectorInfo*)sectors, (uint64_t*)occupancy, nullptr, (uint64_t*)sectorMasks, worldOrigin, &c_charData);
        }

        if (!isShadowed) {
            accumulatedLight += sunColor * phase * density * accumulatedTransmittance * stepSize;
        }
        accumulatedLight += (fogColor * 0.05f) * density * accumulatedTransmittance * stepSize;
        accumulatedTransmittance *= expf(-density * stepSize);
        currentT += stepSize;
    }

    float4 prevClip = c_camera.prevUnjitteredViewProjection * make_float4(endPos.x, endPos.y, endPos.z, 1.0f);
    float3 history = make_float3(0.0f);
    float blendFactor = 0.0f;

    if (prevClip.w > 0.01f) {
        float2 prevNDC = make_float2(prevClip.x / prevClip.w, prevClip.y / prevClip.w);
        float2 prevUV = make_float2(prevNDC.x * 0.5f + 0.5f, 0.5f - prevNDC.y * 0.5f);
        if (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f) {
            half4 hV = tex2D<half4>(texHistory, prevUV.x, prevUV.y);
            history = make_float3(hV.x, hV.y, hV.z);
            float diff = length(history - accumulatedLight);
            blendFactor = 0.8f;
            if (diff > 1.0f) blendFactor = 0.4f;
        }
    }

    float3 result = lerp(accumulatedLight, history, blendFactor);
    surf2Dwrite(make_half4(make_half(result.x), make_half(result.y), make_half(result.z), make_half(1.0f)), texVolumetric, gid.x * sizeof(half4), gid.y);
}

void launch_VolumetricFog(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t depthIn, cudaTextureObject_t prevVolumetricIn, cudaSurfaceObject_t volumetricOut, const int3& worldOrigin, const uint32_t* indirection, const void* sectors, const uint64_t* occupancy, const uint8_t* data, const uint64_t* sectorMasks) {
    dim3 threads(8, 8);
    dim3 blocks((width + 7) / 8, (height + 7) / 8);
    VolumetricFogKernel<<<blocks, threads, 0, stream>>>(width, height, depthIn, prevVolumetricIn, volumetricOut, worldOrigin, indirection, sectors, occupancy, sectorMasks);
}

// ---------------------------------------------------------------------------------------------------------
// Pass 6: Compute Exposure
__device__ __forceinline__ float getLuminance(float3 color) {
    return dot(color, make_float3(0.2126f, 0.7152f, 0.0722f));
}

__global__ void ComputeExposureKernel(uint32_t width, uint32_t height, cudaTextureObject_t texDirect, cudaTextureObject_t texAccum, cudaTextureObject_t texAlbedo, ExposureData* exposure) {
    __shared__ float sharedLogLum[256];
    
    uint pixelCount = 0;
    float localLogSum = 0.0f;
    uint strideX = 32;
    uint strideY = 32;
    uint tX = threadIdx.x;
    uint tY = threadIdx.y;

    for (uint y = tY * strideY; y < height; y += 16 * strideY) {
        for (uint x = tX * strideX; x < width; x += 16 * strideX) {
            half4 dV = tex2D<half4>(texDirect, x, y);
            half4 aV = tex2D<half4>(texAccum, x, y);
            uchar4 albV = tex2D<uchar4>(texAlbedo, x, y);
            
            float3 direct = make_float3(dV.x, dV.y, dV.z);
            float3 indirect = make_float3(aV.x, aV.y, aV.z);
            float3 albedo = make_float3(albV.x / 255.0f, albV.y / 255.0f, albV.z / 255.0f);
            
            float3 color = (direct + indirect) * albedo;
            float lum = getLuminance(color);
            
            float2 uv = make_float2(x, y) / make_float2(width, height);
            float dist = length(uv - make_float2(0.5f, 0.5f));
            float weight = 1.0f - smoothstep(0.2f, 0.6f, dist);
            weight = fmaxf(weight, 0.1f);
            
            localLogSum += logf(fmaxf(lum, 0.0001f)) * weight;
            pixelCount++;
        }
    }
    
    sharedLogLum[tY * 16 + tX] = (pixelCount > 0) ? (localLogSum / float(pixelCount)) : -9.0f;
    __syncthreads();
    
    uint linearTid = tY * 16 + tX;
    if (linearTid < 128) sharedLogLum[linearTid] += sharedLogLum[linearTid + 128]; __syncthreads();
    if (linearTid < 64)  sharedLogLum[linearTid] += sharedLogLum[linearTid + 64];  __syncthreads();
    if (linearTid < 32)  sharedLogLum[linearTid] += sharedLogLum[linearTid + 32];  __syncthreads();
    if (linearTid < 16)  sharedLogLum[linearTid] += sharedLogLum[linearTid + 16];  __syncthreads();
    if (linearTid < 8)   sharedLogLum[linearTid] += sharedLogLum[linearTid + 8];   __syncthreads();
    if (linearTid < 4)   sharedLogLum[linearTid] += sharedLogLum[linearTid + 4];   __syncthreads();
    if (linearTid < 2)   sharedLogLum[linearTid] += sharedLogLum[linearTid + 2];   __syncthreads();
    if (linearTid < 1)   sharedLogLum[linearTid] += sharedLogLum[linearTid + 1];   __syncthreads();
    
    if (linearTid == 0) {
        float avgLogLum = sharedLogLum[0] / 256.0f;
        float currentSceneLum = expf(avgLogLum);
        currentSceneLum = fmaxf(fminf(currentSceneLum, 60.0f), 0.01f);
        
        float lastLum = exposure->sceneLuminance;
        float adaptationSpeed = (currentSceneLum > lastLum) ? 4.0f : 1.0f;
        float interpolatedLum = lastLum + (currentSceneLum - lastLum) * (1.0f - expf(-c_frame.deltaTime * adaptationSpeed));
        
        if (isnan(interpolatedLum)) interpolatedLum = 0.5f;
        exposure->sceneLuminance = interpolatedLum;
    }
}

void launch_ComputeExposure(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t texDirect, cudaTextureObject_t texAccum, cudaTextureObject_t texAlbedo, void* exposureBuffer) {
    dim3 threads(16, 16);
    dim3 blocks(1, 1);
    ComputeExposureKernel<<<blocks, threads, 0, stream>>>(width, height, texDirect, texAccum, texAlbedo, (ExposureData*)exposureBuffer);
}

// ---------------------------------------------------------------------------------------------------------
// Pass 7: Composite
__device__ __forceinline__ float3 ACESFilm(float3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return make_float3(
        fmaxf(0.0f, fminf((x.x*(a*x.x+b))/(x.x*(c*x.x+d)+e), 1.0f)),
        fmaxf(0.0f, fminf((x.y*(a*x.y+b))/(x.y*(c*x.y+d)+e), 1.0f)),
        fmaxf(0.0f, fminf((x.z*(a*x.z+b))/(x.z*(c*x.z+d)+e), 1.0f))
    );
}

__device__ __forceinline__ float3 LinearToSRGB(float3 color) {
    return make_float3(
        color.x <= 0.0031308f ? 12.92f * color.x : 1.055f * powf(color.x, 1.0f / 2.4f) - 0.055f,
        color.y <= 0.0031308f ? 12.92f * color.y : 1.055f * powf(color.y, 1.0f / 2.4f) - 0.055f,
        color.z <= 0.0031308f ? 12.92f * color.z : 1.055f * powf(color.z, 1.0f / 2.4f) - 0.055f
    );
}

__device__ __forceinline__ float3 applySaturation(float3 color, float saturation) {
    float luma = dot(color, make_float3(0.2126f, 0.7152f, 0.0722f));
    return lerp(make_float3(luma), color, saturation);
}

__global__ void CompositeKernel(uint32_t width, uint32_t height, cudaTextureObject_t texDirect, cudaTextureObject_t texAccum, cudaTextureObject_t texAlbedo, cudaTextureObject_t texVolumetric, cudaSurfaceObject_t texFinal, cudaSurfaceObject_t compositeResult, ExposureData* exposure) {
    uint2 gid = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (gid.x >= width || gid.y >= height) return;

    half4 dL = tex2D<half4>(texDirect, gid.x, gid.y);
    half4 ind = tex2D<half4>(texAccum, gid.x, gid.y);
    uchar4 alb = tex2D<uchar4>(texAlbedo, gid.x, gid.y);
    
    float3 directLight = make_float3(dL.x, dL.y, dL.z);
    float3 indirectLight = make_float3(ind.x, ind.y, ind.z);
    float3 albedo = make_float3(alb.x / 255.0f, alb.y / 255.0f, alb.z / 255.0f);

    float2 uv = (make_float2(gid.x, gid.y) + 0.5f) / make_float2(width, height);
    half4 vol = tex2D<half4>(texVolumetric, uv.x, uv.y);
    float3 fog = make_float3(vol.x, vol.y, vol.z);

    float3 linearColor = (directLight + indirectLight) * albedo + fog;

    float avgLum = exposure->sceneLuminance;
    float exposureScale = 0.15f / (fmaxf(avgLum, 0.001f));
    linearColor *= exposureScale;
    
    linearColor = applySaturation(linearColor, 1.2f);
    float3 toneMapped = ACESFilm(linearColor);
    float3 finalColor = LinearToSRGB(toneMapped);

    surf2Dwrite(make_half4(make_half(linearColor.x), make_half(linearColor.y), make_half(linearColor.z), make_half(1.0f)), compositeResult, gid.x * sizeof(half4), gid.y);
    surf2Dwrite(float4_to_uchar4(make_float4(finalColor.x, finalColor.y, finalColor.z, 1.0f)), texFinal, gid.x * sizeof(uchar4), gid.y);
}

void launch_Composite(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t directIn, cudaTextureObject_t albedoIn, cudaTextureObject_t denoisedIn, cudaTextureObject_t volumetricIn, cudaSurfaceObject_t finalHistoryOut, cudaSurfaceObject_t compositeResultOut, void* exposureBuffer) {
    dim3 threads(8, 8);
    dim3 blocks((width + 7) / 8, (height + 7) / 8);
    // Note: finalHistoryOut is effectively texFinal (the swapchain target)
    CompositeKernel<<<blocks, threads, 0, stream>>>(width, height, directIn, denoisedIn, albedoIn, volumetricIn, finalHistoryOut, compositeResultOut, (ExposureData*)exposureBuffer);
}

#endif
