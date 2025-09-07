#include "StateRender.cuh"
#include "CArray.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cumath.cuh"
#include "CoarseArray.cuh"
#include "cuda_fp16.h"
#include "raytracing_functions.cuh"
#include "TerrainGeneration.cuh"
#include <chrono>

__constant__ float3 c_sunDir;
__constant__ float c_time;
__constant__ float3 c_camPos, c_camFo, c_camUp, c_camRi;

__device__ __forceinline__ float3 sampleSky(float3 dir)
{
    // --- Sky color ---
    float sunDot = dot(dir, c_sunDir);
    if (sunDot > 0.999f) {
        // Bright yellow sun
        return c_sunColor;
    } else {
        // Blue-ish sky, darkens towards horizon
        float t = clampf(0.5f * (dir.y + 1.0f), 0.0f, 1.0f); 
        // dir.y=-1 -> 0, dir.y=1 -> 1

        return lerp(make_float3(0.2f, 0.4f, 0.8f),   // horizon blue
                    make_float3(0.6f, 0.8f, 1.0f),   // zenith blue
                    t);
    }
}

__device__ __forceinline__ float3 sampleTexture(half2 uv, float3 pos, cudaTextureObject_t texObj)
{
    const half2 texStoneID = make_half2(0.0f / 16.0f, 1.0f / 16.0f);
    const half2 texDirtID = make_half2(0.0f / 16.0f, 2.0f / 16.0f);
    const half2 texCobbleID = make_half2(1.0f / 16.0f, 0.0f / 16.0f);
    const half2 texIronID = make_half2(2.0f / 16.0f, 1.0f / 16.0f);
    const half2 texDiamondID = make_half2(3.0f / 16.0f, 2.0f / 16.0f);
    const half2 texStone2ID = make_half2(0.0f / 16.0f, 0.0f / 16.0f);
    const half2 texSandStoneID = make_half2(11.0f / 16.0f, 0.0f / 16.0f);
    const half2 texCoalID = make_half2(2.0f / 16.0f, 2.0f / 16.0f);

    half2 whichBlock = make_half2(0.0f, 8.0f/16.0f);

    const float freq = 0.05f;
    float eval = simplex3D(floorf(pos.x) * freq, floorf(pos.y) * freq, floorf(pos.z)* freq);
    float eval2 = simplex3D(floorf(pos.x + 121.3) * freq * 0.3f, floorf(pos.y + 1321.3) * freq * 0.3f, floorf(pos.z + 721.5)* freq * 0.3f);
    eval = eval*0.4f + eval2 * 0.6f;

    if(eval < -1.3f) whichBlock = texStoneID;
    else if(eval < -1.2f) whichBlock = texDiamondID;
    else if(eval < -0.7f) whichBlock = texIronID;
    else if(eval < 0.0f) whichBlock = texStoneID;
    else if(eval < 0.1f) whichBlock = texCoalID;
    else if(eval < 0.4f) whichBlock = texCobbleID;
    else if(eval < 0.8f)whichBlock = texDirtID;
    else if(eval < 1.2f) whichBlock = texStone2ID;
    else whichBlock = texStoneID;

    uv.x = ((uv.x * hrcp(16.0))) + whichBlock.x;
    uv.y = ((uv.y * hrcp(16.0))) + whichBlock.y;

    float4 t = tex2D<float4>(texObj, __half2float(uv.y), __half2float(uv.x));

    return make_float3(t.x, t.y, t.z);
}

// Function which returns the color of a pixel, first casting a primary ray, and depending on this result a shadow ray/ reflections.
__device__ float3 computeColor(float x,
                               float y,
                               half distance,
                               const uint32_t* __restrict__ bits,
                               const unsigned char* __restrict__ csdf,
                               cudaTextureObject_t texObj,
                               cudaTextureObject_t shadowTexture, 
                               const uchar4* __restrict__ GIdata) 
{
    // --- Primary ray direction ---
    float2 NDC = make_float2(x * 2.0f - 1.0f, y * 2.0f - 1.0f);
    float3 dir = normalize(c_camFo + NDC.x * c_camRi + NDC.y * c_camUp);

    // --- Trace primary ray ---
    hitInfo hit = trace(c_camPos, dir, distance, bits, csdf);

    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    //If it hits water, compute the reflection!
    if(hit.hit && hit.pos.y < 31.001f)
    {
        float nx = fbm3D(hit.pos.x, hit.pos.z, c_time , 3, 0.06f, 2.0f, 0.6f);//simplex3D(hit.pos.x * 0.1f, hit.pos.z * 0.1f, c_time * 0.3f);
        float ny = fbm3D(hit.pos.z, hit.pos.x, c_time + 112.0f, 3, 0.06f, 2.0f, 0.6f);

        float3 reflDir = normalize(reflect(dir, hit.normal) + make_float3(nx*0.1f, ny*0.1f, 0.0f));
        hitInfo reflHit = trace(hit.pos + hit.normal * 1e-3f, reflDir, distance, bits, csdf);
        if (reflHit.hit) {
            float reflDiff = fmaxf(dot(reflHit.normal, c_sunDir), 0.0f);
            float3 reflColor = sampleTexture(reflHit.uv, reflHit.pos, texObj);

            hitInfo reflShadow = trace(reflHit.pos + reflHit.normal * 1e-3f, c_sunDir, (half)0.0f, bits, csdf);
            if (reflShadow.hit) {
                reflColor = reflColor * 0.1f;  // darker reflection if in shadow
            }
            // Add reflection (scaled to avoid over-brightness)
            color = lerp(reflColor, make_float3(0.0f, 0.1f, 0.3f), 0.8f);
        }
        else 
            color = sampleSky(reflDir);
    }

    else if (hit.hit) {
        // --- Direct lighting (Lambertian diffuse) ---
        float diffuse = fmaxf(dot(hit.normal, c_sunDir), 0.0f);
        float3 baseColor = sampleTexture(hit.uv, hit.pos, texObj);
        baseColor = baseColor * diffuse;

        // --- Shadow ray --- // from lower resolution as an optimization step.
        float shadow = tex2D<float>(shadowTexture, x, y);

        color = color + baseColor * shadow;
    }
    else color = sampleSky(dir);

    return color;
}

// A function which returns the result of the estimated raydistance x.
__device__ __forceinline__ half approximateDistance(int x, int y, 
                                                    int distWidth, int distHeight, 
                                                    const half* __restrict__ distBuffer)
{
    int lx = x / 2;
    int ly = y / 2;

    int lx1 = min(lx + 1, distWidth - 1);
    int ly1 = min(ly + 1, distHeight - 1);

    int idx00 = lx  + ly  * distWidth; // Top-left
    int idx10 = lx1 + ly  * distWidth; // Top-right
    int idx01 = lx  + ly1 * distWidth; // Bottom-left
    int idx11 = lx1 + ly1 * distWidth; // Bottom-right
    
    return __hmin(__hmin(distBuffer[idx00], distBuffer[idx10]), __hmin(distBuffer[idx01], distBuffer[idx11]));
}

__global__ void renderKernel(uchar4* framebuffer, 
                             int width, 
                             int height, 
                             const uint32_t* __restrict__ bits,
                             const unsigned char* __restrict__ csdf,
                             const half* __restrict__ distBuffer,
                             uchar4* __restrict__ GIdata,
                             cudaTextureObject_t texObj,
                             cudaTextureObject_t shadowTexture) 
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    float x = (float)ix / (float)width;
    float y = (float)iy / (float)height;

    half dist = approximateDistance(ix, iy, width / 2, height/2, distBuffer) - (half)2.0f;

    float3 col = computeColor(x, y, dist, bits, csdf, texObj, shadowTexture, GIdata);

    col = clamp(col, make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f));

    unsigned char r = (unsigned char)(col.x * 255.0f);
    unsigned char g = (unsigned char)(col.y * 255.0f);
    unsigned char b = (unsigned char)(col.z * 255.0f);

    framebuffer[ix + iy * width] = make_uchar4(r, g, b, 255);
}

__global__ void distApproximationKernel(half* distBuffer,
                                  int width, 
                                  int height, 
                                  const uint32_t* __restrict__ bits,
                                  const unsigned char* __restrict__ csdf,
                                  float* shadowBuffer,
                                  int shadowPitchElems) 
{
    uint64_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    float x = (float)ix / (float)width;
    float y = (float)iy / (float)height;

    float2 NDC = make_float2(x * 2.0f - 1.0f, y * 2.0f - 1.0f);
    float3 dir = normalize(c_camFo + NDC.x * c_camRi + NDC.y * c_camUp);

    hitInfo hit = trace(c_camPos, dir, (half)0.0f, bits, csdf);
    half dist = hit.hit ? (half)distance(hit.pos, c_camPos) : (half)SDF_MAX_DIST;

    float shadowValue = 1.0f;
    if(hit.hit)
    {
        hitInfo shadow = trace(hit.pos + hit.normal * 1e-1f, c_sunDir, (half)0.0f, bits, csdf);
        shadowValue = shadow.hit ? 0.2f : 1.0f;
    }
    shadowBuffer[ix + iy * shadowPitchElems] = shadowValue; // w/ 11ms
    distBuffer[ix + iy * width] = dist;
}


void StateRender::drawCUDA(const glm::vec3& pos, const glm::vec3& fo,
                           const glm::vec3& up, const glm::vec3& ri) 
{    
    // Upload camera + sun constants
    float currentTime = (float)(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() % 1000000)* 0.001f; 
    cudaMemcpyToSymbol(c_time, &currentTime, sizeof(float));
    cudaMemcpyToSymbol(c_camPos, &pos, sizeof(glm::vec3));
    cudaMemcpyToSymbol(c_camFo, &fo, sizeof(glm::vec3));
    cudaMemcpyToSymbol(c_camUp, &up, sizeof(glm::vec3));
    cudaMemcpyToSymbol(c_camRi, &ri, sizeof(glm::vec3));

    glm::vec3 sunDir = glm::normalize(glm::vec3(10.f, 5.f, -4.f));
    cudaMemcpyToSymbol(c_sunDir, &sunDir, sizeof(glm::vec3));

    dim3 block(16, 8);
    dim3 grid(((framebuffer.getWidth() / 2) + block.x - 1) / block.x,
              ((framebuffer.getHeight() / 2) + block.y - 1) / block.y);

    distApproximationKernel<<<grid, block>>>(
    reinterpret_cast<half*>(distBuffer.getPtr()),
                            framebuffer.getWidth() / 2,
                            framebuffer.getHeight() / 2,
                            cArray.getPtr(), 
                            csdf.getPtr(),
                            shadowTex.getDevPtr(),
                            shadowTex.getPitchElems()
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    block = dim3(16, 16);
    grid = dim3((framebuffer.getWidth() + block.x - 1) / block.x,
                (framebuffer.getHeight() + block.y - 1) / block.y);

    renderKernel<<<grid, block>>>(
    reinterpret_cast<uchar4*>(framebuffer.devicePtr()),
                              framebuffer.getWidth(),
                              framebuffer.getHeight(),
                              cArray.getPtr(), 
                              csdf.getPtr(),
                              (half*)distBuffer.getPtr(),
                              (uchar4*)GIdata.getPtr(),
                              texturepack.texObject(),
                              shadowTex.getTexObj()
    );
}


__host__ StateRender::StateRender() 
{
}