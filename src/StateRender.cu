#include "StateRender.cuh"
#include "CArray.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cumath.cuh"
#include "csdf.cuh"
#include "cuda_fp16.h"
#include "RayTracing.cuh"
#include <curand_kernel.h> // needed if you use random roughness per pixel


__constant__ float3 c_sunDir;
__constant__ float3 c_camPos, c_camFo, c_camUp, c_camRi;


__device__ __forceinline__ float3 sampleSky(float3 dir)
{
    // --- Sky color ---
    float sunDot = dot(dir, c_sunDir);
    if (sunDot > 0.999f) {
        // Bright yellow sun
        return make_float3(1.0f, 0.9f, 0.2f) * 10.0f;
    } else {
        // Blue-ish sky, darkens towards horizon
        return lerp(make_float3(0.5f, 0.7f, 1.0f), make_float3(0.0f, 0.0f, 0.0f), dir.y * 0.5f + 0.5f);
    }
}

__device__ __forceinline__ float3 sampleTexture(half2 uv, cudaTextureObject_t texObj)
{
    const half2 whichBlock = make_half2(1.0f, 0.0f);
    uv.x = uv.x * hrcp(16.0) + whichBlock.x;
    uv.y = uv.y * hrcp(16.0) + whichBlock.y;

    float4 t = tex2D<float4>(texObj, __half2float(uv.y), __half2float(uv.x));

    return make_float3(t.x, t.y, t.z);
}

// Function which returns the color of a pixel, first casting a primary ray, and depending on this result a shadow ray/ reflections.
__device__ float3 computeColor(float x,
                               float y,
                               half distance,
                               const uint32_t* __restrict__ bits,
                               const unsigned char* __restrict__ csdf,
                               cudaTextureObject_t texObj) 
{
    // --- Primary ray direction ---
    float2 NDC = make_float2(x * 2.0f - 1.0f, y * 2.0f - 1.0f);
    float3 dir = normalize(c_camFo + NDC.x * c_camRi + NDC.y * c_camUp);

    // --- Trace primary ray ---
    hitInfo hit = trace(c_camPos, dir, distance, bits, csdf);

    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    if (hit.hit) {
        // --- Direct lighting (Lambertian diffuse) ---
        float diffuse = fmaxf(dot(hit.normal, c_sunDir), 0.0f);
        float3 baseColor = sampleTexture(hit.uv, texObj);
        baseColor = baseColor * diffuse;

        // --- Shadow ray ---
        hitInfo shadow = trace(hit.pos + hit.normal * 1e-1f, c_sunDir, (half)0.0f, bits, csdf);
        if (shadow.hit) {
            baseColor = baseColor * 0.2f;  // Darker in shadow
        }
        color = color + baseColor;

        // --- Reflection bounce ---

        float3 reflDir = normalize(reflect(dir, hit.normal));
        hitInfo reflHit = trace(hit.pos + hit.normal * 1e-3f, reflDir, distance, bits, csdf);
        if (reflHit.hit) {
            // Simple reflection shading
            float reflDiff = fmaxf(dot(reflHit.normal, c_sunDir), 0.0f);
            float3 reflColor = sampleTexture(reflHit.uv, texObj) * diffuse;

            // // Shadow check for reflection
            hitInfo reflShadow = trace(reflHit.pos + reflHit.normal * 1e-3f, c_sunDir, (half)0.0f, bits, csdf);
            if (reflShadow.hit) {
                reflColor = reflColor * 0.1f;  // darker reflection if in shadow
            }
            // Add reflection (scaled to avoid over-brightness)
            color = color * 0.9 + 0.1f * reflColor;
        }
        else 
            color = color * 0.9 + 0.1f * sampleSky(reflDir);
        
    }
    else color = sampleSky(dir);
    
    // --- Example: force white debug marker if too many iterations ---
    // if (hit.its > 100) {
    //     color.x = 1.0f;
    // }

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
                             cudaTextureObject_t texObj,
                             cudaTextureObject_t shadowTexture) 
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    float x = (float)ix / (float)width;
    float y = (float)iy / (float)height;

    half dist = approximateDistance(ix, iy, width / 2, height/2, distBuffer) - (half)2.0f;

    float3 col = computeColor(x, y, dist, bits, csdf, texObj);

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
                                  float* shadowBuffer) 
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    float x = (float)ix / (float)width;
    float y = (float)iy / (float)height;

    float2 NDC = make_float2(x * 2.0f - 1.0f, y * 2.0f - 1.0f);
    float3 dir = normalize(c_camFo + NDC.x * c_camRi + NDC.y * c_camUp);

    hitInfo hit = trace(c_camPos, dir, (half)0.0f, bits, csdf);
    half dist = hit.hit ? (half)distance(hit.pos, c_camPos) : (half)MAX_DIST;

    float shadowValue = 1.0f;
    if(hit.hit)
    {
        hitInfo shadow = trace(hit.pos + hit.normal * 1e-1f, c_sunDir, (half)0.0f, bits, csdf);
        shadowValue = shadow.hit ? 0.2f : 1.0f;
    }

    //shadowBuffer[ix + iy * width] = shadowValue;
    distBuffer[ix + iy * width] = dist;
}

void StateRender::drawCUDA(const glm::vec3& pos, const glm::vec3& fo,
                           const glm::vec3& up, const glm::vec3& ri) 
{
    // Upload camera + sun constants
    cudaMemcpyToSymbol(c_camPos, &pos, sizeof(glm::vec3));
    cudaMemcpyToSymbol(c_camFo, &fo, sizeof(glm::vec3));
    cudaMemcpyToSymbol(c_camUp, &up, sizeof(glm::vec3));
    cudaMemcpyToSymbol(c_camRi, &ri, sizeof(glm::vec3));

    glm::vec3 sunDir = glm::normalize(glm::vec3(10.f, 5.f, -4.f));
    cudaMemcpyToSymbol(c_sunDir, &sunDir, sizeof(glm::vec3));


    dim3 block(16, 16);
    dim3 grid(((framebuffer.getWidth() / 2) + block.x - 1) / block.x,
              ((framebuffer.getHeight() / 2) + block.y - 1) / block.y);

    distApproximationKernel<<<grid, block>>>(
    reinterpret_cast<half*>(distBuffer.getPtr()),
                            framebuffer.getWidth() / 2,
                            framebuffer.getHeight() / 2,
                            cArray.getPtr(), 
                            csdf.getPtr(),
                            shadowTex.getDevPtr()
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
                              texturepack.texObject(),
                              shadowTex.getTexObj()
    );
}


__host__ StateRender::StateRender() 
{
}