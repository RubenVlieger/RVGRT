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

#define INCLUDEGI

__constant__ float3 c_sunDir;
__constant__ float c_time;
__constant__ float3 c_camPos, c_camFo, c_camUp, c_camRi;
__constant__ float3 c_waterColor = {0.0f, 0.1f, 0.3f};
__constant__ float c_waterReflectivity = 0.08f;

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
        // Add wave distortion to the normal for a wavy look
        float nx_wave = fbm3D(hit.pos.x, hit.pos.z, c_time, 3, 0.06f, 2.0f, 0.6f);
        float ny_wave = fbm3D(hit.pos.z, hit.pos.x, c_time + 112.0f, 3, 0.06f, 2.0f, 0.6f);
        float3 distortedNormal = normalize(hit.normal + make_float3(nx_wave * 0.1f, ny_wave * 0.1f, 0.0f));

        float3 reflDir = reflect(dir, distortedNormal);

        // *** PROBLEM 1 FIX: Use a large, constant view distance for the reflection ray. ***
        hitInfo reflHit = trace(hit.pos, reflDir, 0.001f, bits, csdf);

        float3 finalReflectionColor;
        if (reflHit.hit) {
            // We hit a voxel, get its texture and shadow it
            float3 reflColor = sampleTexture(reflHit.uv, reflHit.pos, texObj);
            hitInfo reflShadow = trace(reflHit.pos + reflHit.normal * 1e-3f, c_sunDir, 0.001f, bits, csdf);
            if (reflShadow.hit) {
                reflColor = reflColor * 0.1f; // Apply shadow to the reflection
            }
            finalReflectionColor = reflColor;
        } else {
            // The reflection ray hit the sky
            finalReflectionColor = sampleSky(reflDir, c_sunDir);
        }

        // *** PROBLEM 2 FIX: Calculate Fresnel effect for realistic reflectivity. ***
        // (Using Schlick's approximation)
        float NdotV = fmaxf(dot(hit.normal, -dir), 0.0f); // Use original normal for stable Fresnel
        float fresnelFactor = c_waterReflectivity + (1.0f - c_waterReflectivity) * powf(1.0f - NdotV, 5.0f);

        // The final color is a mix of the water's base color and the reflected scene,
        // controlled by the Fresnel factor.
        color = lerp(c_waterColor, finalReflectionColor, fresnelFactor);
    }
    else if (hit.hit)
    {
        // --- Get surface properties ---
        float3 baseColor = sampleTexture(hit.uv, hit.pos, texObj);

        // --- Direct lighting (Lambertian diffuse) ---
        float diffuse = fmaxf(dot(hit.normal, c_sunDir), 0.0f);
        float shadow = tex2D<float>(shadowTexture, x, y);
        float3 directLight = baseColor * diffuse * shadow;

        // ==================================================
        // --- GLOBAL ILLUMINATION VIA VOXEL CONE TRACING ---
        // ==================================================
#ifdef INCLUDEGI
        float3 indirectLight = make_float3(0.0f, 0.0f, 0.0f);

        // Define cone directions in a hemisphere around the surface normal.
        float3 up = hit.normal;
        float3 right = normalize(cross(up, make_float3(0.577f, 0.577f, 0.577f))); // Arbitrary non-parallel vector
        float3 forward = normalize(cross(up, right));

        // Trace cones in the hemisphere
        indirectLight += traceCone(hit.pos, up, GIdata, csdf);
        indirectLight += traceCone(hit.pos, lerp(up, right, 0.5f), GIdata, csdf);
        indirectLight += traceCone(hit.pos, lerp(up, -right, 0.5f), GIdata, csdf);
        indirectLight += traceCone(hit.pos, lerp(up, forward, 0.5f), GIdata, csdf);
        indirectLight += traceCone(hit.pos, lerp(up, -forward, 0.5f), GIdata, csdf);
        // Add one more cone for a total of 6
        indirectLight += traceCone(hit.pos, lerp(up, lerp(right, forward, 0.5f), 0.5f), GIdata, csdf);


        // Average the result and modulate by the surface color (albedo)
        // The GI_STRENGTH is an artistic control to balance GI
        const float GI_STRENGTH = 0.6f;
        indirectLight = (indirectLight / (float)NUM_CONES) * baseColor * GI_STRENGTH;

        // --- Final Color Composition ---
        // Combine direct and indirect lighting.
        // Also add a tiny bit of ambient light from the sky for areas that get no light.
        float3 ambient = sampleSky(hit.normal, c_sunDir) * 0.05f * baseColor;
        color = directLight + indirectLight + ambient;
#else
    color = directLight;
#endif

    }
    else
    {
        // No hit, sample sky
        color = sampleSky(dir, c_sunDir);
    }


    float fogfactor = 0.0f;
    if(hit.hit)
        fogfactor = powf(1.0 / 2.71828, (length(hit.pos - c_camPos) * 0.0004f));
    else fogfactor = 1.0f;

    return fogfactor * color + (1.0f - fogfactor) * make_float3(0.95f, 0.95f, 1.0f);
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

__global__ void renderKernel(
                            uchar4* framebuffer, 
                             int width, 
                             int height, 
                            size_t pitchInBytes,

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

#ifdef D3D12
    uchar4 pixel = make_uchar4(r, g, b, 255);
    framebuffer[ix + iy * (pitchInBytes / 4) ] = pixel; 
#else
    framebuffer[ix + iy * width] = make_uchar4(r, g, b, 255);
#endif
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

    dim3 block(16, 16);
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
                              framebuffer.getDevicePtr(),
                              framebuffer.getWidth(),
                              framebuffer.getHeight(),
                            framebuffer.getPitchInBytes(),
                              cArray.getPtr(), 
                              csdf.getPtr(),
                              (half*)distBuffer.getPtr(),
                              (uchar4*)GIdata.getPtr(),
                              texturepack.texObject(),
                              shadowTex.getTexObj()
    );

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
}


__host__ StateRender::StateRender() 
{
}