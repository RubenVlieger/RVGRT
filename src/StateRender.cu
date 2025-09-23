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


__constant__ float c_cam[19];
__constant__ glm::mat4 c_currentViewProjection_unjittered;
__constant__ glm::mat4 c_previousViewProjection_unjittered;

__constant__ float3 c_waterColor = {0.0f, 0.1f, 0.3f};
__constant__ float c_waterReflectivity = 0.08f;

#define c_camPos *((float3*)&c_cam)
#define c_cam_fo *(((float3*)&c_cam) + 1)
#define c_cam_ri *(((float3*)&c_cam) + 2)
#define c_cam_up *(((float3*)&c_cam) + 3)
#define c_sunDir *(((float3*)&c_cam) + 4)
#define c_time *(((float*)&c_cam) + 17)
#define c_jitterX *(((float*)&c_cam) + 18)
#define c_jitterY *(((float*)&c_cam) + 19)


// Function which returns the color of a pixel, first casting a primary ray, and depending on this result a shadow ray/ reflections.
    __device__ float3 computeColor(float x,
                                float y,
                                float distance,
                                float shadowValue,
                                hitInfo& hit,
                                const uint32_t* __restrict__ bits,
                                const unsigned char* __restrict__ csdf,
                                cudaTextureObject_t texObj,
                                const uchar4* __restrict__ GIdata)
    {
        // --- Primary ray direction ---
        float2 NDC = make_float2(x * 2.0f - 1.0f + c_jitterX, y * 2.0f - 1.0f + c_jitterY);
        float3 dir = normalize(c_cam_fo + NDC.x * c_cam_ri + NDC.y * c_cam_up);

        // --- Trace primary ray ---
        hit = trace(c_camPos, dir, distance, bits, csdf);

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
            float3 directLight = baseColor * diffuse * shadowValue;

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

__device__ inline float bilinear(half* buffer, size_t pitchInBytes, float x, float y, int fullWidth, int fullHeight)
{
    int halfWidth = fullWidth >> 1;
    int halfHeight = fullHeight >> 1;

    // Use floating point coordinates for accuracy
    float hx = x * (float)halfWidth;
    float hy = y * (float)halfHeight;

    int ix = floorf(hx);
    int iy = floorf(hy);

    float fx = hx - (float)ix;
    float fy = hy - (float)iy;

    int ix1 = min(ix + 1, halfWidth - 1);
    int iy1 = min(iy + 1, halfHeight - 1);

    // ROBUST PITCH-CORRECT ACCESS
    const half* row0 = (const half*)((const char*)buffer + iy * pitchInBytes);
    const half* row1 = (const half*)((const char*)buffer + iy1 * pitchInBytes);

    half s00 = row0[ix];
    half s10 = row0[ix1];
    half s01 = row1[ix];
    half s11 = row1[ix1];
    
    // Bilinear interpolation
    half s0 = __hadd(__hmul(s00, __float2half(1.0f - fx)), __hmul(s10, __float2half(fx)));
    half s1 = __hadd(__hmul(s01, __float2half(1.0f - fx)), __hmul(s11, __float2half(fx)));
    return __half2float(__hadd(__hmul(s0, __float2half(1.0f - fy)), __hmul(s1, __float2half(fy))));
}


__device__ inline float minDist(cudaTextureObject_t tex, float x, float y)
{
    float half_pixel_x = 1.0f / (float)640;
    float half_pixel_y = 1.0f / (float)400;

    // Find the bottom-left texel for the quad
    float u_low = floor(x * (float)640) / (float)640;
    float v_low = floor(y * (float)400) / (float)400;

    float dist1 = tex2D<float>(tex, u_low, v_low);
    float dist2 = tex2D<float>(tex, u_low + half_pixel_x, v_low);
    float dist3 = tex2D<float>(tex, u_low, v_low + half_pixel_y);
    float dist4 = tex2D<float>(tex, u_low + half_pixel_x, v_low + half_pixel_y);

    // Find the minimum of the four distances
    return fminf(fminf(dist1, dist2), fminf(dist3, dist4));
}

__global__ void renderKernel(
                            uchar4* framebuffer, 
                            half2* motionVectorBuffer, // Stays half2*
                            half* depthBuffer,          // Stays half*
                            cudaTextureObject_t halfDepthTex, 
                            cudaTextureObject_t shadowTex,

                            size_t fbPitchInBytes,
                            size_t mvPitchInBytes,
                            size_t depthPitchInBytes,

                            int width, 
                            int height, 

                            const uint32_t* __restrict__ bits,
                            const unsigned char* __restrict__ csdf,
                            uchar4* __restrict__ GIdata,
                            cudaTextureObject_t texObj) 
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    float2 motionVector = make_float2(0.0f, 0.0f);
    hitInfo hit;
    float x = (float)ix / (float)width;
    float y = (float)iy / (float)height;
  
    float dist = minDist(halfDepthTex, x, y);
    //tex2D<float>(halfDepthTex, x, y );
    float shadowValue = tex2D<float>(shadowTex, x , y);
    float final_depth = 1.0f;
    
    float3 col = computeColor(x, y, dist, shadowValue, hit, bits, csdf, texObj, GIdata);
    if (hit.hit) {
        float4 previousClipPos = mat_mul_vec(c_previousViewProjection_unjittered, make_float4(hit.pos.x, hit.pos.y, hit.pos.z, 1.0f));
        float4 currentClipPos = mat_mul_vec(c_currentViewProjection_unjittered, make_float4(hit.pos.x, hit.pos.y, hit.pos.z, 1.0f));        
        if (previousClipPos.w > 0.0f && currentClipPos.w > 0.0f) {
            float2 previousNDC = make_float2(previousClipPos.x / previousClipPos.w, previousClipPos.y / previousClipPos.w);
            float2 currentNDC = make_float2(currentClipPos.x / currentClipPos.w, currentClipPos.y / currentClipPos.w);
            
            motionVector = make_float2(currentNDC.x - previousNDC.x, currentNDC.y - previousNDC.y);
        }
        if (currentClipPos.w > 0.0f) {
            final_depth = currentClipPos.z / currentClipPos.w;
        }
    }
    col = clamp(col, make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f));
    uchar4 pixel = make_uchar4((unsigned char)(col.x * 255.0f), (unsigned char)(col.y * 255.0f), (unsigned char)(col.z * 255.0f), 255);

    *((uchar4*)((char*)framebuffer + iy * fbPitchInBytes) + ix) = pixel;
    *((half2*)((char*)motionVectorBuffer + iy * mvPitchInBytes) + ix) = __float22half2_rn(make_float2(motionVector.x, -motionVector.y));
    *((half*)((char*)depthBuffer + iy * depthPitchInBytes) + ix) = __float2half(final_depth);
}

__global__ void distApproximationKernel(cudaSurfaceObject_t distSurf, 
                                        cudaSurfaceObject_t shadowSurf,

                                        int width, 
                                        int height, 

                                        const uint32_t* __restrict__ bits,
                                        const unsigned char* __restrict__ csdf) 
{
    uint64_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    float x = ((float)ix + 0.5f) / (float)width;
    float y = ((float)iy + 0.5f) / (float)height;


    float2 NDC = make_float2(x * 2.0f - 1.0f + c_jitterX, y * 2.0f - 1.0f + c_jitterY);
    float3 dir = normalize(c_cam_fo + NDC.x * c_cam_ri + NDC.y * c_cam_up);

    hitInfo hit = trace(c_camPos, dir, (half)0.0f, bits, csdf);
    float dist = hit.hit ? distance(hit.pos, c_camPos) : 300;

    float shadowValue = 1.0f;
    if(hit.hit)
    {
        hitInfo shadow = trace(hit.pos + hit.normal * 1e-1f, c_sunDir, (half)0.0f, bits, csdf);
        shadowValue = shadow.hit ? (half)0.2f : (half)1.0f;
    }
    surf2Dwrite(dist - 8.0f, distSurf, ix * sizeof(float), iy); 
    surf2Dwrite(shadowValue, shadowSurf, ix * sizeof(float), iy);
}


void StateRender::drawCUDA(const glm::vec3& pos, const glm::vec3& fo,
                           const glm::vec3& up, const glm::vec3& ri, 
                           glm::mat4* unjitteredViewProjectionMatrix, 
                           glm::mat4* prevUnjitteredViewProjectionMatrix,
                           float jitterX, float jitterY) 
{    
    cudaMemcpyToSymbol(c_previousViewProjection_unjittered, prevUnjitteredViewProjectionMatrix, sizeof(glm::mat4));
    cudaMemcpyToSymbol(c_currentViewProjection_unjittered, unjitteredViewProjectionMatrix, sizeof(glm::mat4));

    float _currentTime = (float)(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() % 1000000)* 0.001f; 
    glm::vec3 _sunDir = glm::normalize(glm::vec3(10.f, 5.f, -4.f));

    float _c_cam[18] = {pos.x, pos.y, pos.z,
                        fo.x, fo.y, fo.z,
                        ri.x, ri.y, ri.z,
                        up.x, up.y, up.z,
                        _sunDir.x, _sunDir.y, _sunDir.z,
                        _currentTime, jitterX, jitterY};
    
    cudaMemcpyToSymbol(c_cam, _c_cam, sizeof(_c_cam));

    dim3 block(8, 8);
    dim3 grid((unsigned int)((lowResColorBuffer.getWidth() / 2) + block.x - 1) / block.x,
              ((unsigned int)(lowResColorBuffer.getHeight() / 2) + block.y - 1) / block.y);

    distApproximationKernel<<<grid, block>>>(
                            halfDistBuffer.getCudaSurfObject(),
                            shadowTex.getCudaSurfObject(),

                            lowResColorBuffer.getWidth() / 2,
                            lowResColorBuffer.getHeight() / 2,
                            cArray.getPtr(), 
                            csdf.getPtr());

     block = dim3(8, 8);
     grid = dim3((unsigned int)(lowResColorBuffer.getWidth() + block.x - 1) / block.x,
                (unsigned int)(lowResColorBuffer.getHeight() + block.y - 1) / block.y);

    renderKernel<<<grid, block>>>(
                              (uchar4*)lowResColorBuffer.GetCudaDevicePtr(),
                              (half2*)motionVectorTex.GetCudaDevicePtr(),
                              (half*)depthTex.GetCudaDevicePtr(),
                              halfDistBuffer.getCudaTexObject(),
                              shadowTex.getCudaTexObject(),

                              // FIX: Divide by the size of the element, not the pointer.
                              lowResColorBuffer.getPitch(), 
                              motionVectorTex.getPitch(),
                              depthTex.getPitch(),

                              lowResColorBuffer.getWidth(),
                              lowResColorBuffer.getHeight(),

                              cArray.getPtr(), 
                              csdf.getPtr(),
                              (uchar4*)GIdata.getPtr(),
                              texturepack.texObject() );
}


__host__ StateRender::StateRender() 
{
}