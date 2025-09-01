#include "StateRender.cuh"
#include "CArray.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cumath.cuh"
#include "csdf.cuh"

__constant__ float3 c_sunDir;
__constant__ float3 c_camPos, c_camFo, c_camUp, c_camRi;



struct hitInfo {
    bool hit;
    int its;
    float3 pos;
    float3 normal;
};



__device__ __forceinline__ bool IsSolid(int3 p, const uint32_t* __restrict__ bits) {
    int index = toIndex(p);
    return (bits[index >> 5] >> (index & 31)) & 1;
}

__device__ __forceinline__ int toCoarseIndex(float3 pos) {
    int cx = floorf(pos.x) / COARSENESS;
    int cy = floorf(pos.y) / COARSENESS;
    int cz = floorf(pos.z) / COARSENESS;
    return cz * C_SIZEX * C_SIZEY + cy * C_SIZEX + cx;
}

__device__ __forceinline__ float getDistance(float3 pos, const unsigned char* __restrict__ csdf)
{
    int cx = (int)(floorf(pos.x) * (1.0f / COARSENESS));
    int cy = (int)(floorf(pos.y) * (1.0f / COARSENESS));
    int cz = (int)(floorf(pos.z) * (1.0f / COARSENESS));
    
    int cidx = cz * C_SIZEX * C_SIZEY + cy * C_SIZEX + cx;
    unsigned char dist = csdf[cidx];
    return (float)dist;
}
__device__ __forceinline__ unsigned char getDistance(int3 pos, const unsigned char* __restrict__ csdf)
{
    int cx = pos.x / COARSENESS;
    int cy = pos.y / COARSENESS;
    int cz = pos.z / COARSENESS;
    
    int cidx = cz * C_SIZEX * C_SIZEY + cy * C_SIZEX + cx;
    return csdf[cidx];
}

__device__ float3 approximateCSDF(float3 pos, float3 dir, const unsigned char* __restrict__ csdf)
{
    int iterations = 0;
    while (iterations < 100) 
    {
        if (pos.x < 0 || pos.y < 0 || pos.z < 0 ||
            pos.x >= SIZEX || pos.y >= SIZEY || pos.z >= SIZEZ) return make_float3(-100.0f, -100.0f, -100.0f);

        float dist = getDistance(pos, csdf);
        
        if (dist < 2) 
            return pos;
        
        pos = pos + dir * max(2.0f, dist);
        iterations++;
    }
    return pos;
}

__device__ hitInfo trace(float3 camPos, float3 camDir, 
                        const uint32_t* __restrict__ bits,
                        const unsigned char* __restrict__ csdf) 
{
    float3 currentPos = camPos;
    hitInfo HI; HI.hit = false;
    HI.its = 0.0f;
    
    bool jumped;
    for (int majorIteration = 0; majorIteration < 5; majorIteration++) 
    {
        HI.its++;
        // Phase 1: CSDF approximation to get close to surfaces
        currentPos = approximateCSDF(currentPos, camDir, csdf);
        
        // Phase 2: DDA for precise intersection
        int3 ipos = make_int3(floorf(currentPos.x), floorf(currentPos.y), floorf(currentPos.z));
        int3 step = make_int3((camDir.x > 0) - (camDir.x < 0),
                             (camDir.y > 0) - (camDir.y < 0),
                             (camDir.z > 0) - (camDir.z < 0));
        
        float3 deltaDist = make_float3(
            (camDir.x != 0) ? fabsf(1.0f / camDir.x) : 1e10f,
            (camDir.y != 0) ? fabsf(1.0f / camDir.y) : 1e10f,
            (camDir.z != 0) ? fabsf(1.0f / camDir.z) : 1e10f
        );
        
        float3 tMax = make_float3(
            ((step.x > 0) ? (ipos.x + 1.0f - currentPos.x) : (currentPos.x - ipos.x)) * deltaDist.x,
            ((step.y > 0) ? (ipos.y + 1.0f - currentPos.y) : (currentPos.y - ipos.y)) * deltaDist.y,
            ((step.z > 0) ? (ipos.z + 1.0f - currentPos.z) : (currentPos.z - ipos.z)) * deltaDist.z
        );
        
        
        char mask = -128;
        for (int i = 0; i < 200; i++) {
            HI.its++;
            // Check if we're in empty space far from any surface
            if ((i & 7) == 7) {
                unsigned char dist = getDistance(ipos, csdf);
                if (dist > 2) { // Use a smaller threshold
                    // Jump ahead using CSDF
                    currentPos = make_float3(ipos.x, ipos.y, ipos.z) + camDir * dist * COARSENESS;
                    jumped = true;
                    break; // Break out of DDA loop to restart with CSDF
                }
            }
            
            if (ipos.x < 0 || ipos.y < 0 || ipos.z < 0 ||
                ipos.x >= SIZEX || ipos.y >= SIZEY || ipos.z >= SIZEZ) {
                return HI; // Out of bounds
            }
            
            if (IsSolid(ipos, bits)) {
                HI.hit = true;
                HI.normal = make_float3((mask & 1) ? -1.0f : 1.0f,
                                       ((mask >> 1) & 1) ? -1.0f : 1.0f,
                                       ((mask >> 2) & 1) ? -1.0f : 1.0f);
                HI.pos = make_float3(ipos.x, ipos.y, ipos.z);
                return HI;
            }
            
            // DDA step
            if (tMax.x < tMax.y) {
                if (tMax.x < tMax.z) { 
                    tMax.x += deltaDist.x; 
                    ipos.x += step.x; 
                    mask = 0; 
                } else { 
                    tMax.z += deltaDist.z; 
                    ipos.z += step.z; 
                    mask = 2; 
                }
            } else {
                if (tMax.y < tMax.z) { 
                    tMax.y += deltaDist.y; 
                    ipos.y += step.y; 
                    mask = 1; 
                } else { 
                    tMax.z += deltaDist.z; 
                    ipos.z += step.z; 
                    mask = 2; 
                }
            }

        }
    
        // If we completed the DDA loop without hitting anything and didn't jump ahead,
        // we're done
        if(jumped) continue;
        
        if (!HI.hit) break;
    }
    
    return HI;
}

__device__ float3 computeColor(float x, 
                              float y, 
                              const uint32_t* __restrict__ bits, 
                              const unsigned char* __restrict__ csdf) 
{
    float2 NDC = make_float2(x * 2.0f - 1.0f, y * 2.0f - 1.0f);
    float3 dir = normalize(c_camFo + NDC.x * c_camRi + NDC.y * c_camUp);

    hitInfo hit = trace(c_camPos, dir, bits, csdf);

    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    
    if (hit.hit) {
        float diff = fmaxf(dot(hit.normal, c_sunDir), 0.0f);
        color = make_float3(diff, diff, diff);
    }
    else
    {
        color = -dir;
    }
    // if(hit.its > 50.0f)
    // {
    //     color.x = 0.5f - color.x;
    // }
    return color;
}

__global__ void renderKernel(uchar4* framebuffer, 
                             int width, 
                             int height, 
                             const uint32_t* __restrict__ bits,
                             const unsigned char* __restrict__ csdf) 
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    float x = (float)ix / (float)width;
    float y = (float)iy / (float)height;

    float3 col = computeColor(x, y, bits, csdf);
    unsigned char r = (unsigned char)(col.x * 255.0f);
    unsigned char g = (unsigned char)(col.y * 255.0f);
    unsigned char b = (unsigned char)(col.z * 255.0f);

    framebuffer[ix + iy * width] = make_uchar4(b, g, r, 255);
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
    dim3 grid((framebuffer.getWidth() + block.x - 1) / block.x,
                (framebuffer.getHeight() + block.y - 1) / block.y);

        renderKernel<<<grid, block>>>(
    reinterpret_cast<uchar4*>(framebuffer.devicePtr()),
                              framebuffer.getWidth(),
                              framebuffer.getHeight(),
                              cArray.getPtr(), 
                              csdf.getPtr()
    );
}


__host__ StateRender::StateRender() 
{
}







// #include "StateRender.cuh"
// #include "CArray.cuh"
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include "cumath.cuh"
// #include "csdf.cuh"

// __constant__ float3 c_sunDir;
// __constant__ float3 c_camPos, c_camFo, c_camUp, c_camRi;



// struct hitInfo {
//     bool hit;
//     float3 pos;
//     float3 normal;
// };


// __device__ __forceinline__ float sampleCSDFDistanceAtPos(const float3 pos, const unsigned char* __restrict__ csdf)
// {
//     // map fine coords to coarse cell indices
//     int cx = (int)floorf(pos.x) / COARSENESS;
//     int cy = (int)floorf(pos.y) / COARSENESS;
//     int cz = (int)floorf(pos.z) / COARSENESS;

//     // clamp to coarse bounds
//     if (cx < 0) cx = 0;
//     if (cy < 0) cy = 0;
//     if (cz < 0) cz = 0;
//     if (cx >= C_SIZEX) cx = C_SIZEX - 1;
//     if (cy >= C_SIZEY) cy = C_SIZEY - 1;
//     if (cz >= C_SIZEZ) cz = C_SIZEZ - 1;

//     int cidx = cz * (C_SIZEX * C_SIZEY) + cy * C_SIZEX + cx;
//     unsigned char cd = csdf[cidx]; // number of coarse voxels to nearest solid (0..MAX_DIST)
//     // convert to fine voxels distance:
//     float d_fine = (float)cd * (float)COARSENESS;
//     // if cd == MAX_DIST means "no solid found within radius" — still return large bound
//     float maxPossible = (float)(MAX_DIST * COARSENESS);
//     if (d_fine <= 0.0f) return 0.0f;
//     if (d_fine > maxPossible) return maxPossible;
//     return d_fine;
// }


// __device__ __forceinline__ int toIndex(int3 p) 
// {
//     return  ((p.x - (SIZEX>>1)) & MODX) | 
//            (((p.y - (SIZEY>>1)) & MODY) << SHIX) | 
//            (((p.z - (SIZEZ>>1)) & MODZ) << (SHIX + SHIY));
// }

// __device__ __forceinline__ bool IsSolid(int3 p, const uint32_t* __restrict__ bits) {
//     int index = toIndex(p);
//     return (bits[index >> 5] >> (index & 31)) & 1;
// }

// __device__ hitInfo trace(float3 pos, 
//                          float3 dir, 
//                          const uint32_t* __restrict__ bits,
//                          const unsigned char* __restrict__ csdf) 
// {
//     const int COARSE_STEPS_LIMIT = 128;      // max number of coarse steps before falling back
//     const float COARSE_EPS = 1.0f;           // if coarse distance <= COARSE_EPS*COARSENESS -> do DDA

//     // 1) coarse stepping (sphere-trace like) using csdf
//     float3 p = pos;
//     float traveled = 0.0f;
//     hitInfo HI; HI.hit = false;

//     // Bound world extents to avoid infinite stepping
//     const float worldMinX = 0.0f, worldMinY = 0.0f, worldMinZ = 0.0f;
//     const float worldMaxX = (float)SIZEX;
//     const float worldMaxY = (float)SIZEY;
//     const float worldMaxZ = (float)SIZEZ;

//     for (int step = 0; step < COARSE_STEPS_LIMIT; ++step) {
//         // Check bounds quickly
//         if (p.x < worldMinX || p.y < worldMinY || p.z < worldMinZ ||
//             p.x >= worldMaxX || p.y >= worldMaxY || p.z >= worldMaxZ) break;

//         float d_coarse = sampleCSDFDistanceAtPos(p, csdf); // returns distance in fine voxels
//         if (d_coarse <= COARSE_EPS * (float)COARSENESS) {
//             // Coarse field says we're close — break and refine via DDA
//             break;
//         }
//         // step by d_coarse minus tiny epsilon to avoid overshoot
//         float stepDist = d_coarse - 0.5f;
//         if (stepDist < 0.5f) stepDist = 0.5f; // minimal step to make progress
//         p.x += dir.x * stepDist;
//         p.y += dir.y * stepDist;
//         p.z += dir.z * stepDist;
//         traveled += stepDist;

//         // safety stop: if traveled out of the possible world extents big time
//         if (p.x < 0 || p.y < 0 || p.z < 0 ||
//             p.x >= SIZEX || p.y >= SIZEY || p.z >= SIZEZ || traveled > 1e6f) {
//             break;
//         }
//     }



//     int3 ipos = make_int3((int)floorf(p.x), (int)floorf(p.y), (int)floorf(p.z));
//     int3 step = make_int3((dir.x > 0) - (dir.x < 0),
//                           (dir.y > 0) - (dir.y < 0),
//                           (dir.z > 0) - (dir.z < 0));

//     float3 deltaDist = make_float3(fabsf(1.0f / dir.x), fabsf(1.0f / dir.y), fabsf(1.0f / dir.z));
//     float3 tMax = make_float3(
//         ((step.x > 0 ? ipos.x + 1.0f - p.x : p.x - ipos.x)) * deltaDist.x,
//         ((step.y > 0 ? ipos.y + 1.0f - p.y : p.y - ipos.y)) * deltaDist.y,
//         ((step.z > 0 ? ipos.z + 1.0f - p.z : p.z - ipos.z)) * deltaDist.z
//     );

//     char mask = -128;
//     hitInfo HI; HI.hit = false;
//     for (int i = 0; i < 4000; i++) {

//         if (ipos.x < 0 || ipos.y < 0 || ipos.z < 0 ||
//             ipos.x >= SIZEX || ipos.y >= SIZEY || ipos.z >= SIZEZ)
//             break;

//         if (IsSolid(ipos, bits)) {
//             HI.hit = true;
//             HI.normal = make_float3(mask & 1 ? 1.0f : 0.0f,
//                                     (mask >> 1) & 1 ? 1.0f : 0.0f,
//                                     (mask >> 2) & 1 ? 1.0f : 0.0f);
//             HI.pos = make_float3(ipos.x, ipos.y, ipos.z);
//             return HI;
//         }

//         if (tMax.x < tMax.y) {
//             if (tMax.x < tMax.z) { tMax.x += deltaDist.x; ipos.x += step.x; mask = 0; }
//             else { tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; }
//         } else {
//             if (tMax.y < tMax.z) { tMax.y += deltaDist.y; ipos.y += step.y; mask = 1; }
//             else { tMax.z += deltaDist.z; ipos.z += step.z; mask = 2; }
//         }
//     }
//     return HI;

// }

// __device__ float3 computeColor(float x, 
//                               float y, 
//                               const uint32_t* __restrict__ bits, 
//                               const unsigned char* __restrict__ csdf) 
// {
//     float2 NDC = make_float2(x * 2.0f - 1.0f, y * 2.0f - 1.0f);
//     float3 dir = normalize(c_camFo + NDC.x * c_camRi + NDC.y * c_camUp);

//     hitInfo hit = trace(c_camPos, dir, bits, csdf);

//     if (hit.hit) {
//         float diff = fmaxf(dot(hit.normal, c_sunDir), 0.0f);
//         return make_float3(diff, diff, diff);
//     }
//     return -dir;
// }

// __global__ void renderKernel(uchar4* framebuffer, 
//                              int width, 
//                              int height, 
//                              const uint32_t* __restrict__ bits,
//                             const unsigned char* __restrict__ csdf) 
// {
//     int ix = blockIdx.x * blockDim.x + threadIdx.x;
//     int iy = blockIdx.y * blockDim.y + threadIdx.y;
//     if (ix >= width || iy >= height) return;

//     float x = (float)ix / (float)width;
//     float y = (float)iy / (float)height;

//     float3 col = computeColor(x, y, bits, csdf);
//     unsigned char r = (unsigned char)(col.x * 255.0f);
//     unsigned char g = (unsigned char)(col.y * 255.0f);
//     unsigned char b = (unsigned char)(col.z * 255.0f);

//     framebuffer[ix + iy * width] = make_uchar4(b, g, r, 255);
// }

// void StateRender::drawCUDA(const glm::vec3& pos, const glm::vec3& fo,
//                         const glm::vec3& up, const glm::vec3& ri) 
// {
//     // Upload camera + sun constants
//     cudaMemcpyToSymbol(c_camPos, &pos, sizeof(glm::vec3));
//     cudaMemcpyToSymbol(c_camFo, &fo, sizeof(glm::vec3));
//     cudaMemcpyToSymbol(c_camUp, &up, sizeof(glm::vec3));
//     cudaMemcpyToSymbol(c_camRi, &ri, sizeof(glm::vec3));
//     glm::vec3 sunDir = glm::normalize(glm::vec3(10.f, 5.f, -4.f));
//     cudaMemcpyToSymbol(c_sunDir, &sunDir, sizeof(glm::vec3));


//     dim3 block(16, 16);
//     dim3 grid((framebuffer.getWidth() + block.x - 1) / block.x,
//                 (framebuffer.getHeight() + block.y - 1) / block.y);

//         renderKernel<<<grid, block>>>(
//     reinterpret_cast<uchar4*>(framebuffer.devicePtr()),
//                               framebuffer.getWidth(),
//                               framebuffer.getHeight(),
//                               cArray.getPtr(), 
//                               csdf.getPtr()
//     );
// }


// __host__ StateRender::StateRender() 
// {
// }




//     // #include "StateRender.hpp"
//     // #include <algorithm>
//     // #include <glm/gtc/type_ptr.hpp>
//     // #include <glm/gtx/norm.hpp>
//     // #include <iostream>
//     // #include <thread>
//     // #include <vector>

//     // const glm::vec3 sunDir = glm::normalize(glm::vec3(10.f, 5.f, -4.f));

//     // StateRender::StateRender() {
//     //     // Allocate zero-initialized memory for the bits array.
//     //     bits = new uint32_t[CArray::NUM_UINTS]();
//     // }

//     // StateRender::~StateRender() {
//     //     delete[] bits;
//     //     bits = nullptr;
//     // }

//     // void StateRender::drawPixel(int ix, int iy) {
//     //     float x = (float)ix / (float)State::dispWIDTH;
//     //     float y = (float)iy / (float)State::dispHEIGHT;

//     //     glm::vec3 color = computeColor(x, y);
//     //     char r = static_cast<char>(color.r * 255.0f);
//     //     char g = static_cast<char>(color.g * 255.0f);
//     //     char b = static_cast<char>(color.b * 255.0f);

//     //     State::state.bmp[(ix + iy * State::dispWIDTH) * 4 + 2] = r;
//     //     State::state.bmp[(ix + iy * State::dispWIDTH) * 4 + 1] = g;
//     //     State::state.bmp[(ix + iy * State::dispWIDTH) * 4 + 0] = b;
//     // }

//     // void StateRender::drawRow(int row) {
//     //     for(int i = 0; i < State::dispWIDTH; i++)
//     //         drawPixel(i, row);
//     // }

//     // void StateRender::draw() {
//     //     std::vector<std::thread> threads;
//     //     for(int i = 0; i < State::dispHEIGHT; i++)
//     //         threads.push_back(std::thread(&StateRender::drawRow, this, i));

//     //     for(auto &t : threads)
//     //         if(t.joinable())
//     //             t.join();
//     // }

//     // hitInfo StateRender::trace(glm::vec3 pos, glm::vec3 dir) {
//     //     ivec3 ipos = (ivec3)(pos);
//     // 	ivec3 step = (ivec3)sign(dir);
//     // 	vec3 deltaDist = abs(1.0f / dir);
//     // 	//vec3 tMax = (1.0f - glm::fract(pos)) * deltaDist; 
//     // 	//vec3 tMax = abs((pos - floor(pos) + sign(dir)) * deltaDist);
//     // 	//vec3 tMax = (sign(dir) * fract(pos)) + ((sign(dir) * 0.5f) + 0.5f) * deltaDist; 

//     // 	// vec3 tMax = (sign(dir) * (vec3(ipos) - pos) + (sign(dir) * 0.5f) + 0.5f) * deltaDist; 
//     // 	vec3 tMax = (sign(dir) * (vec3(ipos) - pos) + (sign(dir) * 0.5f) + 0.5f) * deltaDist; 

//     // 	char mask = -128;

//     // 	hitInfo HI = hitInfo();
//     // 	for(int i = 0; i < 4000; i++)
//     // 	{
//     // 		if(glm::any(glm::lessThan(ipos, (ivec3)0)) || glm::any(glm::greaterThanEqual(ipos, (ivec3)256)))
//     // 			break;
//     // 		if(this->IsSolid(ipos)) 
//     // 		{
//     // 			HI.hit = true;
//     // 			HI.normal = vec3(mask & 1, 
//     // 			    	        (mask >> 1) & 1,
//     // 					         mask >> 2);	
//     // 			HI.pos = ipos;	
//     // 			return HI;
//     // 		}
//     // 		if(tMax.x < tMax.y) {
//     // 			if(tMax.x < tMax.z) {
//     // 				tMax.x += deltaDist.x;
//     // 				ipos.x += step.x;
//     // 				mask = 0;
//     // 			}
//     // 			else {
//     // 				tMax.z += deltaDist.z;
//     // 				ipos.z += step.z;	
//     // 				mask = 2;
//     // 			}		
//     // 		}
//     // 		else {
//     // 			if(tMax.y < tMax.z) {
//     // 				tMax.y += deltaDist.y;
//     // 				ipos.y += step.y;
//     // 				mask = 1;
//     // 			}
//     // 			else {
//     // 				tMax.z += deltaDist.z;
//     // 				ipos.z += step.z;
//     // 				mask = 2;
//     // 			}
//     // 		}
//     // 	}
//     // 	HI.hit = false;
        
//     // 		// mask = sideDist.x < sideDist.y ? (sideDist.y < sideDist.z ? vec3(1, 0, 0) : vec3(0, 1, 0) ) : vec3(0, 0, 1);//glm::step(sideDist.xyz, sideDist.yzx) * glm::(sideDist.xyz, sideDist.zxy);
//     // 		// sideDist += mask * deltaDist;
//     // 		// ipos += mask * step;
//     //     // for(int i = 0; i < 600; i++)
//     //     // {
//     //     //     if(IsSolid(pos - 125.f))
//     //     //         return true;
//     //     //     pos += dir;
//     //     // }
//     //     return HI;
//     // }

//     // glm::vec3 StateRender::computeColor(float x, float y) {
//     //     glm::vec3 fo = State::state.character.camera.forward;
//     //     glm::vec3 up = State::state.character.camera.up;
//     //     glm::vec3 ri = State::state.character.camera.right;
//     //     glm::vec3 pos = State::state.character.camera.pos;

//     //     glm::vec2 NDC = glm::vec2(x * 2.0f - 1.0f, y * 2.0f - 1.0f);

//     //     glm::vec3 dir = glm::normalize(fo + NDC.x * ri + NDC.y * up);
//     //     hitInfo hit = trace(pos, dir);

//     //     if(hit.hit) {
//     //         float diff = std::max(glm::dot(hit.normal, sunDir), 0.0f);
//     //         return -glm::vec3(diff);
//     //     }

//     //     return -dir;
//     // }


//     // bool StateRender::IsSolid(glm::ivec3 pos) {
//     //     glm::ivec3 p = (pos - 128) & 255;
//     //     int index = p.x | (p.y << 8) | (p.z << 16);
//     //     // Check the corresponding bit in the uint32_t array

//     //     return (bits[index >> 5] >> (index & 31)) & 1;
//     // }

//     // bool StateRender::IsSolid(glm::vec3 pos) {
//     //     return IsSolid((glm::ivec3)pos);
//     // }
