#include "CoarseArray.cuh"
#include "cumath.cuh"
#include <iostream>
#include "raytracing_functions.cuh"
#include "glm/glm.hpp"

#define BOUNCE_STRENGTH 0.9f // How much light is transferred from neighbors.
#define NUM_BOUNCE_SAMPLES 6 // We'll sample the 6 direct neighbors.

// Helper function to check if any bit is set within a coarse block.
__device__ bool isCoarseBlockSolid(uint64_t cx, uint64_t cy, uint64_t cz, const uint32_t* fineData) 
{
    for (uint64_t z = 0; z < COARSENESSSDF; ++z) {
        for (uint64_t y = 0; y < COARSENESSSDF; ++y) {
            for (uint64_t x = 0; x < COARSENESSSDF; ++x) {
                uint64_t fine_x = cx * COARSENESSSDF + x;
                uint64_t fine_y = cy * COARSENESSSDF + y;
                uint64_t fine_z = cz * COARSENESSSDF + z;

                // Simple boundary check
                if (fine_x >= SIZEX || fine_y >= SIZEY || fine_z >= SIZEZ) continue;

                uint64_t index = toIndex(fine_x, fine_y, fine_z);
                //uint64_t index = (uint64_t)(fine_z * SIZEY + fine_y) * SIZEX + fine_x;
                if ((fineData[index >> 5] >> (index & 31)) & 1) {
                    return true; // Found a solid voxel in this block
                }
            }
        }
    }
    return false;
}

// Kernel 1: Initializes the grid and computes distance along the X-axis up to MAX_DIST.
// Note: This is a local scan. It's simpler but less efficient than a true separable pass
// for large radii, but effective for smaller radii like 8.
__global__ void computeDistX(const uint32_t* fineData, unsigned char* distX) 
{
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (idx >= SDF_BYTESIZE) return;

    uint64_t cz = idx / (SDF_SIZEX * SDF_SIZEY);
    uint64_t temp = idx % (SDF_SIZEX * SDF_SIZEY);
    uint64_t cy = temp / SDF_SIZEX;
    uint64_t cx = temp % SDF_SIZEX;

    if (isCoarseBlockSolid(cx, cy, cz, fineData)) 
    {
        distX[idx] = 0;
        return;
    }

    unsigned char min_d = SDF_MAX_DIST;

    // Scan left up to MAX_DIST
    for (uint64_t i = 1; i <= SDF_MAX_DIST; ++i) 
    {
    	if (i <= cx && isCoarseBlockSolid(cx - i, cy, cz, fineData)) 
        {
            min_d = i;
            break;
        }
    }

    // Scan right, but only up to the current minimum distance found
    for (uint64_t i = 1; i < min_d; ++i) 
    {
        if (cx + i < SDF_SIZEX && isCoarseBlockSolid(cx + i, cy, cz, fineData)) 
        {
            min_d = i;
            break;
        }
    }
    distX[idx] = min_d;
}


// Kernel 2: Takes X distances and computes the 2D distance in the XY plane.
__global__ void computeDistY(const unsigned char* distX, unsigned char* distY) 
{
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (idx >= SDF_BYTESIZE) return;

    unsigned char current_dx = distX[idx];
    if (current_dx == 0) {
        distY[idx] = 0;
        return;
    }

    uint64_t cz = idx / (SDF_SIZEX * SDF_SIZEY);
    uint64_t temp = idx % (SDF_SIZEX * SDF_SIZEY);
    uint64_t cy = temp / SDF_SIZEX;

    float min_dist_sq = (float)current_dx * (float)current_dx;

    // Scan along the Y-axis up to MAX_DIST
    for (uint64_t y_offset = 1; y_offset <= SDF_MAX_DIST; ++y_offset) {
        // Early exit optimization: if y^2 is already > min_dist_sq, no closer point can be found on this axis.
        if (y_offset * y_offset >= min_dist_sq) break;

        // Check upwards
        if (cy - y_offset >= 0) {
            uint64_t neighbor_idx = idx - y_offset * SDF_SIZEX;
            float dist_sq = (float)distX[neighbor_idx] * distX[neighbor_idx] + (float)y_offset * y_offset;
            min_dist_sq = fminf(min_dist_sq, dist_sq);
        }
        // Check downwards
        if (cy + y_offset < SDF_SIZEY) {
            uint64_t neighbor_idx = idx + y_offset * SDF_SIZEX;
            float dist_sq = (float)distX[neighbor_idx] * distX[neighbor_idx] + (float)y_offset * y_offset;
            min_dist_sq = fminf(min_dist_sq, dist_sq);
        }
    }
    distY[idx] = (unsigned char)fminf((float)SDF_MAX_DIST, sqrtf(min_dist_sq));
}

// Kernel 3: Takes XY distances and computes the final 3D distance.
__global__ void computeDistZ(const unsigned char* distXY, unsigned char* finalCSDF) 
{
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (idx >= SDF_BYTESIZE) return;

    unsigned char current_dxy = distXY[idx];
    if (current_dxy == 0) {
        finalCSDF[idx] = 0;
        return;
    }
    
    uint64_t cz = idx / (SDF_SIZEX * SDF_SIZEY);

    float min_dist_sq = (float)current_dxy * current_dxy;
    
    // Scan along the Z-axis up to MAX_DIST
    for (uint64_t z_offset = 1; z_offset <= SDF_MAX_DIST; ++z_offset) {
        // Early exit optimization
        if (z_offset * z_offset >= min_dist_sq) break;

        // Check backwards
        if (cz - z_offset >= 0) {
            uint64_t neighbor_idx = idx - z_offset * (SDF_SIZEX * SDF_SIZEY);
            float dist_sq = (float)distXY[neighbor_idx] * distXY[neighbor_idx] + (float)z_offset * z_offset;
            min_dist_sq = fminf(min_dist_sq, dist_sq);
        }
        // Check forwards
        if (cz + z_offset < SDF_SIZEZ) {
            uint64_t neighbor_idx = idx + z_offset * (SDF_SIZEX * SDF_SIZEY);
            float dist_sq = (float)distXY[neighbor_idx] * distXY[neighbor_idx] + (float)z_offset * z_offset;
            min_dist_sq = fminf(min_dist_sq, dist_sq);
        }
    }
    finalCSDF[idx] = (unsigned char)fminf((float)SDF_MAX_DIST, sqrtf(min_dist_sq));
}


CoarseArray::CoarseArray() {}
CoarseArray::~CoarseArray() {}

void CoarseArray::AllocateSDF() {
    m_csdfArray.Allocate(SDF_BYTESIZE);
}
void CoarseArray::AllocateGI() {
    m_csdfArray.Allocate(GI_BYTESIZE);
}
// void CoarseArray::Allocate(const int byteSize) {
//     m_csdfArray.Allocate(byteSize);
// }

unsigned char* CoarseArray::getPtr() 
{
    return reinterpret_cast<unsigned char*>(m_csdfArray.getPtr());
}

void CoarseArray::GenerateSDF(CArray& fineArray) 
{
    if (m_csdfArray.getSize() != SDF_BYTESIZE) 
    {
        std::cerr << "CSDF not allocated or wrong size. Call Allocate() first." << std::endl;
        return;
    }
    // Create a temporary CArray for intermediate calculations.
    CArray tempArray;
    tempArray.Allocate(SDF_BYTESIZE);

    // Common launch configuration for the coarse grid
    const unsigned long threads = 256;
    unsigned int blocks = (unsigned int)((SDF_BYTESIZE + (uint64_t)threads - 1ull) / (uint64_t)threads);
    
    // --- Pass 1: X-axis distance ---
    computeDistX<<<blocks, threads>>>(fineArray.getPtr(), reinterpret_cast<unsigned char*>(tempArray.getPtr()));
    CUDA_CHECK(cudaGetLastError());
    
    // --- Pass 2: Y-axis distance (creates 2D distance) ---
    computeDistY<<<blocks, threads>>>(reinterpret_cast<unsigned char*>(tempArray.getPtr()), getPtr());
    CUDA_CHECK(cudaGetLastError());
    
    // --- Pass 3: Z-axis distance (creates final 3D distance) ---
    computeDistZ<<<blocks, threads>>>(getPtr(), reinterpret_cast<unsigned char*>(tempArray.getPtr()));
    CUDA_CHECK(cudaGetLastError());

    // The final result is now in tempArray. We need to copy it back to our member array.
    CUDA_CHECK(cudaMemcpy(m_csdfArray.getPtr(), tempArray.getPtr(), SDF_BYTESIZE, cudaMemcpyDeviceToDevice));
    
    // Free the temporary buffer
    tempArray.Free();

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "CSDF Generation Complete." << std::endl;
}

__constant__ float3 c_sunDir2;
__global__ void InitialGlobalIlluminate(uchar4* GIdata,
                                        const uint32_t* __restrict__ bits,
                                        const unsigned char* __restrict__ csdf)
{
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (idx >= GI_BYTESIZE) return;

    // Calculate the world position of this GI voxel
    uint64_t cz = idx / (GI_SIZEX * GI_SIZEY);
    uint64_t temp = idx % (GI_SIZEX * GI_SIZEY);
    uint64_t cy = temp / GI_SIZEX;
    uint64_t cx = temp % GI_SIZEX;

    // The world position is the center of the coarse voxel
    float3 worldPos = make_float3((cx + 0.5f) * COARSENESSGI,
                                 (cy + 0.5f) * COARSENESSGI,
                                 (cz + 0.5f) * COARSENESSGI);

    float3 accumulatedColor = make_float3(0.0f, 0.0f, 0.0f);
    hitInfo shadowHit = trace(worldPos, c_sunDir2, 0.0001f, bits, csdf);

    if (!shadowHit.hit) {
        // If we didn't hit anything, this voxel is lit by the sun
        accumulatedColor = c_sunColor;
    }

    // For now, we'll just store the direct light.
    // In a more advanced implementation, you would add bounced light here.

    // Convert the float color to uchar4 and write to the grid
    GIdata[idx] = make_uchar4(accumulatedColor.x * 255,
                                accumulatedColor.y * 255,
                                accumulatedColor.z * 255,
                                255); // Alpha can be used for opacity if needed
}



__device__ unsigned int random_state;
__device__ void init_random_state(int thread_id, int frame_number) {
    // Initialize with a value that is unique for each thread and changes every frame
    random_state = thread_id + frame_number * 198491317;
}

__device__ float random_float() {
    // XOR shift algorithm
    random_state ^= (random_state << 13);
    random_state ^= (random_state >> 17);
    random_state ^= (random_state << 5);
    // Convert to a float in [0, 1]
    return float(random_state) / float(4294967295.0f);
}
__device__ float3 random_direction_in_sphere() {
    float3 p;
    do {
        p = make_float3(random_float() * 2.0f - 1.0f,
                        random_float() * 2.0f - 1.0f,
                        random_float() * 2.0f - 1.0f);
    } while (dot(p, p) >= 1.0f); // Reject points outside the unit sphere
    return normalize(p);
}

__global__ void GlobalIlluminate(uchar4* GIdata_curr,
                                 const uint32_t* __restrict__ bits,
                                 const unsigned char* __restrict__ csdf,
                                 cudaTextureObject_t texturepack,
                                 unsigned int frameNumber,
                                 uint64_t offset) // NEW: for random seeding
{
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x + offset;
    if (idx >= GI_BYTESIZE) return;

        // --- 1. Setup ---
    init_random_state(idx, frameNumber);

    uint64_t cz = idx / (GI_SIZEX * GI_SIZEY);
    uint64_t temp = idx % (GI_SIZEX * GI_SIZEY);
    uint64_t cy = temp / GI_SIZEX;
    uint64_t cx = temp % GI_SIZEX;

    float3 worldPos = make_float3((cx + 0.5f) * COARSENESSGI,
                                 (cy + 0.5f) * COARSENESSGI,
                                 (cz + 0.5f) * COARSENESSGI);

    // If this GI cell is inside a solid block, it should be black.
    int3 voxel_ipos = make_int3(floorf(worldPos.x), floorf(worldPos.y), floorf(worldPos.z));
    if (IsSolid(voxel_ipos, bits)) {
        //GIdata_curr[idx] = make_uchar4(0, 0, 0, 255);
        return;
    }

    // --- 2. Calculate a new color sample for this frame ---
    float3 newSample = make_float3(0.0f, 0.0f, 0.0f);

    // Contribution from Direct Light (Sun)
    hitInfo shadowHit = trace(worldPos, c_sunDir2, 0.001f, bits, csdf);
    if (!shadowHit.hit) {
        newSample += c_sunColor;
    }

    // Contribution from Indirect Light (1 random bounce)
    float3 randomDir = random_direction_in_sphere();
    hitInfo bounceHit = trace(worldPos, randomDir, 0.001f, bits, csdf);

    if (bounceHit.hit) {
        // We hit a surface. Sample the GI data from the *previous* frame at that location.
        float3 hitPos = bounceHit.pos;
        int gx = (int)(floorf(hitPos.x) / COARSENESSGI);
        int gy = (int)(floorf(hitPos.y) / COARSENESSGI);
        int gz = (int)(floorf(hitPos.z) / COARSENESSGI);

        if (gx >= 0 && gx < GI_SIZEX && gy >= 0 && gy < GI_SIZEY && gz >= 0 && gz < GI_SIZEZ) {
            uint64_t hit_idx = (uint64_t)gz * GI_SIZEX * GI_SIZEY + (uint64_t)gy * GI_SIZEX + gx;
            uchar4 prevSample = GIdata_curr[hit_idx];
            float3 bouncedColor = make_float3(prevSample.x / 255.0f, prevSample.y / 255.0f, prevSample.z / 255.0f);

            // Modulate by surface albedo (texture color) for color bleeding
            // NOTE: Assumes sampleTexture exists from previous context
            float3 surfaceAlbedo = sampleTexture(bounceHit.uv, bounceHit.pos, texturepack);
            newSample += bouncedColor * surfaceAlbedo;
        }
    } else {
        // We didn't hit anything, so we hit the sky.
        // NOTE: Assumes sampleSky exists from previous context
        newSample += sampleSky(randomDir, c_sunDir2);
    }

    // --- 3. Blend with the previous frame's result to converge ---
    const float LEARNING_RATE = 0.04f; // How much the new sample influences the result. Lower = smoother convergence.
    uchar4 prevData = GIdata_curr[idx];
    float3 previousColor = make_float3(prevData.x / 255.0f, prevData.y / 255.0f, prevData.z / 255.0f);

    // Linearly interpolate between the old color and the new sample
    float3 finalColor = lerp(previousColor, newSample, LEARNING_RATE);

    // --- 4. Store the updated result in the current buffer ---
    finalColor.x = fminf(finalColor.x, 2.0f); // Allow for brightness > 1 temporarily
    finalColor.y = fminf(finalColor.y, 2.0f);
    finalColor.z = fminf(finalColor.z, 2.0f);

    GIdata_curr[idx] = make_uchar4(fminf(finalColor.x, 1.0f) * 255,
                                   fminf(finalColor.y, 1.0f) * 255,
                                   fminf(finalColor.z, 1.0f) * 255,
                                   255);
}

void CoarseArray::InitializeGIData(CArray& fineArray, CoarseArray csdf, Texturepack& texturepack)
{
    glm::vec3 sunDir = glm::normalize(glm::vec3(10.f, 5.f, -4.f));
    cudaMemcpyToSymbol(c_sunDir2, &sunDir, sizeof(glm::vec3));

    const unsigned long threads = 128;
    unsigned int blocks = (unsigned int)((GI_SIZE + (uint64_t)threads - 1ull) / (uint64_t)threads);
    
    InitialGlobalIlluminate<<<blocks, threads>>>((uchar4*)m_csdfArray.getPtr(), fineArray.getPtr(), reinterpret_cast<unsigned char*>(csdf.getPtr())) ;
    CUDA_CHECK(cudaGetLastError());    
}




#define RAYPS (64*64*64*1)
static int frameNumber = 0;
static uint64_t offsetCounter = 0;

void CoarseArray::UpdateGIData(CArray& fineArray, CoarseArray csdf, Texturepack& texturepack)
{
    glm::vec3 sunDir = glm::normalize(glm::vec3(10.f, 5.f, -4.f));
    cudaMemcpyToSymbol(c_sunDir2, &sunDir, sizeof(glm::vec3));

    const unsigned long threads = 128;
    unsigned int blocks = (unsigned int)((RAYPS + (uint64_t)threads - 1ull) / (uint64_t)threads);

 //   unsigned int blocks = (unsigned int)((RAYPS + (uint64_t)threads - 1ull) / (uint64_t)threads);
    

    GlobalIlluminate<<<blocks, threads>>>((uchar4*)m_csdfArray.getPtr(), fineArray.getPtr(), reinterpret_cast<unsigned char*>(csdf.getPtr()), texturepack.texObject(), frameNumber, offsetCounter) ;
    CUDA_CHECK(cudaGetLastError());    
    frameNumber++;
    

    if(offsetCounter + RAYPS >= GI_SIZE) 
        offsetCounter = 0;
    else offsetCounter += RAYPS;
}
















// __global__ void GlobalIlluminate(uchar4* GIdata_curr,             // The buffer we are writing to
//                                  const uchar4* __restrict__ GIdata_prev, // The buffer from the previous frame
//                                  const uint32_t* __restrict__ bits,
//                                  const unsigned char* __restrict__ csdf)
// {
//     uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
//     if (idx >= GI_BYTESIZE) return;

//     // --- 1. Calculate the world position of this GI voxel ---
//     uint64_t cz = idx / (GI_SIZEX * GI_SIZEY);
//     uint64_t temp = idx % (GI_SIZEX * GI_SIZEY);
//     uint64_t cy = temp / GI_SIZEX;
//     uint64_t cx = temp % GI_SIZEX;

//     int3 gpos = make_int3(cx, cy, cz); // GI voxel grid position

//     // The world position is the center of the coarse voxel
//     float3 worldPos = make_float3((gpos.x + 0.5f) * COARSENESSGI,
//                                  (gpos.y + 0.5f) * COARSENESSGI,
//                                  (gpos.z + 0.5f) * COARSENESSGI);

//     // --- 2. Light Injection: Calculate Direct Light ---
//     float3 directLight = make_float3(0.0f, 0.0f, 0.0f);
//     // Before tracing, check if this GI voxel is inside a solid block. If so, it can't receive light.
//     int3 voxel_ipos = make_int3(floorf(worldPos.x), floorf(worldPos.y), floorf(worldPos.z));
//     if (!IsSolid(voxel_ipos, bits))
//     {
//         hitInfo shadowHit = trace(worldPos, c_sunDir2, 0.0001f, bits, csdf);
//         if (!shadowHit.hit) {
//             // If we didn't hit anything, this voxel is lit by the sun
//             directLight = c_sunColor;
//         }
//     }


//     // --- 3. Light Propagation: Gather Bounced Light from Neighbors ---
//     float3 bouncedLight = make_float3(0.0f, 0.0f, 0.0f);
//     int samples = 0;

//     // Define offsets to the 6 direct neighbors
//     int3 offsets[] = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1} };

//     for(int i = 0; i < NUM_BOUNCE_SAMPLES; i++)
//     {
//         int3 neighbor_gpos = gpos + offsets[i];

//         // Bounds check for the neighbor
//         if (neighbor_gpos.x >= 0 && neighbor_gpos.x < GI_SIZEX &&
//             neighbor_gpos.y >= 0 && neighbor_gpos.y < GI_SIZEY &&
//             neighbor_gpos.z >= 0 && neighbor_gpos.z < GI_SIZEZ)
//         {
//             // Simple occlusion check: is the neighbor voxel solid? If so, don't gather light from it.
//             float3 neighbor_worldPos = make_float3((neighbor_gpos.x + 0.5f) * COARSENESSGI, (neighbor_gpos.y + 0.5f) * COARSENESSGI, (neighbor_gpos.z + 0.5f) * COARSENESSGI);
//             int3 neighbor_voxel_ipos = make_int3(floorf(neighbor_worldPos.x), floorf(neighbor_worldPos.y), floorf(neighbor_worldPos.z));

//             if (!IsSolid(neighbor_voxel_ipos, bits))
//             {
//                 // If not occluded, read the color from the PREVIOUS frame's buffer
//                 uint64_t neighbor_idx = (uint64_t)neighbor_gpos.z * GI_SIZEX * GI_SIZEY + (uint64_t)neighbor_gpos.y * GI_SIZEX + neighbor_gpos.x;
//                 uchar4 prevSample = GIdata_prev[neighbor_idx];
//                 bouncedLight += make_float3(prevSample.x / 255.0f, prevSample.y / 255.0f, prevSample.z / 255.0f);
//                 samples++;
//             }
//         }
//     }

//     if (samples > 0) {
//         bouncedLight = bouncedLight / (float)samples; // Average the collected light
//     }

//     // --- 4. Final Combination ---
//     // The final color is the direct light plus the bounced light from neighbors.
//     float3 finalColor = directLight + bouncedLight * BOUNCE_STRENGTH;

//     // Clamp the color to avoid overflow when converting back to uchar
//     finalColor.x = fminf(finalColor.x, 1.0f);
//     finalColor.y = fminf(finalColor.y, 1.0f);
//     finalColor.z = fminf(finalColor.z, 1.0f);

//     // Write the final result to the CURRENT frame's buffer
//     GIdata_curr[idx] = make_uchar4(finalColor.x * 255,
//                                    finalColor.y * 255,
//                                    finalColor.z * 255,
//                                    255);
// }