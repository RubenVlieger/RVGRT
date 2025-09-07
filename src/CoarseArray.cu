#include "CoarseArray.cuh"
#include "cumath.cuh"
#include <iostream>
#include "raytracing_functions.cuh"
#include "glm/glm.hpp"

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
__global__ void GlobalIlluminate(uchar4* GIdata,
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
void CoarseArray::UpdateGI(CArray& fineArray, CoarseArray csdf)
{

}
void CoarseArray::GenerateGIdata(CArray& fineArray, CoarseArray csdf)
{
    glm::vec3 sunDir = glm::normalize(glm::vec3(10.f, 5.f, -4.f));
    cudaMemcpyToSymbol(c_sunDir2, &sunDir, sizeof(glm::vec3));

    const unsigned long threads = 256;
    unsigned int blocks = (unsigned int)((GI_SIZE + (uint64_t)threads - 1ull) / (uint64_t)threads);
    
    // --- Pass 1: X-axis distance ---
    GlobalIlluminate<<<blocks, threads>>>((uchar4*)m_csdfArray.getPtr(), fineArray.getPtr(), reinterpret_cast<unsigned char*>(csdf.getPtr()));
    CUDA_CHECK(cudaGetLastError());    
}