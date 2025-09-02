#include "CSDF.cuh"
#include "cumath.cuh"
#include <iostream>



//================================================================================
// KERNELS
//================================================================================

// Helper function to check if any bit is set within a coarse block.
__device__ bool isCoarseBlockSolid(uint64_t cx, uint64_t cy, uint64_t cz, const uint32_t* fineData) {
    for (uint64_t z = 0; z < COARSENESS; ++z) {
        for (uint64_t y = 0; y < COARSENESS; ++y) {
            for (uint64_t x = 0; x < COARSENESS; ++x) {
                uint64_t fine_x = cx * COARSENESS + x;
                uint64_t fine_y = cy * COARSENESS + y;
                uint64_t fine_z = cz * COARSENESS + z;

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
__global__ void computeDistX(const uint32_t* fineData, unsigned char* distX) {
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (idx >= C_BYTESIZE) return;

    uint64_t cz = idx / (C_SIZEX * C_SIZEY);
    uint64_t temp = idx % (C_SIZEX * C_SIZEY);
    uint64_t cy = temp / C_SIZEX;
    uint64_t cx = temp % C_SIZEX;

    if (isCoarseBlockSolid(cx, cy, cz, fineData)) {
        distX[idx] = 0;
        return;
    }

    unsigned char min_d = MAX_DIST;

    // Scan left up to MAX_DIST
    for (uint64_t i = 1; i <= MAX_DIST; ++i) {
    	if (i <= cx && isCoarseBlockSolid(cx - i, cy, cz, fineData)) {
            min_d = i;
            break;
        }
    }

    // Scan right, but only up to the current minimum distance found
    for (uint64_t i = 1; i < min_d; ++i) {
        if (cx + i < C_SIZEX && isCoarseBlockSolid(cx + i, cy, cz, fineData)) {
            min_d = i;
            break;
        }
    }
    distX[idx] = min_d;
}


// Kernel 2: Takes X distances and computes the 2D distance in the XY plane.
__global__ void computeDistY(const unsigned char* distX, unsigned char* distY) {
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (idx >= C_BYTESIZE) return;

    unsigned char current_dx = distX[idx];
    if (current_dx == 0) {
        distY[idx] = 0;
        return;
    }

    uint64_t cz = idx / (C_SIZEX * C_SIZEY);
    uint64_t temp = idx % (C_SIZEX * C_SIZEY);
    uint64_t cy = temp / C_SIZEX;

    float min_dist_sq = (float)current_dx * (float)current_dx;

    // Scan along the Y-axis up to MAX_DIST
    for (uint64_t y_offset = 1; y_offset <= MAX_DIST; ++y_offset) {
        // Early exit optimization: if y^2 is already > min_dist_sq, no closer point can be found on this axis.
        if (y_offset * y_offset >= min_dist_sq) break;

        // Check upwards
        if (cy - y_offset >= 0) {
            uint64_t neighbor_idx = idx - y_offset * C_SIZEX;
            float dist_sq = (float)distX[neighbor_idx] * distX[neighbor_idx] + (float)y_offset * y_offset;
            min_dist_sq = fminf(min_dist_sq, dist_sq);
        }
        // Check downwards
        if (cy + y_offset < C_SIZEY) {
            uint64_t neighbor_idx = idx + y_offset * C_SIZEX;
            float dist_sq = (float)distX[neighbor_idx] * distX[neighbor_idx] + (float)y_offset * y_offset;
            min_dist_sq = fminf(min_dist_sq, dist_sq);
        }
    }
    distY[idx] = (unsigned char)fminf((float)MAX_DIST, sqrtf(min_dist_sq));
}

// Kernel 3: Takes XY distances and computes the final 3D distance.
__global__ void computeDistZ(const unsigned char* distXY, unsigned char* finalCSDF) {
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (idx >= C_BYTESIZE) return;

    unsigned char current_dxy = distXY[idx];
    if (current_dxy == 0) {
        finalCSDF[idx] = 0;
        return;
    }
    
    uint64_t cz = idx / (C_SIZEX * C_SIZEY);

    float min_dist_sq = (float)current_dxy * current_dxy;
    
    // Scan along the Z-axis up to MAX_DIST
    for (uint64_t z_offset = 1; z_offset <= MAX_DIST; ++z_offset) {
        // Early exit optimization
        if (z_offset * z_offset >= min_dist_sq) break;

        // Check backwards
        if (cz - z_offset >= 0) {
            uint64_t neighbor_idx = idx - z_offset * (C_SIZEX * C_SIZEY);
            float dist_sq = (float)distXY[neighbor_idx] * distXY[neighbor_idx] + (float)z_offset * z_offset;
            min_dist_sq = fminf(min_dist_sq, dist_sq);
        }
        // Check forwards
        if (cz + z_offset < C_SIZEZ) {
            uint64_t neighbor_idx = idx + z_offset * (C_SIZEX * C_SIZEY);
            float dist_sq = (float)distXY[neighbor_idx] * distXY[neighbor_idx] + (float)z_offset * z_offset;
            min_dist_sq = fminf(min_dist_sq, dist_sq);
        }
    }
    finalCSDF[idx] = (unsigned char)fminf((float)MAX_DIST, sqrtf(min_dist_sq));
}


//================================================================================
// CLASS IMPLEMENTATION
//================================================================================

CSDF::CSDF() {}
CSDF::~CSDF() {}

void CSDF::Allocate() {
    m_csdfArray.Allocate(C_BYTESIZE);
}

unsigned char* CSDF::getPtr() {
    return reinterpret_cast<unsigned char*>(m_csdfArray.getPtr());
}

void CSDF::Generate(CArray& fineArray) {
    if (m_csdfArray.getSize() != C_BYTESIZE) {
        std::cerr << "CSDF not allocated or wrong size. Call Allocate() first." << std::endl;
        return;
    }

    // Create a temporary CArray for intermediate calculations.
    CArray tempArray;
    tempArray.Allocate(C_BYTESIZE);

    // Common launch configuration for the coarse grid
    const uint64_t threads = 256;
    uint64_t blocks = (C_BYTESIZE + threads - 1) / threads;
    
    // --- Pass 1: X-axis distance ---
    // Input: fineArray, Output: tempArray
    computeDistX<<<blocks, threads>>>(fineArray.getPtr(), reinterpret_cast<unsigned char*>(tempArray.getPtr()));
    CUDA_CHECK(cudaGetLastError());
    
    // --- Pass 2: Y-axis distance (creates 2D distance) ---
    // Input: tempArray, Output: m_csdfArray
    computeDistY<<<blocks, threads>>>(reinterpret_cast<unsigned char*>(tempArray.getPtr()), getPtr());
    CUDA_CHECK(cudaGetLastError());
    
    // --- Pass 3: Z-axis distance (creates final 3D distance) ---
    // Input: m_csdfArray, Output: tempArray (reusing buffer)
    computeDistZ<<<blocks, threads>>>(getPtr(), reinterpret_cast<unsigned char*>(tempArray.getPtr()));
    CUDA_CHECK(cudaGetLastError());

    // The final result is now in tempArray. We need to copy it back to our member array.
    CUDA_CHECK(cudaMemcpy(m_csdfArray.getPtr(), tempArray.getPtr(), C_BYTESIZE, cudaMemcpyDeviceToDevice));
    
    // Free the temporary buffer
    tempArray.Free();

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "CSDF Generation Complete." << std::endl;
}

