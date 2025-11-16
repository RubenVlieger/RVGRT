#ifndef CSDF_CUH
#define CSDF_CUH

#include "CArray.h"
#include "Texturepack.h"
#include "cumath.h"

// Define the dimensions of the coarse grid relative to the fine grid.
// A coarseness of 2 means each CSDF cell represents a 2x2x2 block of voxels.
#define COARSENESSSDF 2
#define SDF_SIZEX (SIZEX / COARSENESSSDF)
#define SDF_SIZEY (SIZEY / COARSENESSSDF)
#define SDF_SIZEZ (SIZEZ / COARSENESSSDF)
#define SDF_BYTESIZE (SDF_SIZEX * SDF_SIZEY * SDF_SIZEZ)
#define SDF_MAX_DIST 64

#define COARSENESSGI 4
#define GI_SIZEX (SIZEX / COARSENESSGI)
#define GI_SIZEY (SIZEY / COARSENESSGI)
#define GI_SIZEZ (SIZEZ / COARSENESSGI)
#define GI_SIZE (GI_SIZEX * GI_SIZEY * GI_SIZEZ)
#define GI_BYTESIZE (GI_SIZEX * GI_SIZEY * GI_SIZEZ * sizeof(uint32_t))

// Helper to convert GLM vector to CUDA vector for interop with existing functions
GPU_FUNC GPU_INLINE int3 to_int3(const int3 v) {
    return make_int3(v.x, v.y, v.z);
}
GPU_FUNC GPU_INLINE int3 to_int3(const float3 v) {
    return make_int3((float)v.x, (float)v.y, (float)v.z);
}


GPU_FUNC GPU_INLINE uint32_t init_random_state(uint64_t thread_id, int frame_number) {
    return thread_id + frame_number * 198491317;
}

#if defined(__METAL_VERSION__)
GPU_FUNC GPU_INLINE float random_float(thread uint &state) {
#else
GPU_FUNC GPU_INLINE float random_float(uint &state) {
#endif
    state ^= (state << 13);
    state ^= (state >> 17);
    state ^= (state << 5);
    return float(state) / float(4294967295.0f);
}

// This function must also now accept the state to pass it to random_float.
#if defined(__METAL_VERSION__)
GPU_FUNC GPU_INLINE float3 random_direction_in_sphere(thread uint &state) {
#else
GPU_FUNC GPU_INLINE float3 random_direction_in_sphere(uint &state) {
#endif
    float3 p;
    do {
        p = make_float3(random_float(state) * 2.0f - 1.0f,
                        random_float(state) * 2.0f - 1.0f,
                        random_float(state) * 2.0f - 1.0f);
    } while (dot(p, p) >= 1.0f);
    return normalize(p);
}


// --- SDF Generation Kernels (Unchanged as they use scalar types) ---
GPU_FUNC GPU_INLINE bool isCoarseBlockSolid(uint64_t cx, uint64_t cy, uint64_t cz, DEVICE_PTR(const uint32_t*) RESTRICT fineData)
{
    for (uint64_t z = 0; z < COARSENESSSDF; ++z) {
        for (uint64_t y = 0; y < COARSENESSSDF; ++y) {
            for (uint64_t x = 0; x < COARSENESSSDF; ++x) {
                uint64_t fine_x = cx * COARSENESSSDF + x;
                uint64_t fine_y = cy * COARSENESSSDF + y;
                uint64_t fine_z = cz * COARSENESSSDF + z;

                if (fine_x >= SIZEX || fine_y >= SIZEY || fine_z >= SIZEZ) continue;

                uint64_t index = toIndex(fine_x, fine_y, fine_z);
                if ((fineData[index >> 5] >> (index & 31)) & 1) {
                    return true;
                }
            }
        }
    }
    return false;
}

#ifndef __METAL_VERSION__

class CoarseArray {
public:
    CoarseArray();
    ~CoarseArray();

    // Allocates memory for the Coarse Signed Distance Field.
    void AllocateSDF();
    void AllocateGI();

    // Generates the SDF from the high-resolution voxel data.
    void GenerateSDF(CArray& fineArray);

    // Initializes and updates the Global Illumination data grid.
    void InitializeGIData(CArray& fineArray, CoarseArray& csdf, Texturepack& texture);
    void UpdateGIData(CArray& fineArray, CoarseArray& csdf, Texturepack& texture);

    // Provides access to the device pointer of the generated data.
    unsigned char* getPtr();

private:
    // CArray to hold the data.
    CArray m_csdfArray;
};
#endif

#endif