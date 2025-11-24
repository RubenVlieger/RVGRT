#pragma once

#include "CArray.h"
#include "Texturepack.h"
#include "cumath.h"

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


#ifndef __METAL_VERSION__
#ifdef __OBJC__
@protocol MTLComputeCommandEncoder;
@protocol MTLComputePipelineState;
@protocol MTLTexture; 
#else
typedef void* id;
#endif

class CoarseArray {
public:
    CoarseArray();
    ~CoarseArray();
    void AllocateSDF();
    void AllocateGI();

    void GenerateSDF(void* packedVoxelTexture);
    
    void InitializeGIData(void* packedVoxelTexture, CoarseArray& csdf, Texturepack& texturepack);

    void UpdateGIData(id<MTLComputeCommandEncoder> encoder, void* packedVoxelTexture, CoarseArray& csdf, Texturepack& texturepack);

    void* getSDFTexture();
    void* getGITexture();

    unsigned char* getPtr(); 
    
private:
    CArray m_csdfArray; 

    void* _sdfTexture = nullptr;
    void* _giTexture = nullptr;

    #ifdef __OBJC__ 
    id _psoDistX = nullptr;
    id _psoDistY = nullptr;
    id _psoDistZ = nullptr;
    id _psoGiInit = nullptr;
    id _psoGiUpdate = nullptr;
    #endif
};
#endif