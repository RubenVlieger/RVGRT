// src/renderer/CoarseArray_impl.cu

#include "CoarseArray.h"
#include "cumath.h"
#include "raytracing_functions.h"

// --- SDF Generation Kernels (Unchanged Logic, using cumath.h types) ---

KERNEL_FUNC void computeDistX(const uint32_t* fineData, unsigned char* distX)
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

    for (uint64_t i = 1; i <= SDF_MAX_DIST; ++i)
    {
    	if (i <= cx && isCoarseBlockSolid(cx - i, cy, cz, fineData))
        {
            min_d = i;
            break;
        }
    }

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


KERNEL_FUNC void computeDistY(const unsigned char* distX, unsigned char* distY)
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

    for (uint64_t y_offset = 1; y_offset <= SDF_MAX_DIST; ++y_offset) {
        if (y_offset * y_offset >= min_dist_sq) break;

        if (cy >= y_offset) {
            uint64_t neighbor_idx = idx - y_offset * SDF_SIZEX;
            float dist_sq = (float)distX[neighbor_idx] * distX[neighbor_idx] + (float)y_offset * y_offset;
            min_dist_sq = fminf(min_dist_sq, dist_sq);
        }
        if (cy + y_offset < SDF_SIZEY) {
            uint64_t neighbor_idx = idx + y_offset * SDF_SIZEX;
            float dist_sq = (float)distX[neighbor_idx] * distX[neighbor_idx] + (float)y_offset * y_offset;
            min_dist_sq = fminf(min_dist_sq, dist_sq);
        }
    }
    distY[idx] = (unsigned char)fminf((float)SDF_MAX_DIST, sqrtf(min_dist_sq));
}

KERNEL_FUNC void computeDistZ(const unsigned char* distXY, unsigned char* finalCSDF)
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
    
    for (uint64_t z_offset = 1; z_offset <= SDF_MAX_DIST; ++z_offset) {
        if (z_offset * z_offset >= min_dist_sq) break;

        if (cz >= z_offset) {
            uint64_t neighbor_idx = idx - z_offset * (SDF_SIZEX * SDF_SIZEY);
            float dist_sq = (float)distXY[neighbor_idx] * distXY[neighbor_idx] + (float)z_offset * z_offset;
            min_dist_sq = fminf(min_dist_sq, dist_sq);
        }
        if (cz + z_offset < SDF_SIZEZ) {
            uint64_t neighbor_idx = idx + z_offset * (SDF_SIZEX * SDF_SIZEY);
            float dist_sq = (float)distXY[neighbor_idx] * distXY[neighbor_idx] + (float)z_offset * z_offset;
            min_dist_sq = fminf(min_dist_sq, dist_sq);
        }
    }
    finalCSDF[idx] = (unsigned char)fminf((float)SDF_MAX_DIST, sqrtf(min_dist_sq));
}


// --- GI Generation Kernels (Refactored to use cumath.h types) ---

CONST_MEM float3 c_sunDir2;

KERNEL_FUNC void InitialGlobalIlluminate(uint32_t* GIdata, const uint32_t* RESTRICT bits, const unsigned char* RESTRICT csdf)
{
    // This kernel does not use random numbers, so it is unchanged.
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (idx >= GI_SIZE) return;
    uint64_t cz = idx / (GI_SIZEX * GI_SIZEY);
    uint64_t temp = idx % (GI_SIZEX * GI_SIZEY);
    uint64_t cy = temp / GI_SIZEX;
    uint64_t cx = temp % GI_SIZEX;
    float3 worldPos = make_float3((cx + 0.5f) * COARSENESSGI, (cy + 0.5f) * COARSENESSGI, (cz + 0.5f) * COARSENESSGI);
    float3 accumulatedColor = make_float3(0.0f);
    hitInfo shadowHit = trace(worldPos, c_sunDir2, 0.0001f, bits, csdf);
    if (!shadowHit.hit) {
        accumulatedColor = c_sunColor;
    }
    GIdata[idx] = ((uint32_t)(accumulatedColor.x * 255.f) << 0) | ((uint32_t)(accumulatedColor.y * 255.f) << 8) | ((uint32_t)(accumulatedColor.z * 255.f) << 16) | (255u << 24);
}

KERNEL_FUNC void GlobalIlluminate(uint32_t* GIdata_curr, const uint32_t* RESTRICT bits, const unsigned char* RESTRICT csdf,
                                 TEXTURE_OBJECT texturepack, unsigned int frameNumber, uint64_t offset)
{
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x + offset;
    if (idx >= GI_SIZE) return;

    uint random_state = init_random_state(idx, frameNumber);

    uint64_t cz = idx / (GI_SIZEX * GI_SIZEY);
    uint64_t temp = idx % (GI_SIZEX * GI_SIZEY);
    uint64_t cy = temp / GI_SIZEX;
    uint64_t cx = temp % GI_SIZEX;
    float3 worldPos = make_float3((cx + 0.5f) * COARSENESSGI, (cy + 0.5f) * COARSENESSGI, (cz + 0.5f) * COARSENESSGI);

    if (IsSolid(floor(worldPos), bits)) { return; }

    float3 newSample = make_float3(0.0f);
    hitInfo shadowHit = trace(worldPos, c_sunDir2, 0.001f, bits, csdf);
    if (!shadowHit.hit) {
        newSample += c_sunColor;
    }
    
    float3 randomDir = random_direction_in_sphere(random_state);
    hitInfo bounceHit = trace(worldPos, randomDir, 0.001f, bits, csdf);

    if (bounceHit.hit) {
        int3 g = floor(bounceHit.pos / (float)COARSENESSGI);
        if (g.x >= 0 && g.y >= 0 && g.z >=0 && g.x < GI_SIZEX && g.y < GI_SIZEY && g.z < GI_SIZEZ) {
            uint64_t hit_idx = (uint64_t)g.z * GI_SIZEX * GI_SIZEY + (uint64_t)g.y * GI_SIZEX + g.x;
            uint32_t prevSample = GIdata_curr[hit_idx];
            float3 bouncedColor = make_float3((prevSample & 255) / 255.0f, ((prevSample >> 8) & 255) / 255.0f, ((prevSample >> 16) & 255) / 255.0f);
            float3 surfaceAlbedo = sampleTexture(bounceHit.uv, bounceHit.pos, texturepack);
            newSample += (bouncedColor * surfaceAlbedo);
        }
    } else {
        newSample += sampleSky(randomDir, c_sunDir2);
    }
    const float LEARNING_RATE = 0.04f;
    uint32_t prevData = GIdata_curr[idx];
    float3 previousColor = make_float3((prevData & 255) / 255.0f, ((prevData >> 8) & 255) / 255.0f, ((prevData >> 16) & 255) / 255.0f);
    float3 finalColor = lerp(previousColor, newSample, LEARNING_RATE);
    finalColor.x = fminf(finalColor.x, 2.0f); finalColor.y = fminf(finalColor.y, 2.0f); finalColor.z = fminf(finalColor.z, 2.0f);
    GIdata_curr[idx] = ((uint32_t)(fminf(finalColor.x, 1.0f) * 255.f) << 0) | ((uint32_t)(fminf(finalColor.y, 1.0f) * 255.f) << 8) | ((uint32_t)(fminf(finalColor.z, 1.0f) * 255.f) << 16) | (255u << 24);
}



// --- Host-side C++ Class Implementation ---

CoarseArray::CoarseArray() {}
CoarseArray::~CoarseArray() {}

void CoarseArray::AllocateSDF() 
{
    m_csdfArray.Allocate(SDF_BYTESIZE);
}

void CoarseArray::AllocateGI() 
{
    m_csdfArray.Allocate(GI_BYTESIZE);
}

unsigned char* CoarseArray::getPtr()
{
    return reinterpret_cast<unsigned char*>(m_csdfArray.getPtr());
}

void CoarseArray::GenerateSDF(CArray& fineArray)
{
    if (m_csdfArray.getSize() != SDF_BYTESIZE)
    {
        std::cerr << "CSDF not allocated or wrong size. Call AllocateSDF() first." << std::endl;
        return;
    }
    CArray tempArray;
    tempArray.Allocate(SDF_BYTESIZE);

    const unsigned long threads = 256;
    unsigned int blocks = (unsigned int)((SDF_BYTESIZE + (uint64_t)threads - 1ull) / (uint64_t)threads);
    
    computeDistX<<<blocks, threads>>>(fineArray.getPtr(), reinterpret_cast<unsigned char*>(tempArray.getPtr()));
    CUDA_CHECK(cudaGetLastError());
    
    computeDistY<<<blocks, threads>>>(reinterpret_cast<unsigned char*>(tempArray.getPtr()), getPtr());
    CUDA_CHECK(cudaGetLastError());
    
    computeDistZ<<<blocks, threads>>>(getPtr(), reinterpret_cast<unsigned char*>(tempArray.getPtr()));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(m_csdfArray.getPtr(), tempArray.getPtr(), SDF_BYTESIZE, cudaMemcpyDeviceToDevice));
    
    tempArray.Free();

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "CSDF Generation Complete." << std::endl;
}

void CoarseArray::InitializeGIData(CArray& fineArray, CoarseArray& csdf, Texturepack& texturepack)
{
    float3 sunDir = normalize(make_float3(10.f, 5.f, -4.f));
    cudaMemcpyToSymbol(c_sunDir2, &sunDir, sizeof(float3));

    const unsigned long threads = 128;
    unsigned int blocks = (unsigned int)((GI_SIZE + (uint64_t)threads - 1ull) / (uint64_t)threads);
    
    InitialGlobalIlluminate<<<blocks, threads>>>((uint32_t*)m_csdfArray.getPtr(), fineArray.getPtr(), csdf.getPtr());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


#define RAYPS (64*64*64*1)
static int frameNumber = 0;
static uint64_t offsetCounter = 0;

void CoarseArray::UpdateGIData(CArray& fineArray, CoarseArray& csdf, Texturepack& texturepack)
{
    float3 sunDir = normalize(make_float3(10.f, 5.f, -4.f));
    cudaMemcpyToSymbol(c_sunDir2, &sunDir, sizeof(float3));

    const unsigned long threads = 128;
    unsigned int blocks = (unsigned int)((RAYPS + (uint64_t)threads - 1ull) / (uint64_t)threads);
    
    GlobalIlluminate<<<blocks, threads>>>((uint32_t*)m_csdfArray.getPtr(), fineArray.getPtr(), csdf.getPtr(), texturepack.texObject(), frameNumber, offsetCounter) ;
    CUDA_CHECK(cudaGetLastError());    
    frameNumber++;
    
    if(offsetCounter + RAYPS >= GI_SIZE)
        offsetCounter = 0;
    else
        offsetCounter += RAYPS;
}