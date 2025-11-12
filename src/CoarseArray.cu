#include "CoarseArray.cuh"
#include "cumath.hpp" // Kept for constants and toIndex as requested
#include "glm/glm.hpp" // Use GLM for vector math
#include "raytracing_functions.cuh"
#include <iostream>

#define BOUNCE_STRENGTH 0.9f
#define NUM_BOUNCE_SAMPLES 6

// Helper to convert GLM vector to CUDA vector for interop with existing functions
__device__ inline int3 to_int3(const glm::ivec3& v) {
    return make_int3(v.x, v.y, v.z);
}

// --- SDF Generation Kernels (Unchanged as they use scalar types) ---

__device__ bool isCoarseBlockSolid(uint64_t cx, uint64_t cy, uint64_t cz, const uint32_t* fineData)
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


// --- GI Generation Kernels (Refactored to use GLM) ---

__constant__ glm::vec3 c_sunDir2;

__global__ void InitialGlobalIlluminate(uchar4* GIdata,
                                        const uint32_t* __restrict__ bits,
                                        const unsigned char* __restrict__ csdf)
{
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (idx >= GI_SIZE) return;

    uint64_t cz = idx / (GI_SIZEX * GI_SIZEY);
    uint64_t temp = idx % (GI_SIZEX * GI_SIZEY);
    uint64_t cy = temp / GI_SIZEX;
    uint64_t cx = temp % GI_SIZEX;

    glm::vec3 worldPos((cx + 0.5f) * COARSENESSGI,
                       (cy + 0.5f) * COARSENESSGI,
                       (cz + 0.5f) * COARSENESSGI);

    glm::vec3 accumulatedColor(0.0f);
    
    hitInfo shadowHit = trace(worldPos, c_sunDir2, 0.0001f, bits, csdf);

    if (!shadowHit.hit) {
        accumulatedColor = glm::vec3(c_sunColor.x, c_sunColor.y, c_sunColor.z);
    }

    GIdata[idx] = make_uchar4(accumulatedColor.x * 255,
                              accumulatedColor.y * 255,
                              accumulatedColor.z * 255,
                              255);
}

__device__ unsigned int random_state;

__device__ void init_random_state(uint64_t thread_id, int frame_number) {
    random_state = thread_id + frame_number * 198491317;
}

__device__ float random_float() {
    random_state ^= (random_state << 13);
    random_state ^= (random_state >> 17);
    random_state ^= (random_state << 5);
    return float(random_state) / float(4294967295.0f);
}

__device__ glm::vec3 random_direction_in_sphere() {
    glm::vec3 p;
    do {
        p = glm::vec3(random_float() * 2.0f - 1.0f,
                      random_float() * 2.0f - 1.0f,
                      random_float() * 2.0f - 1.0f);
    } while (glm::dot(p, p) >= 1.0f);
    return glm::normalize(p);
}

__global__ void GlobalIlluminate(uchar4* GIdata_curr,
                                 const uint32_t* __restrict__ bits,
                                 const unsigned char* __restrict__ csdf,
                                 cudaTextureObject_t texturepack,
                                 unsigned int frameNumber,
                                 uint64_t offset)
{
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x + offset;
    if (idx >= GI_SIZE) return;

    init_random_state(idx, frameNumber);

    uint64_t cz = idx / (GI_SIZEX * GI_SIZEY);
    uint64_t temp = idx % (GI_SIZEX * GI_SIZEY);
    uint64_t cy = temp / GI_SIZEX;
    uint64_t cx = temp % GI_SIZEX;

    glm::vec3 worldPos((cx + 0.5f) * COARSENESSGI,
                       (cy + 0.5f) * COARSENESSGI,
                       (cz + 0.5f) * COARSENESSGI);
    
    if (IsSolid(glm::floor(worldPos), bits)) {
        return;
    }

    glm::vec3 newSample(0.0f);

    hitInfo shadowHit = trace(worldPos, c_sunDir2, 0.001f, bits, csdf);
    if (!shadowHit.hit) {
        newSample += glm::vec3(c_sunColor.x, c_sunColor.y, c_sunColor.z);
    }

    glm::vec3 randomDir = random_direction_in_sphere();
    hitInfo bounceHit = trace(worldPos, randomDir, 0.001f, bits, csdf);

    if (bounceHit.hit) {
        glm::ivec3 g = glm::floor(bounceHit.pos / (float)COARSENESSGI);
        if (glm::all(glm::greaterThanEqual(g, glm::ivec3(0))) && glm::all(glm::lessThan(g, glm::ivec3(GI_SIZEX, GI_SIZEY, GI_SIZEZ)))) {
            uint64_t hit_idx = (uint64_t)g.z * GI_SIZEX * GI_SIZEY + (uint64_t)g.y * GI_SIZEX + g.x;
            uchar4 prevSample = GIdata_curr[hit_idx];
            glm::vec3 bouncedColor(prevSample.x / 255.0f, prevSample.y / 255.0f, prevSample.z / 255.0f);
            
            glm::vec3 surfaceAlbedo = sampleTexture(bounceHit.uv, bounceHit.pos, texturepack);
            newSample += bouncedColor * surfaceAlbedo;
        }
    } else {
        newSample += sampleSky(randomDir, c_sunDir2);
    }

    const float LEARNING_RATE = 0.04f;
    uchar4 prevData = GIdata_curr[idx];
    glm::vec3 previousColor(prevData.x / 255.0f, prevData.y / 255.0f, prevData.z / 255.0f);
    glm::vec3 finalColor = glm::mix(previousColor, newSample, LEARNING_RATE);

    finalColor = glm::min(finalColor, glm::vec3(2.0f));

    GIdata_curr[idx] = make_uchar4(glm::min(finalColor.x, 1.0f) * 255,
                                   glm::min(finalColor.y, 1.0f) * 255,
                                   glm::min(finalColor.z, 1.0f) * 255,
                                   255);
}



// --- Class Member Functions ---

CoarseArray::CoarseArray() {}
CoarseArray::~CoarseArray() {}

void CoarseArray::AllocateSDF() {
    m_csdfArray.Allocate(SDF_BYTESIZE);
}

void CoarseArray::AllocateGI() {
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
    glm::vec3 sunDir = glm::normalize(glm::vec3(10.f, 5.f, -4.f));
    cudaMemcpyToSymbol(c_sunDir2, &sunDir, sizeof(glm::vec3));

    const unsigned long threads = 128;
    unsigned int blocks = (unsigned int)((GI_SIZE + (uint64_t)threads - 1ull) / (uint64_t)threads);
    
    InitialGlobalIlluminate<<<blocks, threads>>>((uchar4*)m_csdfArray.getPtr(), fineArray.getPtr(), csdf.getPtr()) ;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


#define RAYPS (64*64*64*1)
static int frameNumber = 0;
static uint64_t offsetCounter = 0;

void CoarseArray::UpdateGIData(CArray& fineArray, CoarseArray& csdf, Texturepack& texturepack)
{
    glm::vec3 sunDir = glm::normalize(glm::vec3(10.f, 5.f, -4.f));
    cudaMemcpyToSymbol(c_sunDir2, &sunDir, sizeof(glm::vec3));

    const unsigned long threads = 128;
    unsigned int blocks = (unsigned int)((RAYPS + (uint64_t)threads - 1ull) / (uint64_t)threads);
    
    GlobalIlluminate<<<blocks, threads>>>((uchar4*)m_csdfArray.getPtr(), fineArray.getPtr(), csdf.getPtr(), texturepack.texObject(), frameNumber, offsetCounter) ;
    CUDA_CHECK(cudaGetLastError());    
    frameNumber++;
    
    if(offsetCounter + RAYPS >= GI_SIZE)
        offsetCounter = 0;
    else
        offsetCounter += RAYPS;
}