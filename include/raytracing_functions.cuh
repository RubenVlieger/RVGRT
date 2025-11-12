#include "cumath.h" 

class CoarseArray;

struct hitInfo
{
    float3 pos;
    float3 normal;
    half2 uv; 
    bool hit;
    int its;
};

/**
 * @brief Checks if a voxel at a given integer coordinate is solid.
 * @param p The integer coordinates (x, y, z) of the voxel.
 * @param bits Pointer to the packed voxel data on the GPU.
 * @return True if the voxel is solid, false otherwise.
 */
__device__ __forceinline__ bool IsSolid(const int3& p, const uint32_t* __restrict__ bits);

/**
 * @brief Samples the Coarse Signed Distance Field (CSDF) at a floating-point world position.
 * @param pos The world position to sample from.
 * @param csdf Pointer to the CSDF data on the GPU.
 * @return The approximate distance to the nearest surface in coarse grid units.
 */
__device__ float getDistance(const float3& pos, const unsigned char* __restrict__ csdf);

/**
 * @brief Samples the CSDF at an integer voxel coordinate.
 * @param pos The integer voxel coordinates.
 * @param csdf Pointer to the CSDF data on the GPU.
 * @return The approximate distance from the coarse cell containing the voxel.
 */
__device__ unsigned char getDistance(const int3& pos, const unsigned char* __restrict__ csdf);

/**
 * @brief Marches a ray using the CSDF to quickly find a point near a surface.
 * @param pos The starting position of the ray.
 * @param dir The normalized direction of the ray.
 * @param csdf Pointer to the CSDF data on the GPU.
 * @return A float3 point on the ray that is close to a solid surface.
 */
__device__ float3 approximateCSDF(float3 pos, const float3& dir, const unsigned char* __restrict__ csdf);

/**
 * @brief The main hybrid ray tracing function combining CSDF marching and DDA.
 * @param camPos The starting position of the ray.
 * @param camDir The normalized direction of the ray.
 * @param distance An initial distance to advance the ray before tracing begins.
 * @param bits Pointer to the packed high-resolution voxel data.
 * @param csdf Pointer to the CSDF data.
 * @return A hitInfo struct containing the results of the trace.
 */
__device__ hitInfo trace(float3 camPos, const float3& camDir, half distance,
                        const uint32_t* __restrict__ bits, const unsigned char* __restrict__ csdf);

/**
 * @brief Traces a cone through the GI data grid to gather indirect illumination.
 * @param pos The starting point of the cone (a surface point).
 * @param dir The central direction of the cone.
 * @param GIdata Pointer to the 3D grid of GI voxel data.
 * @param csdf Pointer to the CSDF for occlusion checks.
 * @return A float3 color representing the accumulated indirect light.
 */
__device__ float3 traceCone(float3 pos, const float3& dir, const uchar4* __restrict__ GIdata,
                           const unsigned char* __restrict__ csdf);

/**
 * @brief Calculates the color of the sky for a given view direction.
 * @param dir The normalized view direction.
 * @param sunDir The normalized direction to the sun.
 * @return A float3 representing the sky color.
 */
__device__ float3 sampleSky(const float3& dir, const float3& sunDir);

/**
 * @brief Samples the texture atlas based on voxel position and hit UVs.
 * @param uv The UV coordinates on the face of the hit voxel.
 * @param pos The world position of the hit.
 * @param texObj The CUDA texture object for the texture atlas.
 * @return A float3 representing the albedo color from the texture.
 */
__device__ float3 sampleTexture(half2 uv, const float3& pos, TEXTURE_OBJECT texObj);