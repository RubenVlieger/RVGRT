#pragma once
#ifdef _WIN32

#include <cuda_runtime.h>
#include "renderer/ShaderTypes.h"
#include "renderer/hitInfo.h"

// Forward declarations of the 8 render pass kernels
void launch_distApproximationKernel(cudaStream_t stream, uint32_t width, uint32_t height, cudaSurfaceObject_t halfDistOut, const int3& worldOrigin, const uint32_t* indirection, const void* sectors, const uint64_t* occupancy, const uint8_t* data, const uint64_t* sectorMasks);

void launch_GBufferAndDirectLight(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t halfDistIn, cudaSurfaceObject_t directLightOut, cudaSurfaceObject_t albedoOut, cudaSurfaceObject_t normalOut, cudaSurfaceObject_t motionOut, cudaSurfaceObject_t depthOut, const int3& worldOrigin, const uint32_t* indirection, const void* sectors, const uint64_t* occupancy, const uint8_t* data, const uint64_t* sectorMasks, cudaTextureObject_t texObj);

void launch_IndirectBounce(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t normalIn, cudaTextureObject_t depthIn, cudaSurfaceObject_t rawIndirectOut, const int3& worldOrigin, const uint32_t* indirection, const void* sectors, const uint64_t* occupancy, const uint8_t* data, const uint64_t* sectorMasks, cudaTextureObject_t texObj);

void launch_TemporalAccumulation(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t rawIndirectIn, cudaTextureObject_t directIn, cudaTextureObject_t motionIn, cudaTextureObject_t depthIn, cudaTextureObject_t prevDepthIn, cudaTextureObject_t prevAccumIn, cudaSurfaceObject_t accumOut, bool resetHistory);

void launch_BilateralDenoise(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t accumIn, cudaTextureObject_t normalIn, cudaTextureObject_t depthIn, cudaSurfaceObject_t denoisedOut, float stepWidth);

void launch_VolumetricFog(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t depthIn, cudaTextureObject_t prevVolumetricIn, cudaSurfaceObject_t volumetricOut, const int3& worldOrigin, const uint32_t* indirection, const void* sectors, const uint64_t* occupancy, const uint8_t* data, const uint64_t* sectorMasks);

void launch_ComputeExposure(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t texDirect, cudaTextureObject_t texAccum, cudaTextureObject_t texAlbedo, void* exposureBuffer);

void launch_Composite(cudaStream_t stream, uint32_t width, uint32_t height, cudaTextureObject_t directIn, cudaTextureObject_t albedoIn, cudaTextureObject_t denoisedIn, cudaTextureObject_t volumetricIn, cudaSurfaceObject_t finalHistoryOut, cudaSurfaceObject_t compositeResultOut, void* exposureBuffer);

// Helper to copy constants to device before dispatch
void update_constant_memory(const CameraData& camData, const FrameData& frameData, const void* charData, size_t charSize);

#endif
