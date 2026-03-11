// ============================================================================
// CudaRenderer.cu - CUDA implementation mirroring MetalRenderer
// ============================================================================

#include "renderer/CUDA/CudaRender.cuh"
#include "Character.hpp"
#include "State.hpp"
#include "cumath.h"
#include "renderer/ShaderTypes.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifdef HAS_STREAMLINE
#include <sl.h>
#include <sl_dlss.h>
#endif

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// CONSTANT MEMORY (matching Metal's constant buffers)
// ============================================================================

__constant__ CameraData c_camera;
__constant__ FrameData c_frame;

// ============================================================================
// KERNEL DECLARATIONS (from preprocessed .shader files)
// These will be linked from the preprocessed shader files
// ============================================================================

// Distance Approximation Pass (half-res)
__global__ void distApproximationKernel(
    cudaSurfaceObject_t distSurf,
    CameraData camera,
    FrameData frame,
    const uint32_t* indirection,
    SectorInfo* sectorBuffer,
    uint64_t* occupancyBuffer,
    uint8_t* dataBuffer,
    uint64_t* sectorMaskBuffer,
    CharacterGPUData charData,
    int width, int height
);

// GBuffer + Direct Light Pass (full-res)
__global__ void GBufferAndDirectLight(
    cudaSurfaceObject_t texDirectLight,
    cudaSurfaceObject_t texAlbedo,
    cudaSurfaceObject_t texNormal,
    cudaSurfaceObject_t texMotion,
    cudaSurfaceObject_t texDepth,
    CameraData camera,
    FrameData frame,
    const uint32_t* indirection,
    SectorInfo* sectorBuffer,
    uint64_t* occupancyBuffer,
    uint8_t* dataBuffer,
    uint64_t* sectorMaskBuffer,
    CharacterGPUData charData,
    cudaTextureObject_t textureAtlas,
    cudaTextureObject_t halfDistTex,
    int width, int height
);

// Indirect Bounce Pass (full-res)
__global__ void IndirectBounce(
    cudaSurfaceObject_t texRawIndirect,
    cudaTextureObject_t texNormal,
    cudaTextureObject_t texDepth,
    CameraData camera,
    FrameData frame,
    cudaTextureObject_t textureAtlas,
    const uint32_t* indirection,
    SectorInfo* sectorBuffer,
    uint64_t* occupancyBuffer,
    uint8_t* dataBuffer,
    uint64_t* sectorMaskBuffer,
    CharacterGPUData charData,
    int width, int height
);

// Temporal Accumulation Pass (full-res)
__global__ void TemporalAccumulation(
    cudaSurfaceObject_t texAccum,
    cudaTextureObject_t texRawIndirect,
    cudaTextureObject_t texHistory,
    cudaTextureObject_t texMotion,
    cudaTextureObject_t texDepth,
    cudaTextureObject_t texPrevDepth,
    cudaTextureObject_t texDirect,
    int width, int height
);

// Bilateral Denoise Pass (full-res, multi-iteration)
__global__ void BilateralDenoise(
    cudaSurfaceObject_t output,
    cudaTextureObject_t input,
    cudaTextureObject_t texNormal,
    cudaTextureObject_t texDepth,
    int stepWidth,
    int width, int height
);

// Volumetric Fog Pass (half-res)
__global__ void VolumetricFog(
    cudaSurfaceObject_t texVolumetric,
    cudaTextureObject_t texDepth,
    cudaTextureObject_t texHistory,
    CameraData camera,
    FrameData frame,
    const uint32_t* indirection,
    SectorInfo* sectorBuffer,
    uint64_t* occupancyBuffer,
    uint64_t* sectorMaskBuffer,
    CharacterGPUData charData,
    int width, int height
);

// Compute Exposure Pass (reduction)
__global__ void ComputeExposure(
    ExposureData* exposure,
    FrameData frame,
    cudaTextureObject_t texDirect,
    cudaTextureObject_t texAccum,
    cudaTextureObject_t texAlbedo,
    int width, int height
);

// Composite Pass (full-res)
__global__ void Composite(
    cudaSurfaceObject_t texFinal,
    cudaTextureObject_t texDirect,
    cudaTextureObject_t texAccum,
    cudaTextureObject_t texAlbedo,
    cudaTextureObject_t texDepth,
    cudaTextureObject_t texVolumetric,
    ExposureData* exposure,
    int width, int height
);

// DLSS Input Preparation (copy composite to interop texture)
__global__ void CopyToInteropKernel(
    cudaSurfaceObject_t src,
    cudaSurfaceObject_t dst,
    int width, int height
);

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

CudaRenderer::CudaRenderer() : _texturepack() {
    // Create CUDA stream for all rendering operations
    CUDA_CHECK(cudaStreamCreate(&_cudaStream));
    
    // Initialize exposure buffer with default value
    ExposureData initialExp = {0.5f, 0.0f, 0.0f, 0.0f};
    CUDA_CHECK(cudaMalloc(&_exposureBuffer, sizeof(ExposureData)));
    CUDA_CHECK(cudaMemcpy(_exposureBuffer, &initialExp, sizeof(ExposureData), cudaMemcpyHostToDevice));
    
    // Initialize character buffer
    CUDA_CHECK(cudaMalloc(&_characterBuffer, sizeof(CharacterGPUData)));
    
    // Generate world (exactly like MetalRenderer)
    printf("Starting Dynamic World Generation (XBrickMap)...\n");
    _materialMap.GenerateDynamic();
    printf("World Generation Complete.\n");
    
    // Create render targets
    createRenderTarget(State::dispWIDTH, State::dispHEIGHT);
}

CudaRenderer::~CudaRenderer() {
    freeRenderTargets();
    
    if (_exposureBuffer) {
        cudaFree(_exposureBuffer);
    }
    if (_characterBuffer) {
        cudaFree(_characterBuffer);
    }
    if (_cudaStream) {
        cudaStreamDestroy(_cudaStream);
    }
}

// ============================================================================
// RENDER TARGET MANAGEMENT
// ============================================================================

void CudaRenderer::allocateTarget(CudaRenderTarget& target, uint32_t width, 
                                   uint32_t height, cudaChannelFormatDesc format) {
    // 1. Allocate CUDA array
    CUDA_CHECK(cudaMallocArray(&target.array, &format, width, height, cudaArraySurfaceLoadStore));
    
    // 2. Create surface object (for kernel write)
    cudaResourceDesc surfRes = {};
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = target.array;
    CUDA_CHECK(cudaCreateSurfaceObject(&target.surface, &surfRes));
    
    // 3. Create texture object (for kernel read with filtering)
    cudaResourceDesc texRes = {};
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = target.array;
    
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = (format.f == cudaChannelFormatKindUnsigned) ? cudaFilterModePoint : cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;  // false - pixel coordinates
    
    CUDA_CHECK(cudaCreateTextureObject(&target.texture, &texRes, &texDesc, nullptr));
}

void CudaRenderer::freeTarget(CudaRenderTarget& target) {
    if (target.texture) {
        cudaDestroyTextureObject(target.texture);
        target.texture = 0;
    }
    if (target.surface) {
        cudaDestroySurfaceObject(target.surface);
        target.surface = 0;
    }
    if (target.array) {
        cudaFreeArray(target.array);
        target.array = nullptr;
    }
}

void CudaRenderer::createRenderTarget(uint32_t width, uint32_t height) {
    _width = width;
    _height = height;
    
    // Format definitions matching Metal
    cudaChannelFormatDesc rgba16f = cudaCreateChannelDescHalf4();
    cudaChannelFormatDesc rgba8 = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaChannelFormatDesc rg16f = cudaCreateChannelDescHalf2();
    cudaChannelFormatDesc r32f = cudaCreateChannelDesc<float>();
    
    // Full resolution targets
    allocateTarget(_texDirectLight, width, height, rgba16f);
    allocateTarget(_texAlbedo, width, height, rgba8);
    allocateTarget(_texNormal, width, height, rgba16f);
    allocateTarget(_texMotion, width, height, rg16f);
    allocateTarget(_texRawIndirect, width, height, rgba16f);
    allocateTarget(_texDenoised, width, height, rgba16f);
    allocateTarget(_texFinal, width, height, rgba8);
    allocateTarget(_texDenoiseTemp, width, height, rgba16f);
    allocateTarget(_texCompositeResult, width, height, rgba16f);
    
    // Ping-pong buffers
    for (int i = 0; i < 2; i++) {
        allocateTarget(_texDepth[i], width, height, r32f);
        allocateTarget(_texAccum[i], width, height, rgba16f);
        allocateTarget(_texFinalHistory[i], width, height, rgba16f);
    }
    
    // Volumetric buffers (half-res)
    for (int i = 0; i < 2; i++) {
        allocateTarget(_texVolumetric[i], width / 2, height / 2, rgba16f);
    }
    
    // Half-resolution distance texture
    allocateTarget(_halfDistTexture, width / 2, height / 2, r32f);
    
    _scalerNeedsReset = true;
}

void CudaRenderer::freeRenderTargets() {
    freeTarget(_texDirectLight);
    freeTarget(_texAlbedo);
    freeTarget(_texNormal);
    freeTarget(_texMotion);
    freeTarget(_texRawIndirect);
    freeTarget(_texDenoised);
    freeTarget(_texFinal);
    freeTarget(_texDenoiseTemp);
    freeTarget(_texCompositeResult);
    
    for (int i = 0; i < 2; i++) {
        freeTarget(_texDepth[i]);
        freeTarget(_texAccum[i]);
        freeTarget(_texFinalHistory[i]);
        freeTarget(_texVolumetric[i]);
    }
    
    freeTarget(_halfDistTexture);
}

void CudaRenderer::OnResize(uint32_t newWidth, uint32_t newHeight) {
    State::dispWIDTH = newWidth;
    State::dispHEIGHT = newHeight;
    State::screenHEIGHT = newHeight;
    State::screenWIDTH = newWidth;
    
    if (_width != newWidth || _height != newHeight) {
        freeRenderTargets();
        createRenderTarget(newWidth, newHeight);
    }
}

void CudaRenderer::ResetScaler() {
    _scalerNeedsReset = true;
}

void CudaRenderer::GenerateWorld() {
    _materialMap.GenerateDynamic();
}

// ============================================================================
// DLSS SUPPORT
// ============================================================================

void CudaRenderer::InitializeDLSS(void* d3dDevice, uint32_t width, uint32_t height) {
#ifdef HAS_STREAMLINE
    // DLSS is initialized via Streamline SDK in D3D12Device
    // Here we just mark it as available
    printf("DLSS support enabled via Streamline SDK\n");
#else
    (void)d3dDevice;
    (void)width;
    (void)height;
#endif
}

void CudaRenderer::UpdateDLSSConstants(float jitterX, float jitterY, bool reset) {
    _jitterX = jitterX;
    _jitterY = jitterY;
    if (reset) {
        _scalerNeedsReset = true;
    }
}

// ============================================================================
// MAIN DRAW LOOP (8-pass pipeline mirroring MetalRenderer)
// ============================================================================

void CudaRenderer::Draw(const Character& character, unsigned int frameCount) {
    int currIdx = _frameIndex % 2;
    int prevIdx = (_frameIndex + 1) % 2;
    
    // Prepare CameraData
    CameraData camData;
    camData.position = make_float3(
        (float)character.position.x,
        (float)character.position.y,
        (float)character.position.z
    );
    camData.forward = make_float3(
        (float)character.direction.x,
        (float)character.direction.y,
        (float)character.direction.z
    );
    
    float tanHalfFov = tanf(glm::radians(character.FOV) * 0.5f);
    float aspect = (float)State::dispWIDTH / (float)State::dispHEIGHT;
    glm::vec3 sRight = character.camera.right * tanHalfFov * aspect;
    glm::vec3 sUp = character.camera.up * tanHalfFov;
    
    camData.right = make_float3(sRight.x, sRight.y, sRight.z);
    camData.up = make_float3(sUp.x, sUp.y, sUp.z);
    camData.jitter = make_float2(character.jitterX, character.jitterY);
    memcpy(&camData.unjitteredViewProjection,
           &character.unjitteredViewProjectionMatrix, sizeof(simd_float4x4));
    memcpy(&camData.prevUnjitteredViewProjection,
           &character.lastRenderedViewProjectionMatrix, sizeof(simd_float4x4));
    
    // Prepare FrameData
    FrameData frameData;
    frameData.sunDirection = normalize(make_float3(10.f, 5.f, -4.f));
    frameData.time = fmodf((float)frameCount / 60.0f, 3600.0f);
    frameData.deltaTime = 1.0f / 60.0f;
    
    // Update material map streaming
    float3 camPosSimd = make_float3(
        (float)character.position.x,
        (float)character.position.y,
        (float)character.position.z
    );
    bool sectorsChanged = _materialMap.UpdateStreaming(camPosSimd);
    if (sectorsChanged) {
        _scalerNeedsReset = true;
    }
    frameData.worldOrigin = _materialMap.GetWorldOrigin();
    
    // Prepare character data
    CharacterGPUData charData;
    memset(&charData, 0, sizeof(CharacterGPUData));
    
    int activeChars = 0;
    auto appendCharacter = [&](const Character& c) {
        if (activeChars < MAX_CHARACTERS) {
            memcpy(&charData.invBoundingBoxes[activeChars],
                   &c.boundingBox.inverseModelMatrix, sizeof(simd_float4x4));
            memcpy(&charData.invBodyParts[activeChars * 6 + 0],
                   &c.head.inverseModelMatrix, sizeof(simd_float4x4));
            memcpy(&charData.invBodyParts[activeChars * 6 + 1],
                   &c.trunk.inverseModelMatrix, sizeof(simd_float4x4));
            memcpy(&charData.invBodyParts[activeChars * 6 + 2],
                   &c.leftArm.inverseModelMatrix, sizeof(simd_float4x4));
            memcpy(&charData.invBodyParts[activeChars * 6 + 3],
                   &c.rightArm.inverseModelMatrix, sizeof(simd_float4x4));
            memcpy(&charData.invBodyParts[activeChars * 6 + 4],
                   &c.leftLeg.inverseModelMatrix, sizeof(simd_float4x4));
            memcpy(&charData.invBodyParts[activeChars * 6 + 5],
                   &c.rightLeg.inverseModelMatrix, sizeof(simd_float4x4));
            activeChars++;
        }
    };
    
    appendCharacter(character);
    for (const auto& npc : State::state.otherCharacters) {
        appendCharacter(npc);
    }
    charData.numCharacters = activeChars;
    
    // Upload character data to GPU
    CUDA_CHECK(cudaMemcpyAsync(_characterBuffer, &charData, sizeof(CharacterGPUData),
                               cudaMemcpyHostToDevice, _cudaStream));
    
    // Upload constant data
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_camera, &camData, sizeof(CameraData),
                                       0, cudaMemcpyHostToDevice, _cudaStream));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_frame, &frameData, sizeof(FrameData),
                                       0, cudaMemcpyHostToDevice, _cudaStream));
    
    // Grid and block sizes
    dim3 gridSizeFull((_width + 15) / 16, (_height + 15) / 16);
    dim3 gridSizeHalf((_width / 2 + 7) / 8, (_height / 2 + 7) / 8);
    dim3 groupSize16(16, 16);
    dim3 groupSize8(8, 8);
    
    // ============================================================================
    // PASS 0: Distance Approximation (half-res, 8x8 threads)
    // ============================================================================
    distApproximationKernel<<<gridSizeHalf, groupSize8, 0, _cudaStream>>>(
        _halfDistTexture.surface,
        camData, frameData,
        _materialMap.GetIndirectionPtr(),
        (SectorInfo*)_materialMap.GetSectorBufferPtr(),
        (uint64_t*)_materialMap.GetOccupancyPtr(),
        (uint8_t*)_materialMap.GetDataPtr(),
        (uint64_t*)_materialMap.GetSectorMaskPtr(),
        charData,
        _width, _height
    );
    
    // ============================================================================
    // PASS 1: GBuffer + Direct Light (full-res, 16x16 threads)
    // ============================================================================
    GBufferAndDirectLight<<<gridSizeFull, groupSize16, 0, _cudaStream>>>(
        _texDirectLight.surface,
        _texAlbedo.surface,
        _texNormal.surface,
        _texMotion.surface,
        _texDepth[currIdx].surface,
        camData, frameData,
        _materialMap.GetIndirectionPtr(),
        (SectorInfo*)_materialMap.GetSectorBufferPtr(),
        (uint64_t*)_materialMap.GetOccupancyPtr(),
        (uint8_t*)_materialMap.GetDataPtr(),
        (uint64_t*)_materialMap.GetSectorMaskPtr(),
        charData,
        _texturepack.getTextureObject(),
        _halfDistTexture.texture,
        _width, _height
    );
    
    // ============================================================================
    // PASS 2: Indirect Bounce (full-res)
    // ============================================================================
    IndirectBounce<<<gridSizeFull, groupSize16, 0, _cudaStream>>>(
        _texRawIndirect.surface,
        _texNormal.texture,
        _texDepth[currIdx].texture,
        camData, frameData,
        _texturepack.getTextureObject(),
        _materialMap.GetIndirectionPtr(),
        (SectorInfo*)_materialMap.GetSectorBufferPtr(),
        (uint64_t*)_materialMap.GetOccupancyPtr(),
        (uint8_t*)_materialMap.GetDataPtr(),
        (uint64_t*)_materialMap.GetSectorMaskPtr(),
        charData,
        _width, _height
    );
    
    // ============================================================================
    // PASS 3: Temporal Accumulation
    // ============================================================================
    TemporalAccumulation<<<gridSizeFull, groupSize16, 0, _cudaStream>>>(
        _texAccum[currIdx].surface,
        _texRawIndirect.texture,
        _texAccum[prevIdx].texture,
        _texMotion.texture,
        _texDepth[currIdx].texture,
        _texDepth[prevIdx].texture,
        _texDirectLight.texture,
        _width, _height
    );
    
    // ============================================================================
    // PASS 4: Bilateral Denoise (3 iterations: step 1, 2, 4)
    // ============================================================================
    // Iteration 1: step = 1
    BilateralDenoise<<<gridSizeFull, groupSize16, 0, _cudaStream>>>(
        _texDenoiseTemp.surface,
        _texAccum[currIdx].texture,
        _texNormal.texture,
        _texDepth[currIdx].texture,
        1,
        _width, _height
    );
    
    // Iteration 2: step = 2
    BilateralDenoise<<<gridSizeFull, groupSize16, 0, _cudaStream>>>(
        _texDenoised.surface,
        _texDenoiseTemp.texture,
        _texNormal.texture,
        _texDepth[currIdx].texture,
        2,
        _width, _height
    );
    
    // Iteration 3: step = 4
    BilateralDenoise<<<gridSizeFull, groupSize16, 0, _cudaStream>>>(
        _texDenoiseTemp.surface,
        _texDenoised.texture,
        _texNormal.texture,
        _texDepth[currIdx].texture,
        4,
        _width, _height
    );
    
    // ============================================================================
    // PASS 5: Volumetric Fog (half-res)
    // ============================================================================
    VolumetricFog<<<gridSizeHalf, groupSize8, 0, _cudaStream>>>(
        _texVolumetric[currIdx].surface,
        _texDepth[currIdx].texture,
        _texVolumetric[prevIdx].texture,
        camData, frameData,
        _materialMap.GetIndirectionPtr(),
        (SectorInfo*)_materialMap.GetSectorBufferPtr(),
        (uint64_t*)_materialMap.GetOccupancyPtr(),
        (uint64_t*)_materialMap.GetSectorMaskPtr(),
        charData,
        _width, _height
    );
    
    // ============================================================================
    // PASS 6: Compute Exposure (reduction kernel)
    // ============================================================================
    dim3 singleGroup(16, 16);
    ComputeExposure<<<singleGroup, singleGroup, 0, _cudaStream>>>(
        (ExposureData*)_exposureBuffer,
        frameData,
        _texDirectLight.texture,
        _texAccum[currIdx].texture,
        _texAlbedo.texture,
        _width, _height
    );
    
    // ============================================================================
    // PASS 7: Composite (full-res)
    // ============================================================================
    Composite<<<gridSizeFull, groupSize16, 0, _cudaStream>>>(
        _texCompositeResult.surface,
        _texDirectLight.texture,
        _texDenoiseTemp.texture,  // Output from final denoise iteration
        _texAlbedo.texture,
        _texDepth[currIdx].texture,
        _texVolumetric[currIdx].texture,
        (ExposureData*)_exposureBuffer,
        _width, _height
    );
    
    // Store jitter for next frame (for DLSS)
    _jitterX = character.jitterX;
    _jitterY = character.jitterY;
    
    // Update frame state
    _frameIndex++;
    const_cast<Character&>(character).lastRenderedViewProjectionMatrix =
        character.unjitteredViewProjectionMatrix;
}

// ============================================================================
// POST-DRAW: Copy to D3D12/DLSS (called after Draw, before present)
// ============================================================================

void CudaRenderer::PostDraw(cudaSurfaceObject_t outputSurface, 
                            uint32_t width, uint32_t height,
                            bool useDLSS) {
    (void)useDLSS;  // DLSS handled in win32_main.cpp via Streamline
    
    // Copy composite result to output surface (which can be D3D12 interop)
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    dim3 blockSize(16, 16);
    
    CopyToInteropKernel<<<gridSize, blockSize, 0, _cudaStream>>>(
        _texCompositeResult.surface,
        outputSurface,
        width, height
    );
    
    CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
}

// ============================================================================
// KERNEL IMPLEMENTATIONS (will be overridden by preprocessed shaders)
// ============================================================================

// These are stub implementations that ensure compilation.
// They will be replaced by the preprocessed shader files.

__global__ void CopyToInteropKernel(
    cudaSurfaceObject_t src,
    cudaSurfaceObject_t dst,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float4 color;
    surf2Dread(&color, src, x * sizeof(float4), y);
    
    // Simple copy (formats should match)
    surf2Dwrite(color, dst, x * sizeof(float4), y);
}
