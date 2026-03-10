// ============================================================================
// CudaRenderer.cu - CUDA implementation mirroring MetalRenderer
// ============================================================================

#include "renderer/CudaRender.cuh"
#include "Character.hpp"
#include "State.hpp"
#include "cumath.h"
#include "renderer/ShaderTypes.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>
#include <cstring>

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
// ============================================================================

// These will be defined in the preprocessed shader files
// We declare them here as extern since they'll be included at the bottom

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

CudaRenderer::CudaRenderer() : _texturepack() {
    // Create CUDA stream for all rendering operations
    CUDA_CHECK(cudaStreamCreate(&_cudaStream));
    
    // Initialize exposure buffer with default value
    ExposureData initialExp = {0.5f};
    CUDA_CHECK(cudaMalloc(&_exposureBuffer, sizeof(ExposureData)));
    CUDA_CHECK(cudaMemcpy(_exposureBuffer, &initialExp, sizeof(ExposureData), cudaMemcpyHostToDevice));
    
    // Initialize character buffer
    CUDA_CHECK(cudaMalloc(&_characterBuffer, sizeof(CharacterGPUData)));
    
    // Generate world (exactly like MetalRenderer line 73-75)
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
    texDesc.filterMode = cudaFilterModeLinear;
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
    
    // Full resolution targets (matching MetalRenderer.hpp lines 64-70)
    allocateTarget(_texDirectLight, width, height, rgba16f);
    allocateTarget(_texAlbedo, width, height, rgba8);
    allocateTarget(_texNormal, width, height, rgba16f);
    allocateTarget(_texMotion, width, height, rg16f);
    allocateTarget(_texRawIndirect, width, height, rgba16f);
    allocateTarget(_texDenoised, width, height, rgba16f);
    allocateTarget(_texFinal, width, height, rgba8);
    allocateTarget(_texDenoiseTemp, width, height, rgba16f);
    allocateTarget(_texCompositeResult, width, height, rgba16f);
    
    // Ping-pong buffers (lines 72-76)
    for (int i = 0; i < 2; i++) {
        allocateTarget(_texDepth[i], width, height, r32f);
        allocateTarget(_texAccum[i], width, height, rgba16f);
        allocateTarget(_texFinalHistory[i], width, height, rgba16f);
    }
    
    // Volumetric buffers (half-res, lines 78-79)
    for (int i = 0; i < 2; i++) {
        allocateTarget(_texVolumetric[i], width / 2, height / 2, rgba16f);
    }
    
    // Half-resolution distance texture (line 86)
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
    State::screenHEIGHT = newWidth;
    State::screenWIDTH = newHeight;
    
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
// MAIN DRAW LOOP (mirroring MetalRenderer.mm lines 197-549)
// ============================================================================

void CudaRenderer::Draw(const Character& character, unsigned int frameCount) {
    int currIdx = _frameIndex % 2;
    int prevIdx = (_frameIndex + 1) % 2;
    
    // Lines 202-219: Prepare CameraData
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
    
    // Lines 221-227: Prepare FrameData
    FrameData frameData;
    frameData.sunDirection = normalize(make_float3(10.f, 5.f, -4.f));
    // Use system clock for time
    #ifdef _WIN32
    frameData.time = (float)(clock() % (CLOCKS_PER_SEC * 3600)) / CLOCKS_PER_SEC;
    #else
    frameData.time = (float)(clock() % (CLOCKS_PER_SEC * 3600)) / CLOCKS_PER_SEC;
    #endif
    frameData.deltaTime = 0.016f;  // TODO: Calculate actual delta time
    
    // Lines 229-236: Update material map streaming
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
    
    // Lines 238-268: Prepare character data
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
    
    // Grid and block sizes matching Metal
    dim3 gridSizeFull((_width + 15) / 16, (_height + 15) / 16);
    dim3 gridSizeHalf((_width / 2 + 7) / 8, (_height / 2 + 7) / 8);
    dim3 groupSize16(16, 16);
    dim3 groupSize8(8, 8);
    
    // ============================================================================
    // PASS 0: Distance Approximation (half-res, 8x8 threads)
    // Lines 283-305
    // ============================================================================
    {
        // Kernel arguments
        void* args[] = {
            &_halfDistTexture.surface,
            &_materialMap.GetIndirectionPtr(),
            &_materialMap.GetSectorBufferPtr(),
            &_materialMap.GetOccupancyPtr(),
            &_materialMap.GetDataPtr(),
            &_materialMap.GetSectorMaskPtr(),
            &_characterBuffer,
            &_width,
            &_height
        };
        
        // Launch kernel - kernel defined in preprocessed dist_approx.shader
        extern __global__ void distApproximationKernel(
            cudaSurfaceObject_t distSurf,
            const uint32_t* indirection,
            const SectorInfo* sectorBuffer,
            const uint64_t* occupancyBuffer,
            const uint8_t* dataBuffer,
            const uint64_t* sectorMaskBuffer,
            const CharacterGPUData* charData,
            int width,
            int height
        );
        
        CUDA_CHECK(cudaLaunchKernel(
            (const void*)distApproximationKernel,
            gridSizeHalf, groupSize8,
            args, 0, _cudaStream
        ));
        CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
    }
    
    // ============================================================================
    // PASS 1: GBuffer + Direct Light (full-res, 16x16 threads)
    // Lines 317-345
    // ============================================================================
    {
        void* args[] = {
            &_texDirectLight.surface,
            &_texAlbedo.surface,
            &_texNormal.surface,
            &_texMotion.surface,
            &_texDepth[currIdx].surface,
            &_halfDistTexture.texture,
            &_texturepack.getTextureObject(),
            &_materialMap.GetIndirectionPtr(),
            &_materialMap.GetSectorBufferPtr(),
            &_materialMap.GetOccupancyPtr(),
            &_materialMap.GetDataPtr(),
            &_materialMap.GetSectorMaskPtr(),
            &_characterBuffer,
            &_width,
            &_height
        };
        
        extern __global__ void GBufferAndDirectLight(
            cudaSurfaceObject_t texDirectLight,
            cudaSurfaceObject_t texAlbedo,
            cudaSurfaceObject_t texNormal,
            cudaSurfaceObject_t texMotion,
            cudaSurfaceObject_t texDepth,
            cudaTextureObject_t halfDistTex,
            cudaTextureObject_t textureAtlas,
            const uint32_t* indirection,
            const SectorInfo* sectorBuffer,
            const uint64_t* occupancyBuffer,
            const uint8_t* dataBuffer,
            const uint64_t* sectorMaskBuffer,
            const CharacterGPUData* charData,
            int width,
            int height
        );
        
        CUDA_CHECK(cudaLaunchKernel(
            (const void*)GBufferAndDirectLight,
            gridSizeFull, groupSize16,
            args, 0, _cudaStream
        ));
        CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
    }
    
    // ============================================================================
    // PASS 2: Indirect Bounce (full-res)
    // Lines 355-380
    // ============================================================================
    {
        void* args[] = {
            &_texRawIndirect.surface,
            &_texNormal.texture,
            &_texDepth[currIdx].texture,
            &_materialMap.GetIndirectionPtr(),
            &_materialMap.GetSectorBufferPtr(),
            &_materialMap.GetOccupancyPtr(),
            &_materialMap.GetDataPtr(),
            &_materialMap.GetSectorMaskPtr(),
            &_characterBuffer,
            &_width,
            &_height
        };
        
        extern __global__ void IndirectBounce(
            cudaSurfaceObject_t texRawIndirect,
            cudaTextureObject_t texNormal,
            cudaTextureObject_t texDepth,
            const uint32_t* indirection,
            const SectorInfo* sectorBuffer,
            const uint64_t* occupancyBuffer,
            const uint8_t* dataBuffer,
            const uint64_t* sectorMaskBuffer,
            const CharacterGPUData* charData,
            int width,
            int height
        );
        
        CUDA_CHECK(cudaLaunchKernel(
            (const void*)IndirectBounce,
            gridSizeFull, groupSize16,
            args, 0, _cudaStream
        ));
        CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
    }
    
    // ============================================================================
    // PASS 3: Temporal Accumulation
    // Lines 390-405
    // ============================================================================
    {
        void* args[] = {
            &_texAccum[currIdx].surface,
            &_texRawIndirect.texture,
            &_texAccum[prevIdx].texture,
            &_texMotion.texture,
            &_texDepth[currIdx].texture,
            &_texDepth[prevIdx].texture,
            &_texDirectLight.texture,
            &_width,
            &_height
        };
        
        extern __global__ void TemporalAccumulation(
            cudaSurfaceObject_t texAccum,
            cudaTextureObject_t texRawIndirect,
            cudaTextureObject_t texHistory,
            cudaTextureObject_t texMotion,
            cudaTextureObject_t texDepth,
            cudaTextureObject_t texPrevDepth,
            cudaTextureObject_t texDirect,
            int width,
            int height
        );
        
        CUDA_CHECK(cudaLaunchKernel(
            (const void*)TemporalAccumulation,
            gridSizeFull, groupSize16,
            args, 0, _cudaStream
        ));
        CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
    }
    
    // ============================================================================
    // PASS 4: Bilateral Denoise (3 iterations: step 1, 2, 4)
    // Lines 412-436
    // ============================================================================
    {
        extern __global__ void BilateralDenoise(
            cudaSurfaceObject_t output,
            cudaTextureObject_t input,
            cudaTextureObject_t texNormal,
            cudaTextureObject_t texDepth,
            int stepWidth,
            int width,
            int height
        );
        
        // Iteration 1: step = 1
        {
            int stepWidth = 1;
            void* args[] = {
                &_texDenoiseTemp.surface,
                &_texAccum[currIdx].texture,
                &_texNormal.texture,
                &_texDepth[currIdx].texture,
                &stepWidth,
                &_width,
                &_height
            };
            CUDA_CHECK(cudaLaunchKernel(
                (const void*)BilateralDenoise,
                gridSizeFull, groupSize16,
                args, 0, _cudaStream
            ));
            CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
        }
        
        // Iteration 2: step = 2
        {
            int stepWidth = 2;
            void* args[] = {
                &_texDenoised.surface,
                &_texDenoiseTemp.texture,
                &_texNormal.texture,
                &_texDepth[currIdx].texture,
                &stepWidth,
                &_width,
                &_height
            };
            CUDA_CHECK(cudaLaunchKernel(
                (const void*)BilateralDenoise,
                gridSizeFull, groupSize16,
                args, 0, _cudaStream
            ));
            CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
        }
        
        // Iteration 3: step = 4
        {
            int stepWidth = 4;
            void* args[] = {
                &_texDenoiseTemp.surface,
                &_texDenoised.texture,
                &_texNormal.texture,
                &_texDepth[currIdx].texture,
                &stepWidth,
                &_width,
                &_height
            };
            CUDA_CHECK(cudaLaunchKernel(
                (const void*)BilateralDenoise,
                gridSizeFull, groupSize16,
                args, 0, _cudaStream
            ));
            CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
        }
    }
    
    // ============================================================================
    // PASS 5: Volumetric Fog (half-res)
    // Lines 448-468
    // ============================================================================
    {
        void* args[] = {
            &_texVolumetric[currIdx].surface,
            &_texDepth[currIdx].texture,
            &_texVolumetric[prevIdx].texture,
            &_materialMap.GetIndirectionPtr(),
            &_materialMap.GetSectorBufferPtr(),
            &_materialMap.GetSectorMaskPtr(),
            &_characterBuffer,
            &_width,
            &_height
        };
        
        extern __global__ void VolumetricFog(
            cudaSurfaceObject_t texVolumetric,
            cudaTextureObject_t texDepth,
            cudaTextureObject_t texHistory,
            const uint32_t* indirection,
            const SectorInfo* sectorBuffer,
            const uint64_t* sectorMaskBuffer,
            const CharacterGPUData* charData,
            int width,
            int height
        );
        
        CUDA_CHECK(cudaLaunchKernel(
            (const void*)VolumetricFog,
            gridSizeHalf, groupSize8,
            args, 0, _cudaStream
        ));
        CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
    }
    
    // ============================================================================
    // PASS 6: Compute Exposure (single workgroup)
    // Lines 478-486
    // ============================================================================
    {
        void* args[] = {
            &_exposureBuffer,
            &_texDirectLight.texture,
            &_texAccum[currIdx].texture,
            &_texAlbedo.texture,
            &_width,
            &_height
        };
        
        extern __global__ void ComputeExposure(
            ExposureData* exposure,
            cudaTextureObject_t texDirect,
            cudaTextureObject_t texAccum,
            cudaTextureObject_t texAlbedo,
            int width,
            int height
        );
        
        dim3 singleGroup(16, 16);
        CUDA_CHECK(cudaLaunchKernel(
            (const void*)ComputeExposure,
            singleGroup, singleGroup,
            args, 0, _cudaStream
        ));
        // No sync needed - runs parallel until composite
    }
    
    // ============================================================================
    // PASS 7: Composite (full-res)
    // Lines 496-506
    // ============================================================================
    {
        void* args[] = {
            &_texCompositeResult.surface,
            &_texDirectLight.texture,
            &_texDenoiseTemp.texture,  // Output from final denoise iteration
            &_texAlbedo.texture,
            &_texDepth[currIdx].texture,
            &_texVolumetric[currIdx].texture,
            &_exposureBuffer,
            &_width,
            &_height
        };
        
        extern __global__ void Composite(
            cudaSurfaceObject_t texFinal,
            cudaTextureObject_t texDirect,
            cudaTextureObject_t texAccum,
            cudaTextureObject_t texAlbedo,
            cudaTextureObject_t texDepth,
            cudaTextureObject_t texVolumetric,
            const ExposureData* exposure,
            int width,
            int height
        );
        
        CUDA_CHECK(cudaLaunchKernel(
            (const void*)Composite,
            gridSizeFull, groupSize16,
            args, 0, _cudaStream
        ));
        CUDA_CHECK(cudaStreamSynchronize(_cudaStream));
    }
    
    // Lines 546-548: Update frame state
    _frameIndex++;
    const_cast<Character&>(character).lastRenderedViewProjectionMatrix =
        character.unjitteredViewProjectionMatrix;
}

// ============================================================================
// KERNEL INCLUDES (preprocessed shader files)
// These will be generated by CMake from .shader sources
// ============================================================================

// Note: These includes are placeholders. The actual .cu files will be generated
// by CMake preprocessing the .shader files. For now, we declare the kernels
// as extern above and they will be linked from the preprocessed files.

// #include "cuda_kernels/tables.cu"
// #include "cuda_kernels/dist_approx.cu"
// #include "cuda_kernels/direct_light.cu"
// #include "cuda_kernels/indirect_bounce.cu"
// #include "cuda_kernels/temporal_acc.cu"
// #include "cuda_kernels/denoise.cu"
// #include "cuda_kernels/volumetric.cu"
// #include "cuda_kernels/exposure.cu"
// #include "cuda_kernels/composite.cu"
