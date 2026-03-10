#ifdef _WIN32
#include "renderer/CudaRender.cuh"
#include "renderer/D3D12Device.hpp"
#include "kernels/cuda_kernels.cuh"
#include "State.hpp"
#include "Character.hpp"
#include "cuda_fp16.h"
#include <iostream>

CudaRenderer::CudaRenderer() : texturepack(), materialMap() {
    cudaStreamCreate((cudaStream_t*)&commandQueue);
    
    cudaMalloc(&exposureBuffer, sizeof(ExposureData));
    ExposureData expData;
    expData.sceneLuminance = 0.5f;
    cudaMemcpy(exposureBuffer, &expData, sizeof(ExposureData), cudaMemcpyHostToDevice);
    
    cudaMalloc(&characterBuffer, sizeof(CharacterGPUData));
    
    std::cout << "Starting Dynamic World Generation (CudaMaterialMap)..." << std::endl;
    materialMap.GenerateDynamic();
    std::cout << "World Generation Complete." << std::endl;
    
    createRenderTargets(State::dispWIDTH, State::dispHEIGHT);
}

CudaRenderer::~CudaRenderer() {
    cudaFree(exposureBuffer);
    cudaFree(characterBuffer);
    cudaStreamDestroy((cudaStream_t)commandQueue);
}

void CudaRenderer::createRenderTargets(uint32_t width, uint32_t height) {
    auto initCudaArray = [&](std::unique_ptr<CudaD3D12Texture>& tex, uint32_t w, uint32_t h, const cudaChannelFormatDesc& desc, cudaTextureFilterMode filter = cudaFilterModePoint) {
        if (!tex) tex = std::make_unique<CudaD3D12Texture>();
        tex->Initialize_Cuda_Array(w, h, desc, filter);
    };

    auto initD3D12Shared = [&](std::unique_ptr<CudaD3D12Texture>& tex, uint32_t w, uint32_t h, DXGI_FORMAT format) {
        if (!tex) tex = std::make_unique<CudaD3D12Texture>();
        auto d3d12dev = (ID3D12Device*)(State::state.graphicsDevice->GetD3D12Device());
        tex->Initialize(d3d12dev, w, h, format, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, L"InteropTexture");
    };

    auto half4fmt = cudaCreateChannelDesc<half4>();
    auto float1fmt = cudaCreateChannelDesc<float>();
    auto half2fmt = cudaCreateChannelDesc<half2>();
    auto uchar4fmt = cudaCreateChannelDesc<uchar4>();
    auto char4fmt = cudaCreateChannelDesc<char4>(); // Signed normal

    initCudaArray(directLightTex, width, height, half4fmt);
    initCudaArray(albedoTex, width, height, uchar4fmt);
    initCudaArray(normalTex, width, height, char4fmt);
    initCudaArray(motionTex, width, height, half2fmt);
    initCudaArray(rawIndirectTex, width, height, half4fmt);
    initCudaArray(denoisedTex, width, height, half4fmt);
    initCudaArray(finalTex, width, height, uchar4fmt);
    initCudaArray(denoiseTempTex, width, height, half4fmt);
    initCudaArray(compositeResultTex, width, height, half4fmt, cudaFilterModeLinear);
    
    for (int i = 0; i < 2; i++) {
        initCudaArray(depthTex[i], width, height, float1fmt);
        initCudaArray(accumTex[i], width, height, half4fmt);
        
        // finalHistory needs to be accessible by D3D12 swapchain (platform code will copy/blit from it)
        initD3D12Shared(finalHistoryTex[i], width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    }

    initCudaArray(halfDistTex, width / 2, height / 2, float1fmt);
    initCudaArray(volumetricTex[0], width / 2, height / 2, half4fmt, cudaFilterModeLinear);
    initCudaArray(volumetricTex[1], width / 2, height / 2, half4fmt, cudaFilterModeLinear);
}

void CudaRenderer::Draw(const Character& character, unsigned int frameCount) {
    if (finalTex->getWidth() != State::dispWIDTH || finalTex->getHeight() != State::dispHEIGHT) {
        createRenderTargets(State::dispWIDTH, State::dispHEIGHT);
    }
    
    uint32_t width = State::dispWIDTH;
    uint32_t height = State::dispHEIGHT;
    
    int currIdx = frameIndex % 2;
    int prevIdx = (frameIndex + 1) % 2;

    CameraData camData;
    camData.position = {(float)character.position.x, (float)character.position.y, (float)character.position.z};
    camData.forward = {(float)character.direction.x, (float)character.direction.y, (float)character.direction.z};

    float tanHalfFov = tan(glm::radians(character.FOV) * 0.5f);
    float aspect = (float)width / (float)height;
    glm::vec3 sRight = character.camera.right * tanHalfFov * aspect;
    glm::vec3 sUp = character.camera.up * tanHalfFov;

    camData.right = {sRight.x, sRight.y, sRight.z};
    camData.up = {sUp.x, sUp.y, sUp.z};
    camData.jitter = {character.jitterX, character.jitterY};
    memcpy(&camData.unjitteredViewProjection, &character.unjitteredViewProjectionMatrix, 64);
    memcpy(&camData.prevUnjitteredViewProjection, &character.lastRenderedViewProjectionMatrix, 64);

    FrameData frameData;
    frameData.sunDirection = normalize(make_float3(10.f, 5.f, -4.f));
    double time = glfwGetTime(); // Use glfwGetTime or similar
    frameData.time = (float)fmod(time, 3600.0);
    static double lastTime = time;
    frameData.deltaTime = max((float)(time - lastTime), 0.001f);
    lastTime = time;

    float3 camPosCuda = make_float3((float)character.position.x, (float)character.position.y, (float)character.position.z);
    bool sectorsChanged = materialMap.UpdateStreaming(camPosCuda);
    frameData.worldOrigin = materialMap.GetWorldOrigin();

    // Populate CharacterGPUData
    CharacterGPUData charDataCpu;
    memset(&charDataCpu, 0, sizeof(CharacterGPUData));
    int activeChars = 0;
    
    auto appendCharacter = [&](const Character& c) {
        if (activeChars < MAX_CHARACTERS) {
            memcpy(&charDataCpu.invBoundingBoxes[activeChars], &c.boundingBox.inverseModelMatrix, 64);
            memcpy(&charDataCpu.invBodyParts[activeChars * 6 + 0], &c.head.inverseModelMatrix, 64);
            memcpy(&charDataCpu.invBodyParts[activeChars * 6 + 1], &c.body.inverseModelMatrix, 64);
            memcpy(&charDataCpu.invBodyParts[activeChars * 6 + 2], &c.arm_l.inverseModelMatrix, 64);
            memcpy(&charDataCpu.invBodyParts[activeChars * 6 + 3], &c.arm_r.inverseModelMatrix, 64);
            memcpy(&charDataCpu.invBodyParts[activeChars * 6 + 4], &c.leg_l.inverseModelMatrix, 64);
            memcpy(&charDataCpu.invBodyParts[activeChars * 6 + 5], &c.leg_r.inverseModelMatrix, 64);
            activeChars++;
        }
    };
    
    appendCharacter(character);
    for (const auto& c : State::state.otherCharacters) {
        appendCharacter(c);
    }
    charDataCpu.numCharacters = activeChars;
    cudaMemcpyAsync(characterBuffer, &charDataCpu, sizeof(CharacterGPUData), cudaMemcpyHostToDevice, (cudaStream_t)commandQueue);

    update_constant_memory(camData, frameData, characterBuffer, sizeof(CharacterGPUData));

    cudaStream_t stream = (cudaStream_t)commandQueue;
    
    // Pass 0: dist_approximation
    launch_distApproximationKernel(stream, width / 2, height / 2, halfDistTex->getCudaSurfObject(), frameData.worldOrigin, (const uint32_t*)materialMap.GetIndirectionTexture(), (const SectorInfo*)materialMap.GetSectorBuffer(), (const uint64_t*)materialMap.GetOccupancyBuffer(), (const uint8_t*)materialMap.GetDataBuffer(), (const uint64_t*)materialMap.GetSectorMaskBuffer());

    // Pass 1: GBuffer + Direct Light
    launch_GBufferAndDirectLight(stream, width, height, halfDistTex->getCudaTexObject(), directLightTex->getCudaSurfObject(), albedoTex->getCudaSurfObject(), normalTex->getCudaSurfObject(), motionTex->getCudaSurfObject(), depthTex[currIdx]->getCudaSurfObject(), frameData.worldOrigin, (const uint32_t*)materialMap.GetIndirectionTexture(), (const SectorInfo*)materialMap.GetSectorBuffer(), (const uint64_t*)materialMap.GetOccupancyBuffer(), (const uint8_t*)materialMap.GetDataBuffer(), (const uint64_t*)materialMap.GetSectorMaskBuffer(), texturepack.getTextureObject());

    // Pass 2: Indirect Bounce
    launch_IndirectBounce(stream, width, height, normalTex->getCudaTexObject(), depthTex[currIdx]->getCudaTexObject(), rawIndirectTex->getCudaSurfObject(), frameData.worldOrigin, (const uint32_t*)materialMap.GetIndirectionTexture(), (const SectorInfo*)materialMap.GetSectorBuffer(), (const uint64_t*)materialMap.GetOccupancyBuffer(), (const uint8_t*)materialMap.GetDataBuffer(), (const uint64_t*)materialMap.GetSectorMaskBuffer(), texturepack.getTextureObject());

    // Pass 3: Temporal Accumulation
    launch_TemporalAccumulation(stream, width, height, rawIndirectTex->getCudaTexObject(), directLightTex->getCudaTexObject(), motionTex->getCudaTexObject(), depthTex[currIdx]->getCudaTexObject(), depthTex[prevIdx]->getCudaTexObject(), accumTex[prevIdx]->getCudaTexObject(), accumTex[currIdx]->getCudaSurfObject(), sectorsChanged);

    // Pass 4: Bilateral Denoise (3 iterations)
    for (int iter = 0; iter < 3; iter++) {
        float stepWidth = 1.0f * (1 << iter);
        cudaTextureObject_t d_in = (iter == 0) ? accumTex[currIdx]->getCudaTexObject() : denoiseTempTex->getCudaTexObject();
        cudaSurfaceObject_t d_out = (iter == 2) ? denoisedTex->getCudaSurfObject() : denoiseTempTex->getCudaSurfObject();
        launch_BilateralDenoise(stream, width, height, d_in, normalTex->getCudaTexObject(), depthTex[currIdx]->getCudaTexObject(), d_out, stepWidth);
    }

    // Pass 5: Volumetric Fog
    launch_VolumetricFog(stream, width / 2, height / 2, depthTex[currIdx]->getCudaTexObject(), volumetricTex[prevIdx]->getCudaTexObject(), volumetricTex[currIdx]->getCudaSurfObject(), frameData.worldOrigin, (const uint32_t*)materialMap.GetIndirectionTexture(), (const SectorInfo*)materialMap.GetSectorBuffer(), (const uint64_t*)materialMap.GetOccupancyBuffer(), (const uint8_t*)materialMap.GetDataBuffer(), (const uint64_t*)materialMap.GetSectorMaskBuffer());

    // Pass 6: Compute Exposure
    launch_ComputeExposure(stream, width, height, directLightTex->getCudaTexObject(), accumTex[currIdx]->getCudaTexObject(), albedoTex->getCudaTexObject(), exposureBuffer);

    // Pass 7: Composite
    launch_Composite(stream, width, height, directLightTex->getCudaTexObject(), albedoTex->getCudaTexObject(), denoisedTex->getCudaTexObject(), volumetricTex[currIdx]->getCudaTexObject(), finalHistoryTex[currIdx]->getCudaSurfObject(), compositeResultTex->getCudaSurfObject(), exposureBuffer);

    frameIndex++;
}

void* CudaRenderer::GetOutputTexture() {
    int currIdx = frameIndex % 2;
    if (finalHistoryTex[currIdx]) {
        return (void*)finalHistoryTex[currIdx]->GetD3D12Resource();
    }
    return nullptr;
}

#endif
