// IMPORTANT: This file must be compiled as Objective-C++ (.mm) on macOS
// to ensure proper type definitions for Metal types (id = objc_object* vs void*)
#ifdef __APPLE__
#import <objc/objc.h>  // Ensure id is defined as objc_object* not void*
#endif

#include "renderer/RendererBase.hpp"
#include "State.hpp"

// ============================================================================
// Template implementation for RendererBase
// ============================================================================

template<typename Traits>
void RendererBase<Traits>::Initialize(uint32_t width, uint32_t height) {
    _width = width;
    _height = height;
    _outputWidth = width;
    _outputHeight = height;
    
    // Create render targets
    _renderTargetManager.CreateTargets(width, height);
    
    // Create GPU buffers (implemented by derived class)
    CreateExposureBuffer();
    CreateCharacterBuffer();
    
    // Create pipeline states (implemented by derived class)
    CreatePipelineStates();
}

template<typename Traits>
void RendererBase<Traits>::OnResize(uint32_t renderW, uint32_t renderH, uint32_t screenW, uint32_t screenH) {
    State::dispWIDTH = renderW;
    State::dispHEIGHT = renderH;
    State::screenWIDTH = screenW;
    State::screenHEIGHT = screenH;
    
    _renderTargetManager.Resize(renderW, renderH);
    _width = renderW;
    _height = renderH;
    _outputWidth = screenW;
    _outputHeight = screenH;
    _scalerNeedsReset = true;
}

template<typename Traits>
void RendererBase<Traits>::Shutdown() {
    // Destroy pipeline states
    DestroyPipelineStates();
    
    // Destroy buffers
    if (_exposureBuffer) {
        Traits::DestroyBuffer(_device, _exposureBuffer);
        _exposureBuffer = nullptr;
    }
    if (_characterBuffer) {
        Traits::DestroyBuffer(_device, _characterBuffer);
        _characterBuffer = nullptr;
    }
    
    // Render targets are automatically destroyed by the manager's destructor
}

template<typename Traits>
typename RendererBase<Traits>::Texture RendererBase<Traits>::GetOutputTexture() {
    return _renderTargetManager.GetFinal().texture;
}

template<typename Traits>
void RendererBase<Traits>::ExecutePipeline(CommandBuffer cmdBuf, const Character& character) {
    // NOTE: This method is disabled until Phase 4 (MaterialMap abstraction) is complete.
    // MetalRenderer and CudaRenderer currently use their own Draw() implementations.
    // This method will be enabled once MaterialMap is properly abstracted.
    (void)cmdBuf;
    (void)character;
}

// ============================================================================
// Individual Pass Implementations
// ============================================================================

// NOTE: Pass implementations are disabled until Phase 4 (MaterialMap abstraction).
// MetalRenderer and CudaRenderer use their own Draw() implementations.
// These methods will be enabled once MaterialMap is properly abstracted.

template<typename Traits>
void RendererBase<Traits>::Pass0_DistApproximation(ComputeEncoder encoder, int currIdx) {
    (void)encoder;
    (void)currIdx;
}

template<typename Traits>
void RendererBase<Traits>::Pass1_GBuffer(ComputeEncoder encoder, int currIdx) {
    (void)encoder;
    (void)currIdx;
}

template<typename Traits>
void RendererBase<Traits>::Pass2_Indirect(ComputeEncoder encoder, int currIdx) {
    (void)encoder;
    (void)currIdx;
}

template<typename Traits>
void RendererBase<Traits>::Pass3_Accumulation(ComputeEncoder encoder, int currIdx, int prevIdx) {
    (void)encoder;
    (void)currIdx;
    (void)prevIdx;
}

template<typename Traits>
void RendererBase<Traits>::Pass4_Denoise(ComputeEncoder encoder, int currIdx) {
    (void)encoder;
    (void)currIdx;
}

template<typename Traits>
void RendererBase<Traits>::Pass5_Volumetric(ComputeEncoder encoder, int currIdx, int prevIdx) {
    (void)encoder;
    (void)currIdx;
    (void)prevIdx;
}

template<typename Traits>
void RendererBase<Traits>::Pass6_Exposure(ComputeEncoder encoder, int currIdx) {
    (void)encoder;
    (void)currIdx;
}

template<typename Traits>
void RendererBase<Traits>::Pass7_Composite(ComputeEncoder encoder, int currIdx) {
    (void)encoder;
    (void)currIdx;
}

// ============================================================================
// Explicit template instantiations
// ============================================================================

// Explicitly instantiate for Metal (macOS)
#if defined(__APPLE__)
template class RendererBase<RendererImpl::MetalRendererTraits>;
#endif

// Explicitly instantiate for CUDA (Windows)
#if defined(_WIN32)
template class RendererBase<RendererImpl::CudaRendererTraits>;
#endif
