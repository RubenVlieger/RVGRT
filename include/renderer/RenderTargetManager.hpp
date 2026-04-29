#pragma once

#include "renderer/RendererTraits.hpp"
#include <vector>
#include <array>
#include <string>

/**
 * RenderTargetManager - Unified render target management
 * 
 * Template-based manager following the BrickPool pattern.
 * Handles creation, destruction, and resizing of all render targets.
 * 
 * Supported targets:
 * - DirectLight (RGBA16Float, full-res)
 * - Albedo (RGBA8Unorm, full-res)
 * - Normal (RGBA8Snorm, full-res)
 * - Motion (RG16Float, full-res)
 * - RawIndirect (RGBA16Float, full-res)
 * - Denoised (RGBA16Float, full-res)
 * - Final (RGBA8Unorm, full-res)
 * - DenoiseTemp (RGBA16Float, full-res)
 * - CompositeResult (RGBA16Float, full-res)
 * - Depth[2] (R32Float, full-res, ping-pong)
 * - Accum[2] (RGBA16Float, full-res, ping-pong)
 * - FinalHistory[2] (RGBA16Float, full-res, ping-pong)
 * - Volumetric[2] (RGBA16Float, half-res, ping-pong)
 * - HalfDist (R32Float, half-res)
 */

template<typename Traits>
class RenderTargetManager {
public:
    using RenderTarget = typename Traits::RenderTarget;
    using Device = typename Traits::Device;
    
    // Indices for ping-pong buffers
    static constexpr int CURRENT = 0;
    static constexpr int PREVIOUS = 1;
    
    RenderTargetManager(Device device) 
        : _device(device)
        , _width(0)
        , _height(0) 
    {
    }
    
    ~RenderTargetManager() {
        DestroyAllTargets();
    }
    
    /**
     * Create all render targets for a given resolution
     */
    void CreateTargets(uint32_t width, uint32_t height) {
        _width = width;
        _height = height;
        
        // Full-resolution targets
        _directLight = Traits::CreateRenderTarget(_device, width, height, 
                                                   TextureFormat::RGBA16Float, "DirectLight");
        _albedo = Traits::CreateRenderTarget(_device, width, height,
                                              TextureFormat::RGBA8Unorm, "Albedo");
        _normal = Traits::CreateRenderTarget(_device, width, height,
                                              TextureFormat::RGBA8Snorm, "Normal");
        _motion = Traits::CreateRenderTarget(_device, width, height,
                                              TextureFormat::RG16Float, "Motion");
        _rawIndirect = Traits::CreateRenderTarget(_device, width, height,
                                                   TextureFormat::RGBA16Float, "RawIndirect");
        _denoised = Traits::CreateRenderTarget(_device, width, height,
                                                TextureFormat::RGBA16Float, "Denoised");
        _final = Traits::CreateRenderTarget(_device, width, height,
                                             TextureFormat::RGBA8Unorm, "Final");
        _denoiseTemp = Traits::CreateRenderTarget(_device, width, height,
                                                   TextureFormat::RGBA16Float, "DenoiseTemp");
        _compositeResult = Traits::CreateRenderTarget(_device, width, height,
                                                       TextureFormat::RGBA16Float, "CompositeResult");
        
        // Ping-pong buffers
        for (int i = 0; i < 2; i++) {
            std::string suffix = std::to_string(i);
            _depth[i] = Traits::CreateRenderTarget(_device, width, height,
                                                    TextureFormat::R32Float, 
                                                    ("Depth_" + suffix).c_str());
            _accum[i] = Traits::CreateRenderTarget(_device, width, height,
                                                    TextureFormat::RGBA16Float,
                                                    ("Accum_" + suffix).c_str());
            _finalHistory[i] = Traits::CreateRenderTarget(_device, width, height,
                                                           TextureFormat::RGBA16Float,
                                                           ("FinalHistory_" + suffix).c_str());
        }
        
        // Half-resolution targets
        uint32_t halfWidth = width / 2;
        uint32_t halfHeight = height / 2;
        
        for (int i = 0; i < 2; i++) {
            std::string suffix = std::to_string(i);
            _volumetric[i] = Traits::CreateRenderTarget(_device, halfWidth, halfHeight,
                                                         TextureFormat::RGBA16Float,
                                                         ("Volumetric_" + suffix).c_str());
        }
        
        _halfDist = Traits::CreateRenderTarget(_device, halfWidth, halfHeight,
                                                TextureFormat::R32Float, "HalfDist");
        
        _rawIndirectHalf = Traits::CreateRenderTarget(_device, halfWidth, halfHeight,
                                                       TextureFormat::RGBA16Float, "RawIndirectHalf");
    }
    
    /**
     * Destroy all render targets
     */
    void DestroyAllTargets() {
        if (_width == 0 || _height == 0) return;
        
        Traits::DestroyRenderTarget(_device, _directLight);
        Traits::DestroyRenderTarget(_device, _albedo);
        Traits::DestroyRenderTarget(_device, _normal);
        Traits::DestroyRenderTarget(_device, _motion);
        Traits::DestroyRenderTarget(_device, _rawIndirect);
        Traits::DestroyRenderTarget(_device, _rawIndirectHalf);
        Traits::DestroyRenderTarget(_device, _denoised);
        Traits::DestroyRenderTarget(_device, _final);
        Traits::DestroyRenderTarget(_device, _denoiseTemp);
        Traits::DestroyRenderTarget(_device, _compositeResult);
        
        for (int i = 0; i < 2; i++) {
            Traits::DestroyRenderTarget(_device, _depth[i]);
            Traits::DestroyRenderTarget(_device, _accum[i]);
            Traits::DestroyRenderTarget(_device, _finalHistory[i]);
            Traits::DestroyRenderTarget(_device, _volumetric[i]);
        }
        
        Traits::DestroyRenderTarget(_device, _halfDist);
        
        _width = 0;
        _height = 0;
    }
    
    /**
     * Recreate all targets at new resolution
     */
    void Resize(uint32_t newWidth, uint32_t newHeight) {
        if (_width == newWidth && _height == newHeight) return;
        
        DestroyAllTargets();
        CreateTargets(newWidth, newHeight);
    }
    
    // Getters for full-resolution targets
    RenderTarget& GetDirectLight() { return _directLight; }
    RenderTarget& GetAlbedo() { return _albedo; }
    RenderTarget& GetNormal() { return _normal; }
    RenderTarget& GetMotion() { return _motion; }
    RenderTarget& GetRawIndirect() { return _rawIndirect; }
    RenderTarget& GetRawIndirectHalf() { return _rawIndirectHalf; }
    RenderTarget& GetDenoised() { return _denoised; }
    RenderTarget& GetFinal() { return _final; }
    RenderTarget& GetDenoiseTemp() { return _denoiseTemp; }
    RenderTarget& GetCompositeResult() { return _compositeResult; }
    
    // Getters for ping-pong buffers (use CURRENT/PREVIOUS indices)
    RenderTarget& GetDepth(int index) { return _depth[index]; }
    RenderTarget& GetAccum(int index) { return _accum[index]; }
    RenderTarget& GetFinalHistory(int index) { return _finalHistory[index]; }
    RenderTarget& GetVolumetric(int index) { return _volumetric[index]; }
    
    // Getters for half-resolution targets
    RenderTarget& GetHalfDist() { return _halfDist; }
    
    // Resolution queries
    uint32_t GetWidth() const { return _width; }
    uint32_t GetHeight() const { return _height; }
    uint32_t GetHalfWidth() const { return _width / 2; }
    uint32_t GetHalfHeight() const { return _height / 2; }
    
private:
    Device _device;
    uint32_t _width;
    uint32_t _height;
    
    // Full-resolution targets
    RenderTarget _directLight;
    RenderTarget _albedo;
    RenderTarget _normal;
    RenderTarget _motion;
    RenderTarget _rawIndirect;
    RenderTarget _rawIndirectHalf;
    RenderTarget _denoised;
    RenderTarget _final;
    RenderTarget _denoiseTemp;
    RenderTarget _compositeResult;
    
    // Ping-pong buffers
    std::array<RenderTarget, 2> _depth;
    std::array<RenderTarget, 2> _accum;
    std::array<RenderTarget, 2> _finalHistory;
    std::array<RenderTarget, 2> _volumetric;
    
    // Half-resolution targets
    RenderTarget _halfDist;
};

// Type aliases for convenience
#if defined(__APPLE__)
using MetalRenderTargetManager = RenderTargetManager<RendererImpl::MetalRendererTraits>;
#elif defined(_WIN32)
using CudaRenderTargetManager = RenderTargetManager<RendererImpl::CudaRendererTraits>;
#endif
