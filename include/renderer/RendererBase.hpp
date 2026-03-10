#pragma once

#include "renderer/Renderer.hpp"
#include "renderer/RendererTraits.hpp"
#include "renderer/RenderTargetManager.hpp"
#include "renderer/FrameDataManager.hpp"
#include "renderer/ShaderTypes.h"
#include "Character.hpp"
#include "Texturepack.h"
#include <cstdint>
#include <memory>

// Forward declaration
class MaterialMap;

/**
 * RendererBase - Template base class for unified renderer implementations
 * 
 * This class extracts the common pipeline logic from MetalRenderer and CudaRenderer.
 * Platform-specific implementations inherit from this and provide their Traits.
 * 
 * Usage:
 *   class MetalRenderer : public RendererBase<MetalRendererTraits> { ... };
 *   class CudaRenderer : public RendererBase<CudaRendererTraits> { ... };
 *   class WebRenderer : public RendererBase<WebRendererTraits> { ... };
 * 
 * The Traits class provides:
 * - Type aliases for platform-specific handles
 * - Static methods for platform operations
 * - RenderTarget structure definition
 */

template<typename Traits>
class RendererBase : public Renderer {
public:
    using Device = typename Traits::Device;
    using CommandBuffer = typename Traits::CommandBuffer;
    using ComputeEncoder = typename Traits::ComputeEncoder;
    using PipelineState = typename Traits::PipelineState;
    using Texture = typename Traits::Texture;
    using Buffer = typename Traits::Buffer;
    using Scaler = typename Traits::Scaler;
    using RenderTarget = typename Traits::RenderTarget;
    using MaterialMapType = typename Traits::MaterialMapType;
    
    // Constructor
    RendererBase(Device device) 
        : _device(device)
        , _renderTargetManager(device)
        , _frameIndex(0)
        , _width(0)
        , _height(0)
        , _scalerNeedsReset(true)
    {
    }
    
    // Destructor
    virtual ~RendererBase() = default;
    
    // Delete copy/move
    RendererBase(const RendererBase&) = delete;
    RendererBase& operator=(const RendererBase&) = delete;
    RendererBase(RendererBase&&) = delete;
    RendererBase& operator=(RendererBase&&) = delete;

    // =========================================================================
    // Common interface methods (to be called by platform-specific implementations)
    // =========================================================================
    
    /**
     * Initialize the renderer - create render targets and buffers
     * Called once during construction
     */
    void Initialize(uint32_t width, uint32_t height);
    
    /**
     * Clean up resources
     */
    void Shutdown();
    
    /**
     * Handle window resize
     */
    void OnResize(uint32_t newWidth, uint32_t newHeight);
    
    /**
     * Reset the temporal scaler (call on teleport/scene change)
     */
    void ResetScaler() { _scalerNeedsReset = true; }
    
    /**
     * Get the output texture (for presentation)
     */
    Texture GetOutputTexture();
    
    /**
     * Get current frame index
     */
    uint32_t GetFrameIndex() const { return _frameIndex; }
    
    // =========================================================================
    // Platform-specific interface (must be implemented by derived classes)
    // =========================================================================
    
    /**
     * Create platform-specific pipeline states (kernels)
     * Called during initialization
     */
    virtual void CreatePipelineStates() = 0;
    
    /**
     * Destroy platform-specific pipeline states
     */
    virtual void DestroyPipelineStates() = 0;
    

    
    /**
     * Get the texture pack
     */
    virtual Texturepack& GetTexturePack() = 0;
    
    /**
     * Get the temporal scaler
     */
    virtual Scaler GetTemporalScaler() const = 0;
    
    /**
     * Create the exposure buffer
     */
    virtual void CreateExposureBuffer() = 0;
    
    /**
     * Create the character buffer
     */
    virtual void CreateCharacterBuffer() = 0;
    
    /**
     * Upload constant data to GPU (camera, frame, characters)
     */
    virtual void UploadConstantData(CommandBuffer cmdBuf, 
                                    const CameraData& camera,
                                    const FrameData& frame,
                                    const CharacterGPUData& characters) = 0;
    
    /**
     * Execute the complete rendering pipeline
     * This is the main method that orchestrates all passes
     */
    void ExecutePipeline(CommandBuffer cmdBuf, const Character& character);
    
    // =========================================================================
    // Individual pass execution (called from ExecutePipeline)
    // =========================================================================
    
protected:
    // Individual pass methods - each dispatches one kernel
    void Pass0_DistApproximation(ComputeEncoder encoder, int currIdx);
    void Pass1_GBuffer(ComputeEncoder encoder, int currIdx);
    void Pass2_Indirect(ComputeEncoder encoder, int currIdx);
    void Pass3_Accumulation(ComputeEncoder encoder, int currIdx, int prevIdx);
    void Pass4_Denoise(ComputeEncoder encoder, int currIdx);
    void Pass5_Volumetric(ComputeEncoder encoder, int currIdx, int prevIdx);
    void Pass6_Exposure(ComputeEncoder encoder, int currIdx);
    void Pass7_Composite(ComputeEncoder encoder, int currIdx);
    
    // =========================================================================
    // Helper methods
    // =========================================================================
    
    /**
     * Get current and previous frame indices for ping-pong buffers
     */
    std::pair<int, int> GetFrameIndices() const {
        int currIdx = _frameIndex % 2;
        int prevIdx = (_frameIndex + 1) % 2;
        return {currIdx, prevIdx};
    }
    
    /**
     * Get grid sizes for kernel dispatch
     */
    GridSize GetFullResGrid() const {
        return {_width, _height, 1};
    }
    
    GridSize GetHalfResGrid() const {
        return {_width / 2, _height / 2, 1};
    }
    
    static constexpr GroupSize GROUP_SIZE_16 = {16, 16, 1};
    static constexpr GroupSize GROUP_SIZE_8 = {8, 8, 1};
    
    // =========================================================================
    // Member variables (accessible by derived classes)
    // =========================================================================
    
protected:
    // Platform device
    Device _device;
    
    // Render target manager
    RenderTargetManager<Traits> _renderTargetManager;
    
    // Frame data manager (shared data preparation)
    FrameDataManager _frameDataManager;
    
    // Pipeline states (to be set by derived classes)
    PipelineState _psoDistApprox = nullptr;
    PipelineState _psoGBuffer = nullptr;
    PipelineState _psoIndirect = nullptr;
    PipelineState _psoAccumulate = nullptr;
    PipelineState _psoDenoise = nullptr;
    PipelineState _psoComposite = nullptr;
    PipelineState _psoVolumetric = nullptr;
    PipelineState _psoExposure = nullptr;
    
    // GPU buffers (managed by derived classes via Traits)
    Buffer _exposureBuffer = nullptr;
    Buffer _characterBuffer = nullptr;
    
    // Frame state
    uint32_t _frameIndex;
    uint32_t _width;
    uint32_t _height;
    bool _scalerNeedsReset;
    
    // Upscaling jitter (for DLSS/MetalFX)
    float _jitterX = 0.0f;
    float _jitterY = 0.0f;
};
