#pragma once

#include "renderer/Renderer.hpp"
#include "renderer/Buffer.hpp" 

#include <cstdint> // For uint32_t
#include <memory> 

// This preprocessor check is the key to cross-compatibility.
#ifdef __OBJC__
// --- Objective-C++ Context ---
// When a .mm file includes this header, the compiler knows what these are.
@protocol MTLComputePipelineState;
@protocol MTLTexture;
@protocol MTLDevice;
@protocol MTLCommandQueue;
#else
// --- Pure C++ Context ---
// When a .cpp file includes this header, we define these as opaque pointers.
// This allows C++ code to know that the type exists without needing
// to know its internal Objective-C details.
typedef void* id;
#endif

class Character; // Forward declaration

class MetalRenderer : public Renderer
{
public:
    // The constructor now correctly uses the opaque 'id' type, which
    // is compatible with both C++ (as void*) and Objective-C++.
    MetalRenderer(id device);
    ~MetalRenderer() override;

    // The main entry point to kick off a frame's compute work.
    void Draw(const Character& character, unsigned int frameCount) override;

    // Called by the main view when it needs a texture to display.
    id GetOutputTexture();
    void GenerateWorld();
    
    // Called if the window size changes.
    void OnResize(uint32_t newWidth, uint32_t newHeight);

private:
    void createRenderTarget(uint32_t width, uint32_t height);

    id _device;
    id _computePSO;
    id _renderTargetTexture; // Our writable texture
    std::unique_ptr<Buffer> _voxelBuffer;
    id _generationPSO; 
};