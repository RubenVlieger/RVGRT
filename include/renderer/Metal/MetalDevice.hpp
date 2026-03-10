#pragma once

#include "renderer/GraphicsDevice.hpp"

// Forward declare Objective-C types only when compiling Objective-C++ code
#ifdef __OBJC__
#import <MetalKit/MetalKit.h> // Import full header here for delegate protocols
#else
// Opaque forward declarations for C++
typedef void* MTKView;
typedef void* id;
#endif

class MetalDevice : public GraphicsDevice
{
public:
    MetalDevice();
    ~MetalDevice() override = default;

    void Initialize(void* viewHandle) override;
    void BeginFrame() override;
    void EndFrame() override;

#ifdef __OBJC__
    id<MTLDevice> GetMetalDevice() override;
    id<MTLCommandQueue> GetMetalCommandQueue() override;
#else
    void* GetMetalDevice() override;
    void* GetMetalCommandQueue() override;
#endif

private:
    id _device;         // Use 'id' for Objective-C objects in headers
    id _commandQueue;
    MTKView* _view;
    id _currentCommandBuffer;
};
