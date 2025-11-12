#include "renderer/MetalDevice.hpp"

MetalDevice::MetalDevice() {
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        // In a real app, you would throw an exception or display an alert
        NSLog(@"Fatal: Metal is not supported on this device.");
        exit(1);
    }
    _commandQueue = [_device newCommandQueue];
}

void MetalDevice::Initialize(void* viewHandle) {
    _view = (__bridge MTKView*)viewHandle;
    _view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    // Set other view properties if needed
}

void MetalDevice::BeginFrame() {
    _currentCommandBuffer = [_commandQueue commandBuffer];
    
    MTLRenderPassDescriptor* renderPassDescriptor = _view.currentRenderPassDescriptor;
    
    if (renderPassDescriptor != nil) {
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.1, 0.2, 0.4, 1.0);
        id<MTLRenderCommandEncoder> renderEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        [renderEncoder endEncoding];
    }
}

void MetalDevice::EndFrame() {
    if (_currentCommandBuffer) {
        [_currentCommandBuffer presentDrawable:_view.currentDrawable];
        [_currentCommandBuffer commit];
        _currentCommandBuffer = nil;
    }
}

// --- Implementation of the required Getters ---

#ifdef __OBJC__
// Objective-C++ implementation (returns the real type)
id<MTLDevice> MetalDevice::GetMetalDevice() {
    return _device;
}
id<MTLCommandQueue> MetalDevice::GetMetalCommandQueue() {
    return _commandQueue;
}
#else
// Pure C++ implementation (returns the void*)
void* MetalDevice::GetMetalDevice() {
    return _device;
}
void* MetalDevice::GetMetalCommandQueue() {
    return _commandQueue;
}
#endif