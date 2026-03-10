#include "renderer/Metal/MetalDevice.hpp"

MetalDevice::MetalDevice() 
{

    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    _device = nil;

    for (id<MTLDevice> device in devices) {
        if (![device isLowPower] && ![device isHeadless]) {
            _device = device;
            break;
        }
    }

    if (!_device) {
        _device = MTLCreateSystemDefaultDevice();
    }

    if (!_device) {
        NSLog(@"Fatal: Metal is not supported on this device.");
        exit(1);
    }

    NSLog(@"Selected GPU: %@", [(id<MTLDevice>)_device name]);

    _commandQueue = [_device newCommandQueue];
}

void MetalDevice::Initialize(void* viewHandle) {
    _view = (__bridge MTKView*)viewHandle;
    _view.framebufferOnly = NO;
    _view.colorPixelFormat = MTLPixelFormatRGBA8Unorm;
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
