#include "renderer/BrickPoolTraits.hpp"
#include "State.hpp"
#include "renderer/MetalDevice.hpp"

#ifdef __OBJC__
#import <Metal/Metal.h>

namespace BrickPoolImpl {

MetalBrickPoolTraits::DeviceHandle MetalBrickPoolTraits::GetDevice() {
    return static_cast<MetalDevice*>(State::state.graphicsDevice.get())->GetMetalDevice();
}

MetalBrickPoolTraits::OccupancyBuffer MetalBrickPoolTraits::AllocateOccupancy(DeviceHandle device, size_t size) {
    id<MTLBuffer> buffer = [(id<MTLDevice>)device newBufferWithLength:size 
                                                              options:MTLResourceStorageModePrivate];
    buffer.label = @"BrickPool_Occupancy";
    return buffer;
}

MetalBrickPoolTraits::DataBuffer MetalBrickPoolTraits::AllocateData(DeviceHandle device, size_t size) {
    id<MTLBuffer> buffer = [(id<MTLDevice>)device newBufferWithLength:size 
                                                              options:MTLResourceStorageModePrivate];
    buffer.label = @"BrickPool_Data";
    return buffer;
}

void MetalBrickPoolTraits::ZeroOccupancy(DeviceHandle device, OccupancyBuffer buffer, size_t size) {
    id<MTLCommandQueue> queue = static_cast<MetalDevice*>(State::state.graphicsDevice.get())->GetMetalCommandQueue();
    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [cmdBuffer blitCommandEncoder];
    [encoder fillBuffer:(id<MTLBuffer>)buffer range:NSMakeRange(0, size) value:0];
    [encoder endEncoding];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
}

void MetalBrickPoolTraits::ZeroData(DeviceHandle device, DataBuffer buffer, size_t size) {
    id<MTLCommandQueue> queue = static_cast<MetalDevice*>(State::state.graphicsDevice.get())->GetMetalCommandQueue();
    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [cmdBuffer blitCommandEncoder];
    [encoder fillBuffer:(id<MTLBuffer>)buffer range:NSMakeRange(0, size) value:0];
    [encoder endEncoding];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
}

void MetalBrickPoolTraits::Log(const char* format, ...) {
    va_list args;
    va_start(args, format);
    NSString* nsFormat = [[NSString alloc] initWithUTF8String:format];
    NSLogv(nsFormat, args);
    va_end(args);
}

void MetalBrickPoolTraits::LogError(const char* format, ...) {
    va_list args;
    va_start(args, format);
    NSString* nsFormat = [[NSString alloc] initWithUTF8String:format];
    NSLogv(nsFormat, args);
    va_end(args);
}

}
#endif
