// src/renderer/CoarseArray_impl.mm

#import <Metal/Metal.h>
#include <stdexcept>
#include <iostream>

#include "CoarseArray.h"
#include "State.hpp"
#include "renderer/GraphicsDevice.hpp"
#include "renderer/MetalDevice.hpp"
#include "Texturepack.h"


// --- Helper Functions ---

static id<MTLDevice> get_metal_device() {
    GraphicsDevice* gDevice = State::state.graphicsDevice.get();
    if (!gDevice) throw std::runtime_error("GraphicsDevice not initialized in State.");
    return static_cast<MetalDevice*>(gDevice)->GetMetalDevice();
}

static id<MTLComputePipelineState> create_pso(id<MTLDevice> device, const char* kernel_name) {
    NSError* error = nil;
    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) throw std::runtime_error("Could not load default Metal library.");
    
    id<MTLFunction> function = [library newFunctionWithName:@(kernel_name)];
    if (!function) {
        std::string msg = "Failed to find kernel function: ";
        msg += kernel_name;
        throw std::runtime_error(msg);
    }
    
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pso) {
        std::string msg = "Failed to create PSO for ";
        msg += kernel_name;
        msg += ". Error: ";
        msg += [error.localizedDescription UTF8String];
        throw std::runtime_error(msg);
    }
    return pso;
}


// --- C++ Class Method Implementations ---
CoarseArray::CoarseArray() {}
CoarseArray::~CoarseArray() {}

void CoarseArray::AllocateSDF()
{
    m_csdfArray.Allocate(SDF_BYTESIZE);
}

void CoarseArray::AllocateGI()
{
    m_csdfArray.Allocate(GI_BYTESIZE);
}

unsigned char* CoarseArray::getPtr()
{
    // The implementation simply retrieves the pointer from the underlying CArray member.
    return reinterpret_cast<unsigned char*>(m_csdfArray.getPtr());
}


void CoarseArray::GenerateSDF(CArray& fineArray)
{
    if (m_csdfArray.getSize() != SDF_BYTESIZE) {
        std::cerr << "CSDF not allocated or wrong size. Call AllocateSDF() first." << std::endl;
        return;
    }
    
    id<MTLDevice> device = get_metal_device();
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Create a temporary buffer for intermediate SDF generation passes
    CArray tempArray;
    tempArray.Allocate(SDF_BYTESIZE);

    // Get native MTLBuffer handles
    id<MTLBuffer> fineBuffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(fineArray.getPtr());
    id<MTLBuffer> csdfBuffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(m_csdfArray.getPtr());
    id<MTLBuffer> tempBuffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(tempArray.getPtr());
    
    // --- Create Pipeline States ---
    id<MTLComputePipelineState> psoX = create_pso(device, "CoarseArray_computeDistX");
    id<MTLComputePipelineState> psoY = create_pso(device, "CoarseArray_computeDistY");
    id<MTLComputePipelineState> psoZ = create_pso(device, "CoarseArray_computeDistZ");

    // --- Dispatch Kernels Sequentially ---
    MTLSize gridSize = MTLSizeMake(SDF_BYTESIZE, 1, 1);
    NSUInteger tgSize = [psoX maxTotalThreadsPerThreadgroup];
    if (tgSize > SDF_BYTESIZE) tgSize = SDF_BYTESIZE;
    MTLSize threadgroupSize = MTLSizeMake(tgSize, 1, 1);
    
    // Pass 1: X distance
    id<MTLCommandBuffer> cmdBufX = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encX = [cmdBufX computeCommandEncoder];
    [encX setComputePipelineState:psoX];
    [encX setBuffer:fineBuffer offset:0 atIndex:0];
    [encX setBuffer:tempBuffer offset:0 atIndex:1];
    [encX dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encX endEncoding];
    [cmdBufX commit];

    // Pass 2: Y distance
    id<MTLCommandBuffer> cmdBufY = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encY = [cmdBufY computeCommandEncoder];
    [encY setComputePipelineState:psoY];
    [encY setBuffer:tempBuffer offset:0 atIndex:0];
    [encY setBuffer:csdfBuffer offset:0 atIndex:1];
    [encY dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encY endEncoding];
    [cmdBufY commit];

    // Pass 3: Z distance
    id<MTLCommandBuffer> cmdBufZ = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encZ = [cmdBufZ computeCommandEncoder];
    [encZ setComputePipelineState:psoZ];
    [encZ setBuffer:csdfBuffer offset:0 atIndex:0];
    [encZ setBuffer:tempBuffer offset:0 atIndex:1]; // Write final result to temp
    [encZ dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encZ endEncoding];
    [cmdBufZ commit];

    // Wait for all passes to complete
    [cmdBufX waitUntilCompleted];
    [cmdBufY waitUntilCompleted];
    [cmdBufZ waitUntilCompleted];
    
    // Copy final result from temp buffer to the main csdf buffer
    id<MTLCommandBuffer> blitCmdBuf = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blitEnc = [blitCmdBuf blitCommandEncoder];
    [blitEnc copyFromBuffer:tempBuffer sourceOffset:0 toBuffer:csdfBuffer destinationOffset:0 size:SDF_BYTESIZE];
    [blitEnc endEncoding];
    [blitCmdBuf commit];
    [blitCmdBuf waitUntilCompleted];

    tempArray.Free();
    std::cout << "Metal CSDF Generation Complete." << std::endl;
}


void CoarseArray::InitializeGIData(CArray& fineArray, CoarseArray& csdf, Texturepack& texturepack)
{
    id<MTLDevice> device = get_metal_device();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    
    id<MTLComputePipelineState> pso = create_pso(device, "CoarseArray_InitialGlobalIlluminate");

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:(__bridge id<MTLBuffer>)reinterpret_cast<void*>(getPtr()) offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)reinterpret_cast<void*>(fineArray.getPtr()) offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)reinterpret_cast<void*>(csdf.getPtr()) offset:0 atIndex:2];

    float3 sunDir = normalize(make_float3(10.f, 5.f, -4.f));
    [encoder setBytes:&sunDir length:sizeof(float3) atIndex:3];

    MTLSize gridSize = MTLSizeMake(GI_SIZE, 1, 1);
    NSUInteger tgSize = [pso maxTotalThreadsPerThreadgroup];
    if (tgSize > GI_SIZE) tgSize = GI_SIZE;
    MTLSize threadgroupSize = MTLSizeMake(tgSize, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}


static int g_frameNumber = 0;
static uint64_t g_offsetCounter = 0;

void CoarseArray::UpdateGIData(CArray& fineArray, CoarseArray& csdf, Texturepack& texturepack)
{
    #define METAL_RAYPS (64*64*64*1)
    
    id<MTLDevice> device = get_metal_device();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    
    id<MTLComputePipelineState> pso = create_pso(device, "CoarseArray_GlobalIlluminate");

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:pso];
    [encoder setBuffer:(__bridge id<MTLBuffer>)reinterpret_cast<void*>(getPtr()) offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)reinterpret_cast<void*>(fineArray.getPtr()) offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)reinterpret_cast<void*>(csdf.getPtr()) offset:0 atIndex:2];
    
    // Texturepack is not yet implemented for Metal, so we pass a nil texture.
    // Replace this once you have a MetalTexturepack class.
    id<MTLTexture> tex = nil; 
    [encoder setTexture:tex atIndex:3];

    float3 sunDir = normalize(make_float3(10.f, 5.f, -4.f));
    [encoder setBytes:&sunDir length:sizeof(float3) atIndex:4];
    
    uint frameNum = g_frameNumber;
    [encoder setBytes:&frameNum length:sizeof(uint) atIndex:5];
    [encoder setBytes:&g_offsetCounter length:sizeof(uint64_t) atIndex:6];

    MTLSize gridSize = MTLSizeMake(METAL_RAYPS, 1, 1);
    NSUInteger tgSize = [pso maxTotalThreadsPerThreadgroup];
    if (tgSize > METAL_RAYPS) tgSize = METAL_RAYPS;
    MTLSize threadgroupSize = MTLSizeMake(tgSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [cmdBuf commit];
    // NOTE: Don't wait for completion here for performance. Let it run in the background.

    g_frameNumber++;
    g_offsetCounter = (g_offsetCounter + METAL_RAYPS) % GI_SIZE;
}