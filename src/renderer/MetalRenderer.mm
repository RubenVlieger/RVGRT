#import <MetalKit/MetalKit.h>
#include "renderer/MetalRenderer.hpp"
#include "State.hpp"
#include "Character.hpp"
#include "renderer/ShaderTypes.h"

#include "renderer/MetalDevice.hpp"
#include "renderer/MetalBuffer.hpp"
#include "cumath.h"

#include "Texturepack.h"
#include <cassert>
#include <chrono> // Ensure chrono is included

MetalRenderer::MetalRenderer(id device_id) : _texturepack() 
{
    _device = (id<MTLDevice>)device_id;

    id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];
    if (!defaultLibrary) {
        NSLog(@"FATAL ERROR: Failed to find the default shader library.");
        abort();
    }
    NSError *error = nil;

    _supportsTimestamps = [((id<MTLDevice>)_device) supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
    
    if (_supportsTimestamps) 
    {
        id<MTLDevice> dev = (id<MTLDevice>)_device;
        
        // 2. Find the specific Counter Set for Timestamps (Fixing M4 Index 0 crash)
        id<MTLCounterSet> timestampSet = nil;
        for (id<MTLCounterSet> set in [dev counterSets]) {
            if ([set.name caseInsensitiveCompare:@"timestamp"] == NSOrderedSame) {
                timestampSet = set;
                break;
            }
            // Fallback check inside counters just in case
            for (id<MTLCounter> counter in set.counters) {
                if ([counter.name caseInsensitiveCompare:@"timestamp"] == NSOrderedSame) {
                    timestampSet = set;
                    break;
                }
            }
            if(timestampSet) break;
        }

        if (timestampSet) {
            MTLCounterSampleBufferDescriptor* desc = [[MTLCounterSampleBufferDescriptor alloc] init];
            desc.counterSet = timestampSet;
            desc.label = @"TimestampCounter";
            desc.sampleCount = 4;
            desc.storageMode = MTLStorageModePrivate;
            
            NSError* err = nil;
            _counterSampleBuffer = [dev newCounterSampleBufferWithDescriptor:desc error:&err];
            
            if(err || !_counterSampleBuffer) {
                NSLog(@"Error creating counter buffer: %@", err);
                _supportsTimestamps = false;
            } else {
                _timestampBuffer = [dev newBufferWithLength:4 * sizeof(uint64_t) options:MTLResourceStorageModeShared];
            }
        } else {
            NSLog(@"Warning: Device supports sampling, but no 'timestamp' CounterSet found.");
            _supportsTimestamps = false;
        }
    }


    // 1. World Gen PSO
    id<MTLFunction> worldGenFunc = [defaultLibrary newFunctionWithName:@"GeneratePackedWorld"];
    _worldGenerationPSO = [_device newComputePipelineStateWithFunction:worldGenFunc error:&error];
    if (!_worldGenerationPSO) NSLog(@"Error creating World Gen PSO: %@", error);

    // 2. Distance Approx PSO (Pre-pass)
    id<MTLFunction> approxFunc = [defaultLibrary newFunctionWithName:@"distApproximationKernel"];
    _distApproxPSO = [_device newComputePipelineStateWithFunction:approxFunc error:&error];
    if (!_distApproxPSO) NSLog(@"Error creating Dist Approx PSO: %@", error);

    // 3. Main Tiled Deferred PSO (The Imageblock Kernel)
    id<MTLFunction> tileFunc = [defaultLibrary newFunctionWithName:@"tiledDeferredRaytraceKernel"];
    _tiledDeferredPSO = [_device newComputePipelineStateWithFunction:tileFunc error:&error];
    if (!_tiledDeferredPSO) NSLog(@"Error creating Tiled Deferred PSO: %@", error);

    // --- One-time World Generation ---
    auto t1 = std::chrono::high_resolution_clock::now();

    const int packedWidth  = SIZEX / 4;
    const int packedHeight = SIZEY / 4;
    const int packedDepth  = SIZEZ / 2;

    MTLTextureDescriptor *voxelDesc = [[MTLTextureDescriptor alloc] init];
    voxelDesc.textureType = MTLTextureType3D;
    voxelDesc.pixelFormat = MTLPixelFormatR32Uint; 
    voxelDesc.width = packedWidth;
    voxelDesc.height = packedHeight;
    voxelDesc.depth = packedDepth;
    voxelDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    voxelDesc.storageMode = MTLStorageModePrivate;

    _voxelTexture = [_device newTextureWithDescriptor:voxelDesc];

    id<MTLCommandQueue> queue = [_device newCommandQueue];
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:_worldGenerationPSO]; 
    [encoder setTexture:_voxelTexture atIndex:0];

    MTLSize threadGroupSize = MTLSizeMake(8, 8, 8);
    MTLSize gridSize = MTLSizeMake(packedWidth, packedHeight, packedDepth);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    
    auto t2 = std::chrono::high_resolution_clock::now();
    NSLog(@"Voxel world generated in %.2f ms", std::chrono::duration<double, std::milli>(t2 - t1).count());

    _csdf.AllocateSDF();
    _csdf.GenerateSDF(_voxelTexture);
    
    auto t3 = std::chrono::high_resolution_clock::now();
    NSLog(@"CSDF generated in %.2f ms", std::chrono::duration<double, std::milli>(t3 - t2).count());

    _giData.AllocateGI();
    _giData.InitializeGIData(_voxelTexture, _csdf, _texturepack);
    
    auto t4 = std::chrono::high_resolution_clock::now();
    NSLog(@"GI grid initialized in %.2f ms", std::chrono::duration<double, std::milli>(t4 - t3).count());

    createRenderTarget(State::dispWIDTH, State::dispHEIGHT);
}

MetalRenderer::~MetalRenderer() {}

void MetalRenderer::createRenderTarget(uint32_t width, uint32_t height) 
{
    // 1. Main Output Texture
    MTLTextureDescriptor *mainDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
    mainDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    mainDesc.storageMode = MTLStorageModePrivate;
    _renderTargetTexture = [_device newTextureWithDescriptor:mainDesc];

    // 2. Half-Res Distance Texture (Accelerator)
    // Used to speed up ray marching in the main pass
    uint32_t halfW = width / 2;
    uint32_t halfH = height / 2;
    MTLTextureDescriptor *distDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:halfW height:halfH mipmapped:NO];
    distDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    distDesc.storageMode = MTLStorageModePrivate;
    _halfDistTexture = [_device newTextureWithDescriptor:distDesc];

    // Note: No G-Buffer textures are created here! 
    // They are implicit imageblocks living in Tile Memory.
}

void MetalRenderer::OnResize(uint32_t newWidth, uint32_t newHeight)
{
    if ([(id<MTLTexture>)_renderTargetTexture width] != newWidth || [(id<MTLTexture>)_renderTargetTexture height] != newHeight) {
        createRenderTarget(newWidth, newHeight);
    }
}

void MetalRenderer::Draw(id<MTLComputeCommandEncoder> encoder, const Character& character, unsigned int frameCount)
{
    if (_supportsTimestamps) {
        // Sample Index 0: Start of Pass 0
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:0 withBarrier:NO];
    }

    // --- Setup Data ---
    CameraData camData;
    camData.position = { (float)character.position.x, (float)character.position.y, (float)character.position.z };
    camData.forward  = { (float)character.direction.x, (float)character.direction.y, (float)character.direction.z };
    camData.right    = { character.camera.right.x, character.camera.right.y, character.camera.right.z };
    camData.up       = { character.camera.up.x, character.camera.up.y, character.camera.up.z };

    FrameData frameData;
    frameData.sunDirection = simd_normalize(simd_make_float3(10.f, 5.f, -4.f));
    frameData.time = (float)CFAbsoluteTimeGetCurrent();

    // ---------------------------------------------------------
    // PASS 0: DISTANCE APPROXIMATION (Accelerator)
    // ---------------------------------------------------------
    [encoder pushDebugGroup:@"Pass 0: Approx"];
    [encoder setComputePipelineState:_distApproxPSO];
    
    // Outputs
    [encoder setTexture:_halfDistTexture atIndex:0];
    
    // Inputs
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder setTexture:_voxelTexture atIndex:2];
    [encoder setTexture:(__bridge id<MTLTexture>)_csdf.getSDFTexture() atIndex:3];
    
    MTLSize gridHalf = MTLSizeMake([(id<MTLTexture>)_halfDistTexture width], [(id<MTLTexture>)_halfDistTexture height], 1);
    [encoder dispatchThreads:gridHalf threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [encoder popDebugGroup];

    if (_supportsTimestamps) {
        // Sample Index 1: End of Pass 0
        // Barrier YES ensures the kernel finishes before the time is recorded
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:1 withBarrier:YES];
    }

    // Barrier: Ensure Pass 0 writes are visible to Pass 1
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];



    // ---------------------------------------------------------
    // PASS 1: MAIN TILE-BASED DEFERRED RAYTRACING
    // ---------------------------------------------------------
    if (_supportsTimestamps) {
        // Sample Index 2: Start of Pass 1
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:2 withBarrier:NO];
    }
    [encoder pushDebugGroup:@"Pass 1: TBDR"];
    [encoder setComputePipelineState:_tiledDeferredPSO];

    // Output
    [encoder setTexture:_renderTargetTexture atIndex:0];
    
    // Inputs
    [encoder setBytes:&camData length:sizeof(CameraData) atIndex:0];
    [encoder setBytes:&frameData length:sizeof(FrameData) atIndex:1];
    [encoder setTexture:_voxelTexture atIndex:2];
    [encoder setTexture:(__bridge id<MTLTexture>)_csdf.getSDFTexture() atIndex:3];
    [encoder setTexture:(id<MTLTexture>)_texturepack.getTextureObject() atIndex:5];
    [encoder setTexture:_halfDistTexture atIndex:6]; // Read the accelerator

    // Dispatch Configuration for Imageblocks
    // Apple GPUs prefer tile sizes like 16x16 or 32x32. 
    // The threadgroup size MUST match the imageblock tile dimensions we want implicitly.
    // 16x16 = 256 threads per group, which fits within the max limit (usually 1024).
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1); 
    
    MTLSize gridFull = MTLSizeMake([(id<MTLTexture>)_renderTargetTexture width], 
                                   [(id<MTLTexture>)_renderTargetTexture height], 
                                   1);

    // Important: dispatchThreads automatically handles edge cases where grid size isn't a multiple of threadgroup size
    [encoder dispatchThreads:gridFull threadsPerThreadgroup:threadGroupSize];
    
    [encoder popDebugGroup];
    if (_supportsTimestamps) {
        // Sample Index 3: End of Pass 1
        [encoder sampleCountersInBuffer:_counterSampleBuffer atSampleIndex:3 withBarrier:YES];
    }
}

void MetalRenderer::Draw(const Character& character, unsigned int frameCount) {
    NSLog(@"Warning: MetalRenderer::Draw(character, frameCount) was called directly.");
}

id MetalRenderer::GetOutputTexture()
{
    return _renderTargetTexture;
}
