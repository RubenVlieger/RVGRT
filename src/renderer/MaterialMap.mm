#import "renderer/MaterialMap.hpp"
#import "State.hpp"
#import "renderer/MetalDevice.hpp"
#import "cumath.h"
#include <iostream>
#include <vector>
#include <cmath>

// --- Constants ---
// Geometry Atlas (R32Uint): 1 pixel = 4x4x2 voxels.
// A brick is 8x8x8 voxels.
// Therefore, we need 2x2x4 pixels in the atlas to represent 1 brick.
#ifndef GEO_TEX_SCALE_X
#define GEO_TEX_SCALE_X 2
#define GEO_TEX_SCALE_Y 2
#define GEO_TEX_SCALE_Z 4
#endif

// Indirection Offset (0=Air, 1=SolidGeneric, 2+=Index)
#ifndef INDIRECTION_BASE_OFFSET
#define INDIRECTION_BASE_OFFSET 2
#endif


namespace {
    id<MTLDevice> get_device() {
        return static_cast<MetalDevice*>(State::state.graphicsDevice.get())->GetMetalDevice();
    }

    inline uint32_t expandBits(uint32_t v) {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }

    inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
        return expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
    }

    struct BrickSortInfo {
        uint32_t linearIndex;
        uint32_t mortonCode;
    };
}
MaterialMap::MaterialMap() : 
    _indirectionTexture(nil), _geoBuffer(nil), _matBuffer(nil) 
{
    _device = get_device();
    id<MTLLibrary> lib = [_device newDefaultLibrary];
    
    NSError* err = nil;
    _psoAnalyze = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"AnalyzeWorldStructure"] error:&err];
    if(err || !_psoAnalyze) NSLog(@"Shader Load Error: %@", err);

    _psoFill    = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"FillDynamicAtlases"] error:&err];
    if(err || !_psoFill) NSLog(@"Shader Load Error: %@", err);
    
    _psoJFAInit   = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"JFA_Init"] error:&err];
    if(err || !_psoJFAStep) NSLog(@"Shader Load Error: %@", err);

    _psoJFAStep   = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"JFA_Step"] error:&err];
    if(err || !_psoJFAStep) NSLog(@"Shader Load Error: %@", err);

    _psoJFACommit = [_device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"JFA_Commit"] error:&err];
    if(err || !_psoJFACommit) NSLog(@"Shader Load Error: %@", err);
}


MaterialMap::~MaterialMap() {
    _indirectionTexture = nil;
    _geoBuffer = nil;
    _matBuffer = nil;
}
// Helper to create 3D Texture
id<MTLTexture> create3DTex(id<MTLDevice> dev, int w, int h, int d, MTLPixelFormat fmt, NSString* label) {
    MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType3D;
    desc.pixelFormat = fmt;
    desc.width = w;
    desc.height = h;
    desc.depth = d;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate; 
    id<MTLTexture> tex = [dev newTextureWithDescriptor:desc];
    tex.label = label;
    return tex;
}

void MaterialMap::GenerateDynamic() 
{
    id<MTLDevice> dev = (id<MTLDevice>)_device;
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    
    NSUInteger totalBricks = IND_X * IND_Y * IND_Z;
    id<MTLBuffer> statusBuffer = [dev newBufferWithLength:totalBricks * sizeof(uint32_t) options:MTLResourceStorageModeShared];

    // 1. RUN ANALYSIS PASS (GPU) 
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:_psoAnalyze];
    [enc setBuffer:statusBuffer offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(IND_X, IND_Y, IND_Z) threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // 2. CPU COMPACTION & MORTON SORTING 
    uint32_t* gridData = (uint32_t*)[statusBuffer contents];
    std::vector<BrickSortInfo> activeBricks;
    activeBricks.reserve(totalBricks / 10);

    for(size_t i = 0; i < totalBricks; i++) {
        uint32_t status = gridData[i];
        if (status == 2) { // Mixed
            uint32_t z = i / (IND_X * IND_Y);
            uint32_t rem = i % (IND_X * IND_Y);
            uint32_t y = rem / IND_X;
            uint32_t x = rem % IND_X;
            uint32_t mCode = morton3D(x, y, z);
            activeBricks.push_back({ (uint32_t)i, mCode });
        } 
    }

    if (activeBricks.empty()) {
        std::cout << "No active bricks found." << std::endl;
        return;
    }

    // Sort by Morton Code
    std::sort(activeBricks.begin(), activeBricks.end(), [](const BrickSortInfo& a, const BrickSortInfo& b) {
        return a.mortonCode < b.mortonCode;
    });

    // =========================================================
    // 3. ALLOCATE LINEAR BUFFERS (THE BIG CHANGE)
    // =========================================================
    uint32_t count = (uint32_t)activeBricks.size();
    
    std::cout << "Active Bricks: " << count << " (Linear Mode)" << std::endl;

    NSUInteger geoSize = count * 16 * sizeof(uint32_t);
    NSUInteger matSize = count * 512 * sizeof(uint8_t);

    // Release old
    _geoBuffer = nil; 
    _matBuffer = nil;
    _indirectionTexture = nil;

    // Allocate New
    _geoBuffer = [dev newBufferWithLength:geoSize options:MTLResourceStorageModePrivate];
    _matBuffer = [dev newBufferWithLength:matSize options:MTLResourceStorageModePrivate];
    ((id<MTLTexture>)_geoBuffer).label = @"GeometryPoolBuffer";
    ((id<MTLTexture>)_matBuffer).label = @"MaterialPoolBuffer";

    _indirectionTexture = create3DTex(dev, IND_X, IND_Y, IND_Z, MTLPixelFormatR32Uint, @"IndirectionGrid");

    // 4. WRITE BACK SORTED INDICES
    for(size_t i = 0; i < activeBricks.size(); ++i) {
        uint32_t originalLinearIndex = activeBricks[i].linearIndex;
        gridData[originalLinearIndex] = INDIRECTION_BASE_OFFSET + (uint32_t)i;
    }

    cmd = [queue commandBuffer];
    
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:statusBuffer 
            sourceOffset:0 
       sourceBytesPerRow:IND_X * 4 
     sourceBytesPerImage:IND_X * IND_Y * 4 
              sourceSize:MTLSizeMake(IND_X, IND_Y, IND_Z) 
               toTexture:(id<MTLTexture>)_indirectionTexture 
        destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];

    enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:_psoFill];
    [enc setTexture:(id<MTLTexture>)_indirectionTexture atIndex:0];
    
    [enc setBuffer:_geoBuffer offset:0 atIndex:0]; // Buffer Index 0
    [enc setBuffer:_matBuffer offset:0 atIndex:1]; // Buffer Index 1    
    
    [enc dispatchThreads:MTLSizeMake(IND_X, IND_Y, IND_Z) threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];


    id<MTLTexture> jfaTexA = create3DTex(dev, IND_X, IND_Y, IND_Z, MTLPixelFormatR32Uint, @"JFA_A");
    id<MTLTexture> jfaTexB = create3DTex(dev, IND_X, IND_Y, IND_Z, MTLPixelFormatR32Uint, @"JFA_B");

    cmd = [queue commandBuffer];
    enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:_psoJFAInit];
    [enc setTexture:(id<MTLTexture>)_indirectionTexture atIndex:0];
    [enc setTexture:jfaTexA atIndex:1];
    [enc dispatchThreads:MTLSizeMake(IND_X, IND_Y, IND_Z) threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];

    int step = 128;
    id<MTLTexture> input = jfaTexA;
    id<MTLTexture> output = jfaTexB;
    
    [enc setComputePipelineState:_psoJFAStep];
    while(step >= 1) {
        [enc setTexture:input atIndex:0];
        [enc setTexture:output atIndex:1];
        [enc setBytes:&step length:sizeof(int) atIndex:0];
        [enc dispatchThreads:MTLSizeMake(IND_X, IND_Y, IND_Z) threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
        [enc memoryBarrierWithScope:MTLBarrierScopeTextures];
        id<MTLTexture> tmp = input; input = output; output = tmp;
        step /= 2;
    }

    [enc setComputePipelineState:_psoJFACommit];
    [enc setTexture:input atIndex:0];
    [enc setTexture:(id<MTLTexture>)_indirectionTexture atIndex:1];
    [enc dispatchThreads:MTLSizeMake(IND_X, IND_Y, IND_Z) threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    
    statusBuffer = nil;
}

// --- GETTERS (Ensure these are present!) ---
id MaterialMap::GetIndirectionTexture() { return _indirectionTexture; }
id MaterialMap::GetGeoBuffer() { return _geoBuffer; }
id MaterialMap::GetMatBuffer() { return _matBuffer; }