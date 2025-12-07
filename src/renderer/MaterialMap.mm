#import "renderer/MaterialMap.hpp"
#import "State.hpp"
#import "renderer/MetalDevice.hpp"
#import "cumath.h" // For SIZEX, SIZEY, SIZEZ definitions
#include <iostream>
#include <stdexcept>

// Indirection Grid Dimensions
// World is (2048, 512, 2048). Brick is 8.
// Grid is (256, 64, 256).
#define IND_X (SIZEX / 8)
#define IND_Y (SIZEY / 8)
#define IND_Z (SIZEZ / 8)

namespace {
    id<MTLDevice> get_device() {
        GraphicsDevice* gDevice = State::state.graphicsDevice.get();
        if (!gDevice) throw std::runtime_error("MaterialMap: GraphicsDevice not initialized.");
        return static_cast<MetalDevice*>(gDevice)->GetMetalDevice();
    }

    id<MTLComputePipelineState> load_kernel(id<MTLDevice> device, NSString* name) {
        NSError* error = nil;
        id<MTLLibrary> lib = [device newDefaultLibrary];
        id<MTLFunction> func = [lib newFunctionWithName:name];
        if(!func) {
            NSLog(@"FATAL: Could not find function '%@'. Did you compile the shader?", name);
            abort();
        }
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error];
        if(!pso) {
            NSLog(@"FATAL: Failed to create PSO for '%@': %@", name, error);
            abort();
        }
        return pso;
    }
}

MaterialMap::MaterialMap() : _indirectionTexture(nil), _brickPoolBuffer(nil), _allocCounterBuffer(nil) {
    _device = get_device();
    _psoClassify = load_kernel(_device, @"MaterialMap_Classify");
    _psoFill     = load_kernel(_device, @"MaterialMap_Fill");
}

MaterialMap::~MaterialMap() {
    _indirectionTexture = nil;
    _brickPoolBuffer = nil;
    _allocCounterBuffer = nil;
}

void MaterialMap::Allocate() {
    id<MTLDevice> device = (id<MTLDevice>)_device;

    // 1. Allocate Indirection Texture (3D R32Uint)
    MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType3D;
    desc.pixelFormat = MTLPixelFormatR32Uint;
    desc.width = IND_X;
    desc.height = IND_Y;
    desc.depth = IND_Z;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate; // GPU only
    
    _indirectionTexture = [device newTextureWithDescriptor:desc];
    [(id<MTLTexture>)_indirectionTexture setLabel:@"MaterialIndirectionGrid"];
    
    if(!_indirectionTexture) {
        throw std::runtime_error("Failed to allocate Material Indirection Texture");
    }

    // 2. Allocate Brick Pool (Linear Buffer)
    // Size = MAX_BRICKS * 8*8*8 bytes (512 bytes per brick)
    NSUInteger poolSize = (NSUInteger)MAX_BRICKS * 512;
    _brickPoolBuffer = [device newBufferWithLength:poolSize options:MTLResourceStorageModePrivate];
    [(id<MTLBuffer>)_brickPoolBuffer setLabel:@"MaterialBrickPool"];

    if(!_brickPoolBuffer) {
        throw std::runtime_error("Failed to allocate Material Brick Pool (OOM?)");
    }

    // 3. Atomic Counter (Shared so CPU can reset it easily, or Private with a clear kernel)
    // We'll use private with a clear, or shared. Shared is fine for a 4-byte buffer.
    _allocCounterBuffer = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    std::cout << "MaterialMap Allocated: Indirection Grid (" << IND_X << "x" << IND_Y << "x" << IND_Z 
              << "), Brick Pool Capacity: " << MAX_BRICKS << " chunks (" << (poolSize / 1024 / 1024) << " MB)" << std::endl;
}

void MaterialMap::Generate(id packedVoxelTexture) {
    if(!_indirectionTexture || !_brickPoolBuffer) Allocate();

    id<MTLDevice> device = (id<MTLDevice>)_device;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    cmdBuf.label = @"MaterialGenerationCmds";
    
    // Reset Counter
    memset([(id<MTLBuffer>)_allocCounterBuffer contents], 0, sizeof(uint32_t));

    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    // --- PASS 1: CLASSIFY ---
    // Reads: Geometry Bits
    // Writes: Indirection Texture, Atomically increments Counter
    [enc pushDebugGroup:@"Material Classification"];
    [enc setComputePipelineState:_psoClassify];
    [enc setTexture:(id<MTLTexture>)packedVoxelTexture atIndex:0];
    [enc setTexture:(id<MTLTexture>)_indirectionTexture atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)_allocCounterBuffer offset:0 atIndex:0];
    
    // Dispatch threads covering the Indirection Grid dimensions
    MTLSize gridSize = MTLSizeMake(IND_X, IND_Y, IND_Z);
    MTLSize threadGroupSize = MTLSizeMake(8, 8, 4); // Adjust based on GPU architecture
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [enc popDebugGroup];

    // Barrier: Ensure Indirection texture writes and Counter updates are visible
    [enc memoryBarrierWithScope:MTLBarrierScopeTextures | MTLBarrierScopeBuffers];

    // --- PASS 2: FILL POOL ---
    // Reads: Indirection Texture (to check if a block is mixed)
    // Writes: Brick Pool Buffer
    [enc pushDebugGroup:@"Material Fill"];
    [enc setComputePipelineState:_psoFill];
    [enc setTexture:(id<MTLTexture>)_indirectionTexture atIndex:0]; // Read-only now
    [enc setBuffer:(id<MTLBuffer>)_brickPoolBuffer offset:0 atIndex:0]; // Write output
    
    // We dispatch over the same grid. 
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [enc popDebugGroup];

    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    
    // Debug output: How many bricks did we use?
    uint32_t usedBricks = *(uint32_t*)[(id<MTLBuffer>)_allocCounterBuffer contents];
    std::cout << "Material Generation Complete. Used Mixed Bricks: " << usedBricks 
              << " / " << MAX_BRICKS << " (" << (usedBricks * 100.0f / MAX_BRICKS) << "%)" << std::endl;
}

id MaterialMap::GetIndirectionTexture() {
    return _indirectionTexture;
}

id MaterialMap::GetBrickPoolBuffer() {
    return _brickPoolBuffer;
}