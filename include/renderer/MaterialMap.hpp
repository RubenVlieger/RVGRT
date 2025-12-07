#pragma once

#include <cstdint>

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* id;
#endif

class MaterialMap {
public:
    MaterialMap();
    ~MaterialMap();

    // Allocates the Indirection Texture and the Brick Pool Buffer
    void Allocate();

    // Runs the GPU generation logic. 
    // Requires the packed voxel geometry texture (R32Uint) as input.
    void Generate(id packedVoxelTexture);

    // Getters for binding to the Render Loop
    id GetIndirectionTexture();
    id GetBrickPoolBuffer();

private:
    id _device;

    // 1. Indirection Grid: 3D Texture (R32Uint)
    // Stores: 
    //   0 -> Air
    //   0x8000XXXX -> Constant Material ID (e.g., Stone)
    //   0x0000XXXX -> Index into Brick Pool
    id _indirectionTexture;

    // 2. Brick Pool: Linear Buffer (Raw Bytes)
    // Stores dense 8x8x8 blocks for mixed areas.
    // Size = MaxBricks * 512 bytes.
    id _brickPoolBuffer;

    // 3. Atomic Counter
    // Used by shaders to allocate next available slot in Brick Pool
    id _allocCounterBuffer;

    // Compute Pipelines
    id _psoClassify; // Pass 1: Analyze geometry, reserve space
    id _psoFill;     // Pass 2: Calculate noise, fill reserved space

    // Constants
    const uint32_t BRICK_SIZE = 8;
    
    // Estimate: 2 million mixed chunks max (~1GB VRAM). 
    // Adjust based on your memory budget.
    const uint32_t MAX_BRICKS = 2 * 1024 * 100; 
};