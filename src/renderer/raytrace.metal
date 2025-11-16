#include <metal_stdlib>
#include "renderer/ShaderTypes.h" 
#include "TerrainGeneration.h"

#include "cumath.h"

using namespace metal;


kernel void generate_world_kernel(
    // The output buffer to write the packed voxel data into.
    device uint* world_data [[buffer(0)]],

    // The unique ID for this thread, corresponding to the word index in the buffer.
    uint wordIdx [[thread_position_in_grid]])
{
    // Calculate the starting absolute index of the first voxel this thread will process.
    uint64_t baseBit = (uint64_t)wordIdx * 32ull;
    
    // This will hold the 32 packed bits for the voxels we're about to compute.
    uint w = 0u;

    // Loop 32 times, once for each bit in our output unsigned int.
    for (uint bit_offset = 0; bit_offset < 32; ++bit_offset) {
        
        uint64_t bitIndex = baseBit + bit_offset;

        uint64_t z = bitIndex >> (SHIX + SHIY);
        uint64_t y = (bitIndex >> SHIX) & (uint64_t)MODY;
        uint64_t x = bitIndex & (uint64_t)MODX;
        
        float density = Evaluate((float)x, (float)y, (float)z);
        
        if (density > 0.7f) {
            w |= (1u << bit_offset);
        }
    }
    
    // Write the final packed 32-bit word to the output buffer.
    world_data[wordIdx] = w;
}

// The kernel signature is now simpler because CameraData is defined in the header.
kernel void raytrace_kernel(
    texture2d<float, access::write> outputTexture [[texture(0)]],
    device const CameraData& cameraData [[buffer(0)]],
    device const uint* world_data [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = outputTexture.get_width();
    uint height = outputTexture.get_height();

    if (gid.x >= width || gid.y >= height) {
        return;
    }

    float2 uv = float2(gid) / float2(width, height);
    float2 ndc = uv * 2.0 - 1.0;
    
    float3 ray_direction = normalize(cameraData.forward + ndc.x * cameraData.right + ndc.y * cameraData.up);

    float3 final_color = ray_direction;

    outputTexture.write(float4(final_color, 1.0), gid);
}