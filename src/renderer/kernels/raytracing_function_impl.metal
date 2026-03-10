#include "raytracing_functions.h"
#include "TerrainGeneration.h" // Ensure simplex3D is also cross-platform

// Only compile this code when using the Metal compiler
#if defined(PLATFORM_METAL)

int getTextureIndex(half2 oldGridID) {
    int gridX = (int)(oldGridID.x * 16.0h + 0.1h); // +0.1 for safety
    int gridY = (int)(oldGridID.y * 16.0h + 0.1h);

    return gridY * 16 + gridX;
}


// half3 sampleTexture(half2 uv, float3 pos, texture2d_array<float, access::sample> texObj, float depth)
// {
//     constexpr sampler s(coord::normalized, address::repeat, filter::linear, mip_filter::linear); 

//     const half2 texStoneID = make_half2(0.0f / 16.0f, 1.0f / 16.0f);
//     const half2 texDirtID = make_half2(0.0f / 16.0f, 2.0f / 16.0f);
//     const half2 texCobbleID = make_half2(1.0f / 16.0f, 0.0f / 16.0f);
//     const half2 texIronID = make_half2(2.0f / 16.0f, 1.0f / 16.0f);
//     const half2 texDiamondID = make_half2(3.0f / 16.0f, 2.0f / 16.0f);
//     const half2 texStone2ID = make_half2(0.0f / 16.0f, 0.0f / 16.0f);
//     const half2 texCoalID = make_half2(2.0f / 16.0f, 2.0f / 16.0f);
//     half2 whichBlock = make_half2(0.0f, 8.0f / 16.0f);

//     // Voxel material selection based on 3D noise
//     const float freq = 0.05f;
//     pos = floor3(pos);
//     float eval = simplex3D(pos.x * freq, pos.y * freq, pos.z * freq);
    
//     float eval2 = simplex3D((pos.x + 121.3f) * freq * 0.3f, 
//                              (pos.y + 1321.3f) * freq * 0.3f, 
//                              (pos.z + 721.5f) * freq * 0.3f);
//     eval = eval * 0.4f + eval2 * 0.6f;

//     if(eval < -1.3h) whichBlock = texStoneID;
//     else if(eval < -1.2f) whichBlock = texDiamondID;
//     else if(eval < -0.7f) whichBlock = texIronID;
//     else if(eval < 0.0f) whichBlock = texStoneID;
//     else if(eval < 0.1f) whichBlock = texCoalID;
//     else if(eval < 0.4f) whichBlock = texCobbleID;
//     else if(eval < 0.8f) whichBlock = texDirtID;
//     else if(eval < 1.2f) whichBlock = texStone2ID;
//     else whichBlock = texStoneID;


//     int sliceIndex = getTextureIndex(whichBlock);
//     float lod = 0.5 * log2(depth) - 6.0f;


//     // Calculate final UV in the atlasacc

//     float4 t = texObj.sample(s, float2(uv.y, uv.x), sliceIndex, level(lod));
//     return make_half3((half)t.x, (half)t.y, (half)t.z);
// }

#endif // PLATFORM_METAL
