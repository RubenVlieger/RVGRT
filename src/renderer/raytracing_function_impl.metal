#include "raytracing_functions.h"
#include "TerrainGeneration.h" // Ensure simplex3D is also cross-platform

// Only compile this code when using the Metal compiler
#if defined(PLATFORM_METAL)

float3 sampleTexture(half2 uv, const float3 pos, TEXTURE_OBJECT texObj)
{
    sampler s(filter::linear);
    // Texture selection constants
    const half2 texStoneID = make_half2(0.0f / 16.0f, 1.0f / 16.0f);
    const half2 texDirtID = make_half2(0.0f / 16.0f, 2.0f / 16.0f);
    const half2 texCobbleID = make_half2(1.0f / 16.0f, 0.0f / 16.0f);
    const half2 texIronID = make_half2(2.0f / 16.0f, 1.0f / 16.0f);
    const half2 texDiamondID = make_half2(3.0f / 16.0f, 2.0f / 16.0f);
    const half2 texStone2ID = make_half2(0.0f / 16.0f, 0.0f / 16.0f);
    const half2 texSandStoneID = make_half2(11.0f / 16.0f, 0.0f / 16.0f);
    const half2 texCoalID = make_half2(2.0f / 16.0f, 2.0f / 16.0f);
    half2 whichBlock = make_half2(0.0f, 8.0f / 16.0f);

    // Voxel material selection based on 3D noise
    const float freq = 0.05f;
    int3 ipos = to_int3(floor3(pos));
    float eval = simplex3D(ipos.x * freq, ipos.y * freq, ipos.z * freq);
    float eval2 = simplex3D((ipos.x + 121.3f) * freq * 0.3f, (ipos.y + 1321.3f) * freq * 0.3f, (ipos.z + 721.5f) * freq * 0.3f);
    eval = eval * 0.4f + eval2 * 0.6f;

    if(eval < -1.3f) whichBlock = texStoneID;
    else if(eval < -1.2f) whichBlock = texDiamondID;
    else if(eval < -0.7f) whichBlock = texIronID;
    else if(eval < 0.0f) whichBlock = texStoneID;
    else if(eval < 0.1f) whichBlock = texCoalID;
    else if(eval < 0.4f) whichBlock = texCobbleID;
    else if(eval < 0.8f) whichBlock = texDirtID;
    else if(eval < 1.2f) whichBlock = texStone2ID;
    else whichBlock = texStoneID;

    // Calculate final UV in the atlas
    uv.x = ((uv.x * ((half)1.0f/16.0))) + whichBlock.x;
    uv.y = ((uv.y * ((half)1.0f/16.0))) + whichBlock.y;
    
    half4 t = texObj.sample(s, float2(uv)); // Metal sample takes float2 UVs
    return make_float3(t.x, t.y, t.z);
}

#endif // PLATFORM_METAL