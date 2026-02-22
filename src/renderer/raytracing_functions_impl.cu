#if defined(PLATFORM_CUDA_DEVICE) || defined(PLATFORM_CUDA_HOST)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cumath.h"
#include "raytracing_functions.h"


GPU_FUNC GPU_FUNC_INLINE half3 sampleTexture(half2 uv, const float3 pos, TEXTURE_OBJECT texObj)
{
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
    pos = floor3(pos);
    half eval = simplex3D_h(pos.x * freq, pos.y * freq, pos.z * freq);
    
    half eval2 = simplex3D_h((pos.x + 121.3f) * freq * 0.3f, 
                             (pos.y + 1321.3f) * freq * 0.3f, 
                             (pos.z + 721.5f) * freq * 0.3f);
    eval = eval * 0.4h + eval2 * 0.6h;

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

    // Sample the texture and return as float3
    float4 t = tex2D<float4>(texObj, (float)(uv.y), (float)(uv.x));
    return make_half3((half)t.x, (half)t.y, (half)t.z);
}
#endif