#include "shader_macros.h"

#if defined(PLATFORM_METAL)
#include <metal_stdlib>
using namespace metal;
#endif

#include "cumath.h"
#include "renderer/ShaderTypes.h"

KERNEL(BilateralUpsample)(
    PARAM_TEXTURE_WRITE(tex2d_f32_w, texOutput, 0),
    PARAM_TEXTURE_READ(tex2d_f32_r, texHalfRes, 1),
    PARAM_TEXTURE_READ(tex2d_f32_r, texDepth, 2),

    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    CHECK_BOUNDS(texOutput);
    int width = texOutput.get_width();
    int height = texOutput.get_height();
#else
    CHECK_BOUNDS(_width, _height);
    int2 gid = GET_GID();
    int width = _width;
    int height = _height;
#endif

    float2 screenSize = make_float2(float(width), float(height));
    float2 lowResSize = screenSize * 0.5f;

    float centerDepth = TEX_READ_2D(texDepth, gid).r;

    float2 uv = (AS_FLOAT2(gid) + 0.5f) / screenSize;
    float2 halfPixelPos = uv * lowResSize - 0.5f;

    int2 basePos;
    basePos.x = int(MATH_FLOOR(halfPixelPos.x));
    basePos.y = int(MATH_FLOOR(halfPixelPos.y));

    float2 fracPart;
    fracPart.x = halfPixelPos.x - MATH_FLOOR(halfPixelPos.x);
    fracPart.y = halfPixelPos.y - MATH_FLOOR(halfPixelPos.y);

    float3 sumColor = make_float3(0.0f, 0.0f, 0.0f);
    float sumWeight = 0.0f;

    for (int y = 0; y <= 1; y++) {
        for (int x = 0; x <= 1; x++) {
            int2 halfCoord;
            halfCoord.x = basePos.x + x;
            halfCoord.y = basePos.y + y;

            halfCoord.x = MATH_CLAMP(halfCoord.x, 0, int(lowResSize.x) - 1);
            halfCoord.y = MATH_CLAMP(halfCoord.y, 0, int(lowResSize.y) - 1);

#if defined(PLATFORM_METAL)
            float3 indirect = texHalfRes.read(uint2(halfCoord.x, halfCoord.y)).rgb;
            uint2 depthCoord = uint2(halfCoord.x * 2, halfCoord.y * 2);
            depthCoord.x = min(depthCoord.x, uint(width) - 1);
            depthCoord.y = min(depthCoord.y, uint(height) - 1);
            float neighborDepth = texDepth.read(depthCoord).r;
#else
            float4 indirect4 = texHalfRes.read(halfCoord);
            float3 indirect = make_float3(indirect4.x, indirect4.y, indirect4.z);
            int2 depthCoord = make_int2(halfCoord.x * 2, halfCoord.y * 2);
            depthCoord.x = max(0, min(depthCoord.x, width - 1));
            depthCoord.y = max(0, min(depthCoord.y, height - 1));
            float neighborDepth = texDepth.read(depthCoord).x;
#endif

            float depthDiff = MATH_ABS(centerDepth - neighborDepth);
            float depthWeight = 1.0f / (1.0f + depthDiff * 4.0f);

            float bilinearWeight = ((x == 0) ? (1.0f - fracPart.x) : fracPart.x) *
                                   ((y == 0) ? (1.0f - fracPart.y) : fracPart.y);

            float weight = depthWeight * bilinearWeight;
            sumColor = make_float3(sumColor.x + indirect.x * weight,
                                   sumColor.y + indirect.y * weight,
                                   sumColor.z + indirect.z * weight);
            sumWeight += weight;
        }
    }

    if (sumWeight < 1e-4f) {
        sumColor = TEX_READ_2D(texHalfRes, uint2(AS_FLOAT2(gid) * 0.5f)).rgb;
        sumWeight = 1.0f;
    }

    float3 result = make_float3(sumColor.x / sumWeight,
                                sumColor.y / sumWeight,
                                sumColor.z / sumWeight);

#if defined(PLATFORM_METAL)
    TEX_WRITE_2D(texOutput, float4(result, 1.0), gid);
#else
    float4 outVal = make_float4(result.x, result.y, result.z, 1.0f);
    TEX_WRITE_2D(texOutput, outVal, gid);
#endif
}