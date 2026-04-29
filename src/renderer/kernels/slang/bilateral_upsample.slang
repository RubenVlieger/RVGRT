// Bilateral Upsample

RWTexture2D<float4> texOutput;
Texture2D<float4> texHalfRes;
Texture2D<float4> texDepth;

[shader("compute")]
[numthreads(16, 16, 1)]
void BilateralUpsample(uint3 gid : SV_DispatchThreadID)
{
    uint2 dim;
    texOutput.GetDimensions(dim.x, dim.y);
    if (gid.x >= dim.x || gid.y >= dim.y) return;

    float2 screenSize = float2(dim);
    float2 lowResSize = screenSize * 0.5f;

    float centerDepth = texDepth[gid.xy].r;

    float2 uv = (float2(gid.xy) + 0.5f) / screenSize;
    float2 halfPixelPos = uv * lowResSize - 0.5f;

    int2 basePos = int2(floor(halfPixelPos));
    float2 fracPart = halfPixelPos - floor(halfPixelPos);

    float3 sumColor = float3(0.0f, 0.0f, 0.0f);
    float sumWeight = 0.0f;

    for (int y = 0; y <= 1; y++) {
        for (int x = 0; x <= 1; x++) {
            int2 halfCoord = basePos + int2(x, y);
            halfCoord.x = clamp(halfCoord.x, 0, int(lowResSize.x) - 1);
            halfCoord.y = clamp(halfCoord.y, 0, int(lowResSize.y) - 1);

            float3 indirect = texHalfRes[halfCoord].rgb;
            uint2 depthCoord = uint2(halfCoord.x * 2, halfCoord.y * 2);
            depthCoord.x = min(depthCoord.x, dim.x - 1);
            depthCoord.y = min(depthCoord.y, dim.y - 1);
            float neighborDepth = texDepth[depthCoord].r;

            float depthDiff = abs(centerDepth - neighborDepth);
            float depthWeight = 1.0f / (1.0f + depthDiff * 4.0f);

            float bilinearWeight = ((x == 0) ? (1.0f - fracPart.x) : fracPart.x) *
                                   ((y == 0) ? (1.0f - fracPart.y) : fracPart.y);

            float weight = depthWeight * bilinearWeight;
            sumColor += indirect * weight;
            sumWeight += weight;
        }
    }

    if (sumWeight < 1e-4f) {
        sumColor = texHalfRes[uint2(float2(gid.xy) * 0.5f)].rgb;
        sumWeight = 1.0f;
    }

    float3 result = sumColor / sumWeight;
    texOutput[gid.xy] = float4(result, 1.0f);
}
