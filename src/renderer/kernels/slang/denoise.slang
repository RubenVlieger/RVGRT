// A-Trous Edge-Avoiding Bilateral Filter
// Performs edge-preserving denoising using normal and depth discontinuities.

RWTexture2D<float4> texDenoised;
Texture2D<float4> texAccum;
Texture2D<float4> texNormal;
Texture2D<float4> texDepth;

uniform int step_width;

[shader("compute")]
[numthreads(16, 16, 1)]
void BilateralDenoise(uint3 gid : SV_DispatchThreadID)
{
    uint2 dim;
    texDenoised.GetDimensions(dim.x, dim.y);
    if (gid.x >= dim.x || gid.y >= dim.y) return;

    // Center tap data
    float3 centerC = texAccum[gid.xy].rgb;
    float3 centerN = texNormal[gid.xy].rgb;
    float centerD = texDepth[gid.xy].r;

    // Gaussian-approximate weights for 3x3
    const float kernelWeights[3] = { 1.0f, 2.0f / 1.0f, 4.0f / 1.0f };

    float3 sumColor = float3(0.0f);
    float sumWeight = 0.0f;

    // 3x3 grid with holes (A-Trous)
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            int2 offset = int2(x, y) * step_width;
            uint2 tapCoord = uint2(gid.x + offset.x, gid.y + offset.y);
            if(tapCoord.x >= dim.x || tapCoord.y >= dim.y) {
                tapCoord = gid.xy;
            }

            float3 tapC = texAccum[tapCoord].rgb;
            float3 tapN = texNormal[tapCoord].rgb;
            float tapD = texDepth[tapCoord].r;

            // Normal weight
            float dotN = max(dot(centerN, tapN), 0.0f);
            float wNormal = pow(dotN, 16.0f);

            // Depth weight
            float wDepth = (abs(centerD - tapD) < 1.5f) ? 1.0f : 0.0f;

            // Kernel weight
            float kWeight = kernelWeights[abs(x)] * kernelWeights[abs(y)];
            float w = wNormal * wDepth * kWeight;

            sumColor += tapC * w;
            sumWeight += w;
        }
    }

    if (sumWeight < 1e-4f) {
        sumColor = centerC;
        sumWeight = 1.0f;
    }
    
    texDenoised[gid.xy] = float4(sumColor / sumWeight, 1.0f);
}
