// Distance Approximation (Half-Resolution)
// Performs a coarse ray trace to find approximate distance to first hit.

#include "ShaderTypes.slang"
#include "Intersection.slang"

RWTexture2D<float4> distTex;

uniform CameraData camera;
uniform FrameData frame;

Texture3D<uint> indirection;
StructuredBuffer<SectorInfo> sectorBuffer;
StructuredBuffer<uint64_t> occupancyBuffer;
StructuredBuffer<uint8_t> dataBuffer;
StructuredBuffer<uint64_t> sectorMaskBuffer;
ConstantBuffer<CharacterGPUData> charData;

[shader("compute")]
[numthreads(16, 16, 1)]
void distApproximationKernel(uint3 gid : SV_DispatchThreadID)
{
    uint2 dim;
    distTex.GetDimensions(dim.x, dim.y);
    if (gid.x >= dim.x || gid.y >= dim.y) return;

    float2 uv = (float2(gid.xy) + float2(0.5f, 0.5f)) / float2(dim);
    float2 ndc = uv * 2.0f - 1.0f;
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    // LOD-only trace
    hitInfo hit = trace(camera.position, dir, 
                        frame.worldOrigin, IND_X, IND_Y, IND_Z,
                        DIST_APPROX_MAX_ITERS, true);
    
    float dist = hit.hit ? length(hit.pos - camera.position) : 5000.0f;
    
    // Safety padding for the main raymarch
    dist = max(0.0f, dist - 8.0f);

    distTex[gid.xy] = float4(dist, 0, 0, 0);
}
