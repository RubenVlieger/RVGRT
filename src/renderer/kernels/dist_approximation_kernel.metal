#include <metal_stdlib>
#include "cumath.h" 
#include "raytracing_functions.h"  
#include "renderer/ShaderTypes.h"
#include "TerrainGeneration.h"
using namespace metal;


kernel void distApproximationKernel(
    texture2d<float, access::write> distTex [[texture(0)]],
    constant const CameraData& camera [[buffer(0)]],
    constant const FrameData& frame   [[buffer(1)]],
    
    texture3d<uint, access::read> indirection [[texture(2)]],
    device uint* geoPool    [[buffer(3)]],
    device uchar* matPool   [[buffer(4)]],   
     
    uint2 gid [[thread_position_in_grid]])
{
    uint width = distTex.get_width();
    uint height = distTex.get_height();
    if (gid.x >= width || gid.y >= height) return;

    float2 uv = (float2(gid) + 0.5f) / float2(width, height);
    float2 ndc = uv * 2.0f - 1.0f; 
    float3 dir = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);

    hitInfo hit = trace(camera.position, dir, indirection, geoPool);
    
    float dist = hit.hit ? length(hit.pos - camera.position) : 5000.0f;
    
    // Safety padding for the main raymarch
    dist = max(0.0f, dist - 8.0f);

    distTex.write(float4(dist, 0, 0, 0), gid);
}
