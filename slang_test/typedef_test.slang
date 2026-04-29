typedef float3 simd_float3;

[shader("compute")]
[numthreads(8,8,1)]
void typedefTest(uint3 tid : SV_DispatchThreadID)
{
    simd_float3 a = simd_float3(1,2,3);
}
