#pragma once

// ============================================================================
// UNIFIED SHADER MACROS
// 
// These macros abstract the differences between Metal and CUDA syntax.
// They are processed by the C preprocessor before compilation.
//
// IMPORTANT: This header MUST be included FIRST in all .shader files.
// Example usage:
//   #include "shader_macros.h"
//   #if defined(PLATFORM_METAL)
//   #include <metal_stdlib>
//   using namespace metal;
//   #endif
//   #include "cumath.h"
// ============================================================================

// Force platform detection if not already set
#if defined(__METAL_VERSION__) && !defined(PLATFORM_METAL)
    #define PLATFORM_METAL 1
#elif defined(__CUDA_ARCH__) && !defined(PLATFORM_CUDA)
    #define PLATFORM_CUDA 1
#endif

// Metal-specific block (only compiled by Metal compiler)
#if defined(PLATFORM_METAL) && defined(__METAL_VERSION__)
    #include <metal_stdlib>
    using namespace metal;
    
    // Texture type aliases to avoid commas in macro arguments
    typedef texture2d<float, access::read> tex2d_f32_r;
    typedef texture2d<float, access::write> tex2d_f32_w;
    typedef texture2d<float, access::sample> tex2d_f32_s;
    typedef texture2d<half, access::read> tex2d_f16_r;
    typedef texture2d<half, access::write> tex2d_f16_w;
    typedef texture2d<half, access::sample> tex2d_f16_s;
    typedef texture3d<uint, access::read> tex3d_u32;
    typedef texture2d_array<float, access::sample> tex2d_arr_f32_s;
#endif

// Note: For CUDA, we rely on the CUDA compiler defining __CUDA_ARCH__
// and the CMake preprocessing step setting PLATFORM_CUDA

// CUDA-specific texture type aliases (defined at all times for CUDA compilation)
#if defined(PLATFORM_CUDA)
    // CUDA uses cudaSurfaceObject_t for writable surfaces
    typedef cudaSurfaceObject_t tex2d_f32_w;
    typedef cudaTextureObject_t tex2d_f32_r;
    typedef cudaTextureObject_t tex2d_f32_s;
    typedef cudaTextureObject_t tex2d_f16_r;
    typedef cudaSurfaceObject_t tex2d_f16_w;
    typedef cudaTextureObject_t tex2d_f16_s;
    typedef cudaTextureObject_t tex3d_u32;
    typedef cudaTextureObject_t tex2d_arr_f32_s;
    
    // Additional type aliases for CUDA compatibility
    typedef unsigned long long ulong;
    typedef unsigned char uchar;
    
    // Conversion functions to make CUDA behave like Metal for vector constructors
    __device__ __forceinline__ float2 to_float2(int2 v) { return make_float2(v.x, v.y); }
    __device__ __forceinline__ float2 to_float2(uint2 v) { return make_float2(v.x, v.y); }
    __device__ __forceinline__ float2 to_float2(float2 v) { return v; }
    __device__ __forceinline__ float3 to_float3(int3 v) { return make_float3(v.x, v.y, v.z); }
    __device__ __forceinline__ float3 to_float3(uint3 v) { return make_float3(v.x, v.y, v.z); }
    __device__ __forceinline__ float3 to_float3(float3 v) { return v; }
    __device__ __forceinline__ float4 to_float4(float4 v) { return v; }
    __device__ __forceinline__ int2 to_int2(uint2 v) { return make_int2(v.x, v.y); }
    __device__ __forceinline__ int2 to_int2(int2 v) { return v; }
    __device__ __forceinline__ int3 to_int3(uint3 v) { return make_int3(v.x, v.y, v.z); }
    __device__ __forceinline__ int3 to_int3(int3 v) { return v; }
    __device__ __forceinline__ uint2 to_uint2(int2 v) { return make_uint2(v.x, v.y); }
    __device__ __forceinline__ uint2 to_uint2(uint2 v) { return v; }
    __device__ __forceinline__ uint3 to_uint3(int3 v) { return make_uint3(v.x, v.y, v.z); }
    __device__ __forceinline__ uint3 to_uint3(uint3 v) { return v; }
#endif

// ============================================================================
// KERNEL DECLARATION
// ============================================================================

#if defined(PLATFORM_METAL)
    #define KERNEL(name) kernel void name
#elif defined(PLATFORM_CUDA)
    #define KERNEL(name) __global__ void name
#endif

// ============================================================================
// PARAMETER QUALIFIERS
// ============================================================================

#if defined(PLATFORM_METAL)
    #define PARAM_TEXTURE_READ(type, name, slot) type name [[texture(slot)]]
    #define PARAM_TEXTURE_WRITE(type, name, slot) type name [[texture(slot)]]
    #define PARAM_BUFFER(type, name, slot) device type* name [[buffer(slot)]]
    #define PARAM_CONSTANT(type, name, slot) constant type& name [[buffer(slot)]]
#elif defined(PLATFORM_CUDA)
    #define PARAM_TEXTURE_READ(type, name, slot) type name
    #define PARAM_TEXTURE_WRITE(type, name, slot) cudaSurfaceObject_t name
    #define PARAM_BUFFER(type, name, slot) type* name
    #define PARAM_CONSTANT(type, name, slot) type name
#endif

// ============================================================================
// THREAD INDEXING
// ============================================================================

#if defined(PLATFORM_METAL)
    #define GET_GID() gid
    #define GET_GID_X() gid.x
    #define GET_GID_Y() gid.y
    #define DECLARE_GID() uint2 gid [[thread_position_in_grid]]
    #define DECLARE_TID() uint2 tid [[thread_position_in_threadgroup]]
#elif defined(PLATFORM_CUDA)
    #define GET_GID_X() (blockIdx.x * blockDim.x + threadIdx.x)
    #define GET_GID_Y() (blockIdx.y * blockDim.y + threadIdx.y)
    #define GET_GID() make_int2(GET_GID_X(), GET_GID_Y())
    #define DECLARE_GID() int _width, int _height
    #define DECLARE_TID() /* CUDA: tid computed from threadIdx */
#endif

// ============================================================================
// TEXTURE OPERATIONS
// ============================================================================

#if defined(PLATFORM_METAL)
    #define TEX_READ_2D(tex, coord) tex.read(coord)
    #define TEX_READ_3D(tex, coord) tex.read(coord).r
    #define TEX_WRITE_2D(tex, val, coord) tex.write(val, coord)
    #define TEX_SAMPLE_2D(tex, uv) tex.sample(sLinear, uv)
    #define TEX_SAMPLE_2D_ARRAY(tex, uv, idx) tex.sample(sLinear, uv, idx)
    #define TEX_GET_WIDTH(tex) tex.get_width()
    #define TEX_GET_HEIGHT(tex) tex.get_height()
    #define TEX_GET_DEPTH(tex) tex.get_depth()
#elif defined(PLATFORM_CUDA)
    #define TEX_READ_2D(surf, coord) surf2Dread<float4>(surf, (coord).x * sizeof(float4), (coord).y)
    #define TEX_READ_3D(tex, coord) tex3D<uint>(tex, (coord).x, (coord).y, (coord).z)
    #define TEX_WRITE_2D(surf, val, coord) surf2Dwrite(val, surf, (coord).x * sizeof(float4), (coord).y)
    #define TEX_SAMPLE_2D(tex, uv) tex2D<float4>(tex, (uv).x, (uv).y)
    #define TEX_SAMPLE_2D_ARRAY(tex, uv, idx) tex2DLayered<float4>(tex, (uv).x, (uv).y, idx)
    #define TEX_GET_WIDTH(tex) _tex_width
    #define TEX_GET_HEIGHT(tex) _tex_height
    #define TEX_GET_DEPTH(tex) _tex_depth
#endif

// ============================================================================
// SYNCHRONIZATION
// ============================================================================

#if defined(PLATFORM_METAL)
    #define SHARED_MEM(type, name, size) threadgroup type name[size]
    #define BARRIER_GROUP() threadgroup_barrier(mem_flags::mem_threadgroup)
#elif defined(PLATFORM_CUDA)
    #define SHARED_MEM(type, name, size) __shared__ type name[size]
    #define BARRIER_GROUP() __syncthreads()
#endif

// ============================================================================
// SAMPLER DECLARATION
// ============================================================================

#if defined(PLATFORM_METAL)
    #define DECLARE_SAMPLER(name, filter_mode, addr_mode) \
        constexpr sampler name(filter::filter_mode, address::addr_mode)
#elif defined(PLATFORM_CUDA)
    #define DECLARE_SAMPLER(name, filter_mode, addr_mode)
#endif

// ============================================================================
// BOUNDS CHECKING
// ============================================================================

#if defined(PLATFORM_METAL)
    #define CHECK_BOUNDS(tex) \
        if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return
#elif defined(PLATFORM_CUDA)
    #define CHECK_BOUNDS(width, height) \
        if (GET_GID_X() >= width || GET_GID_Y() >= height) return
#endif

// ============================================================================
// ATOMIC OPERATIONS
// ============================================================================

#if defined(PLATFORM_METAL)
    #define ATOMIC_ADD(addr, val) atomic_fetch_add_explicit(addr, val, memory_order_relaxed)
#elif defined(PLATFORM_CUDA)
    #define ATOMIC_ADD(addr, val) atomicAdd(addr, val)
#endif

// ============================================================================
// UTILITY MACROS
// ============================================================================

#if defined(PLATFORM_METAL)
    #define GET_THREADGROUP_POS() threadgroup_position_in_grid
    #define GET_THREAD_POS() thread_position_in_threadgroup
#elif defined(PLATFORM_CUDA)
    #define GET_THREADGROUP_POS() blockIdx
    #define GET_THREAD_POS() threadIdx
#endif

// ============================================================================
// FLOAT16/HALF SUPPORT
// ============================================================================

#if defined(PLATFORM_METAL)
    // Metal uses 'half' natively
#elif defined(PLATFORM_CUDA)
    // CUDA uses __half
    #define half __half
    #define half2 __half2
    #include <cuda_fp16.h>
#endif

// ============================================================================
// VECTOR CONVERSION MACROS
// ============================================================================

#if defined(PLATFORM_METAL)
    #define AS_FLOAT2(v) float2(v)
    #define AS_FLOAT3(v) float3(v)
    #define AS_INT2(v) int2(v)
    #define AS_UINT2(v) uint2(v)
    #define HALF3(x,y,z) half3(x,y,z)
    #define HALF3_FROM_FLOAT3(v) half3(v)
    #define HALF_LITERAL(v) (v##h)
    #define SELECT(t, f, cond) select(t, f, cond)
    #define ANY_ISNAN(v) any(isnan(v))
    #define ANY_ISINF(v) any(isinf(v))
#elif defined(PLATFORM_CUDA)
    #define AS_FLOAT2(v) make_float2((v).x, (v).y)
    #define AS_FLOAT3(v) make_float3((v).x, (v).y, (v).z)
    #define AS_INT2(v) make_int2((v).x, (v).y)
    #define AS_UINT2(v) make_uint2((v).x, (v).y)
    #define HALF3(x,y,z) make_half3(x, y, z)
    #define HALF3_FROM_FLOAT3(v) make_half3((v).x, (v).y, (v).z)
    #define HALF_LITERAL(v) (__float2half(v))
    #define SELECT(t, f, cond) ((cond) ? (t) : (f))
    #define ANY_ISNAN(v) (isnan((v).x) || isnan((v).y) || isnan((v).z))
    #define ANY_ISINF(v) (isinf((v).x) || isinf((v).y) || isinf((v).z))
#endif

// ============================================================================
// WORKGROUP/DIM3 SUPPORT
// ============================================================================

#if defined(PLATFORM_METAL)
    // Metal uses implicit workgroup size
    #define WORKGROUP_SIZE_2D(x, y)
#elif defined(PLATFORM_CUDA)
    // CUDA uses explicit <<<grid, block>>>
    #define WORKGROUP_SIZE_2D(x, y) /* Passed at kernel launch */
#endif

// ============================================================================
// BUFFER OPERATIONS (NEW)
// Read/write operations for buffer arrays
// ============================================================================

#if defined(PLATFORM_METAL)
    #define BUFFER_READ(buf, idx) buf[idx]
    #define BUFFER_WRITE(buf, idx, val) buf[idx] = val
    #define BUFFER_ATOMICS_SUPPORTED 1
#elif defined(PLATFORM_CUDA)
    #define BUFFER_READ(buf, idx) buf[idx]
    #define BUFFER_WRITE(buf, idx, val) buf[idx] = val
    #define BUFFER_ATOMICS_SUPPORTED 1
#endif

// ============================================================================
// 3D TEXTURE OPERATIONS (Expanded)
// ============================================================================

#if defined(PLATFORM_METAL)
    #define TEX_READ_3D_F32(tex, coord) tex.read(coord)
    #define TEX_WRITE_3D(tex, val, coord) tex.write(val, coord)
    #define TEX_READ_3D_UINT(tex, coord) tex.read(coord).r
#elif defined(PLATFORM_CUDA)
    #define TEX_READ_3D_F32(tex, coord) tex3D<float4>(tex, (coord).x, (coord).y, (coord).z)
    #define TEX_WRITE_3D(surf, val, coord) surf3Dwrite(val, surf, (coord).x * sizeof(float4), (coord).y, (coord).z)
    #define TEX_READ_3D_UINT(tex, coord) tex3D<uint>(tex, (coord).x, (coord).y, (coord).z)
#endif

// ============================================================================
// MATH FUNCTIONS (NEW)
// ============================================================================

#if defined(PLATFORM_METAL)
    #define MATH_POW(x, y) pow(x, y)
    #define MATH_SQRT(x) sqrt(x)
    #define MATH_RSQR(x) rsqrt(x)
    #define MATH_SIN(x) sin(x)
    #define MATH_COS(x) cos(x)
    #define MATH_TAN(x) tan(x)
    #define MATH_ASIN(x) asin(x)
    #define MATH_ACOS(x) acos(x)
    #define MATH_ATAN(x) atan(x)
    #define MATH_ATAN2(y, x) atan2(y, x)
    #define MATH_EXP(x) exp(x)
    #define MATH_LOG(x) log(x)
    #define MATH_LOG2(x) log2(x)
    #define MATH_FMA(a, b, c) fma(a, b, c)
    #define MATH_MIN(x, y) min(x, y)
    #define MATH_MAX(x, y) max(x, y)
    #define MATH_CLAMP(x, lo, hi) clamp(x, lo, hi)
    #define MATH_SATURATE(x) saturate(x)
    #define MATH_FRACT(x) fract(x)
    #define MATH_FLOOR(x) floor(x)
    #define MATH_CEIL(x) ceil(x)
    #define MATH_ABS(x) abs(x)
    #define MATH_DOT(a, b) dot(a, b)
    #define MATH_CROSS(a, b) cross(a, b)
    #define MATH_LENGTH(v) length(v)
    #define MATH_NORMALIZE(v) normalize(v)
    #define MATH_REFLECT(i, n) reflect(i, n)
    #define MATH_REFRACT(i, n, eta) refract(i, n, eta)
#elif defined(PLATFORM_CUDA)
    #define MATH_POW(x, y) powf(x, y)
    #define MATH_SQRT(x) sqrtf(x)
    #define MATH_RSQR(x) rsqrtf(x)
    #define MATH_SIN(x) sinf(x)
    #define MATH_COS(x) cosf(x)
    #define MATH_TAN(x) tanf(x)
    #define MATH_ASIN(x) asinf(x)
    #define MATH_ACOS(x) acosf(x)
    #define MATH_ATAN(x) atanf(x)
    #define MATH_ATAN2(y, x) atan2f(y, x)
    #define MATH_EXP(x) expf(x)
    #define MATH_LOG(x) logf(x)
    #define MATH_LOG2(x) log2f(x)
    #define MATH_FMA(a, b, c) fmaf(a, b, c)
    #define MATH_MIN(x, y) fminf(x, y)
    #define MATH_MAX(x, y) fmaxf(x, y)
    #define MATH_CLAMP(x, lo, hi) fminf(fmaxf(x, lo), hi)
    #define MATH_SATURATE(x) MATH_CLAMP(x, 0.0f, 1.0f)
    #define MATH_FRACT(x) (x - floorf(x))
    #define MATH_FLOOR(x) floorf(x)
    #define MATH_CEIL(x) ceilf(x)
    #define MATH_ABS(x) fabsf(x)
    #define MATH_DOT(a, b) dot(a, b)
    #define MATH_CROSS(a, b) cross(a, b)
    #define MATH_LENGTH(v) length(v)
    #define MATH_NORMALIZE(v) normalize(v)
    #define MATH_REFLECT(i, n) reflect(i, n)
    #define MATH_REFRACT(i, n, eta) refract(i, n, eta)
#endif

// ============================================================================
// MEMORY QUALIFIERS (NEW)
// ============================================================================

#if defined(PLATFORM_METAL)
    #define MEM_DEVICE device
    #define MEM_CONSTANT constant
    #define MEM_THREAD thread
    #define MEM_THREADGROUP threadgroup
#elif defined(PLATFORM_CUDA)
    #define MEM_DEVICE __device__
    #define MEM_CONSTANT __constant__
    #define MEM_THREAD
    #define MEM_THREADGROUP __shared__
#endif

// ============================================================================
// OPTIMIZATION HINTS (NEW)
// Loop unrolling and branch prediction hints
// ============================================================================

#if defined(PLATFORM_METAL)
    #define UNROLL_LOOP [[unroll]]
    #define NO_UNROLL_LOOP [[dont_unroll]]
#elif defined(PLATFORM_CUDA)
    #define UNROLL_LOOP #pragma unroll
    #define NO_UNROLL_LOOP
#endif

// ============================================================================
// BRANCH PREDICTION (NEW)
// ============================================================================

#if defined(PLATFORM_METAL)
    #define LIKELY(x) (x)
    #define UNLIKELY(x) (x)
#elif defined(PLATFORM_CUDA)
    #define LIKELY(x) __builtin_expect((x), 1)
    #define UNLIKELY(x) __builtin_expect((x), 0)
#endif

// ============================================================================
// MATRIX OPERATIONS (NEW)
// ============================================================================

#if defined(PLATFORM_METAL)
    #define MAT4_MUL(m, v) ((m) * (v))
    #define MAT4_IDENTITY() float4x4(1.0f)
#elif defined(PLATFORM_CUDA)
    #define MAT4_MUL(m, v) mul((m), (v))
    #define MAT4_IDENTITY() mat4_identity()
#endif

// ============================================================================
// FLOAT VECTOR CONSTRUCTORS (NEW)
// Unified constructors for float vectors
// ============================================================================

#if defined(PLATFORM_METAL)
    #define FLOAT2(x, y) float2(x, y)
    #define FLOAT3(x, y, z) float3(x, y, z)
    #define FLOAT4(x, y, z, w) float4(x, y, z, w)
    #define INT2(x, y) int2(x, y)
    #define INT3(x, y, z) int3(x, y, z)
    #define UINT2(x, y) uint2(x, y)
    #define UINT3(x, y, z) uint3(x, y, z)
#elif defined(PLATFORM_CUDA)
    #define FLOAT2(x, y) make_float2(x, y)
    #define FLOAT3(x, y, z) make_float3(x, y, z)
    #define FLOAT4(x, y, z, w) make_float4(x, y, z, w)
    #define INT2(x, y) make_int2(x, y)
    #define INT3(x, y, z) make_int3(x, y, z)
    #define UINT2(x, y) make_uint2(x, y)
    #define UINT3(x, y, z) make_uint3(x, y, z)
#endif

// ============================================================================
// CONDITIONAL COMPILATION HELPERS (NEW)
// ============================================================================

#define IF_METAL(...) IF_DEFINED(PLATFORM_METAL, __VA_ARGS__)
#define IF_CUDA(...) IF_DEFINED(PLATFORM_CUDA, __VA_ARGS__)

#if defined(PLATFORM_METAL)
    #define IF_DEFINED(platform, code) code
#else
    #define IF_DEFINED(platform, code)
#endif
