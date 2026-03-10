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
#endif

// Note: For CUDA, we rely on the CUDA compiler defining __CUDA_ARCH__
// and the CMake preprocessing step setting PLATFORM_CUDA

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
    #define PARAM_BUFFER(type, name, slot) type name [[buffer(slot)]]
    #define PARAM_CONSTANT(type, name, slot) constant type name [[buffer(slot)]]
#elif defined(PLATFORM_CUDA)
    #define PARAM_TEXTURE_READ(type, name, slot) type name
    #define PARAM_TEXTURE_WRITE(type, name, slot) cudaSurfaceObject_t name
    #define PARAM_BUFFER(type, name, slot) type name
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
    #define TEX_SAMPLE_2D(tex, uv) tex.sample(sampler, uv)
    #define TEX_SAMPLE_2D_ARRAY(tex, uv, idx) tex.sample(sampler, uv, idx)
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
    #define half3 make_half3
    #define half4 make_half4
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
