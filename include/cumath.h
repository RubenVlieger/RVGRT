#pragma once


#if defined(PLATFORM_METAL)
    #define GLOBAL_CONST constant
#else
    // constexpr works for both modern C++ (CPU) and CUDA
    #define GLOBAL_CONST constexpr
#endif

#if defined(__METAL_VERSION__)
/*******************************************************************************
 * METAL SHADING LANGUAGE (MSL)
 *******************************************************************************/
#include <metal_stdlib>
using namespace metal;

#define PLATFORM_METAL
#define GPU_FUNC         static
#define GPU_INLINE       inline
#define KERNEL_FUNC      kernel
#define CONST_MEM        constant
#define DEVICE_MEM       device
#define THREAD_MEM       thread
#define HOST_FUNC        

#define DEVICE_PTR(type)   device type
#define CONSTANT_PTR(type) constant type
#define THREAD_PTR(type)   threadgroup type

#define RESTRICT         
#define TEXTURE_OBJECT    texture2d<half, access::sample>
#define ARRAY_OBJECT      MTLTexture
// MSL uses these types natively
using int2 = metal::int2;
using int3 = metal::int3;
using int4 = metal::int4;
using uint2 = metal::uint2;
using uint3 = metal::uint3;
using uint4 = metal::uint4;
using float2 = metal::float2;
using float3 = metal::float3;
using float4 = metal::float4;
using mat2 = metal::float2x2;
using mat3 = metal::float3x3;
using mat4 = metal::float4x4;

#define F32(x) float(x)

#elif defined(__CUDA_ARCH__)
/*******************************************************************************
 * CUDA (Device-Side Compilation)
 *******************************************************************************/
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_fp16.h> // For half precision support

#define PLATFORM_CUDA
#define GPU_FUNC         __device__
#define GPU_INLINE       __forceinline__
#define KERNEL_FUNC      extern "C" __global__
#define CONST_MEM        __constant__
#define DEVICE_MEM       __device__
#define THREAD_MEM       __shared__
#define HOST_FUNC        __host__

#define RESTRICT         __restrict__
#define TEXTURE_OBJECT    cudaTextureObject_t
#define ARRAY_OBJECT     cudaArray_t 
// CUDA provides vector types like float2, int2, etc. We just need to alias them.
// We will define our own structs later to ensure API consistency.
using uint = unsigned int;
using half = __half;
#define F32(x) float(x)

#define DEVICE_PTR(type)   type
#define CONSTANT_PTR(type) const type
#define THREAD_PTR(type)   __shared__ type



#else
/*******************************************************************************
 * C++ (CPU-Side Compilation, potentially with CUDA host code)
 *******************************************************************************/
#include <cmath>
#include <algorithm> // for std::min/max
#include <limits>    // for infinity

#define PLATFORM_CPU

#if defined(__CUDACC__) // Compiling a .cu file with NVCC for host code
#define GPU_FUNC         __host__ __device__
#define HOST_FUNC        __host__
#include <cuda_fp16.h> // Allow host code in .cu files to see `half`
using half = __half;
#else // Compiling with a standard C++ compiler (Clang, GCC, MSVC)
#define GPU_FUNC
#define HOST_FUNC

#endif

#define GPU_INLINE       inline
#define KERNEL_FUNC      /* Not Applicable */
#define CONST_MEM        /* Not Applicable */
#define DEVICE_MEM       /* Not Applicable */
#define THREAD_MEM       /* Not Applicable */

#define DEVICE_PTR(type)   type
#define CONSTANT_PTR(type) const type
#define THREAD_PTR(type)   type

#define RESTRICT
#define TEXTURE_OBJECT    void*
#define ARRAY_OBJECT     void* 


using uint = unsigned int;
#define F32(x) static_cast<float>(x)

#endif

#if defined(PLATFORM_METAL)
#define PI (3.1415926535);
#elif defined(PLATFORM_CUDA)
GLOBAL_CONST float PI = CUDART_PI_F;
#else
GLOBAL_CONST float PI = 3.1415926535f;
#endif


//-------------------------------------------------------------------------------------------------
// 2. CPU-SIDE HALF-PRECISION FLOAT IMPLEMENTATION
//-------------------------------------------------------------------------------------------------
#if defined(PLATFORM_CPU) && !defined(__CUDACC__)
// A simple class for CPU-side half-precision float representation.
// This ensures memory layout compatibility when sending data to the GPU.
// Note: Most operations should be done in 32-bit float for performance on CPU.
class half {
protected:
    uint16_t value;

public:
    half() = default;
    // Conversion from float to half
    inline half(float f) {
        // Based on Fabian Giesen's float-to-half conversion
        uint32_t x;
        memcpy(&x, &f, sizeof(f));
        uint32_t sign = (x >> 31) & 0x0001;
        uint32_t exp = (x >> 23) & 0x00FF;
        uint32_t mant = x & 0x007FFFFF;
        if (exp == 255) { // NaN/Inf
            exp = 31;
            mant = mant ? 0x0200 : 0;
        } else if (exp > 127 + 15) { // Overflow to Inf
            exp = 31; mant = 0;
        } else if (exp <= 127 - 15) { // Underflow to denormalized
            mant = (mant | 0x00800000) >> (127 - 15 - exp + 1);
            exp = 0;
        } else {
            exp -= (127 - 15);
        }
        value = (sign << 15) | (exp << 10) | (mant >> 13);
    }

    // Conversion from half to float
    inline operator float() const {
        uint32_t sign = (value >> 15) & 0x0001;
        uint32_t exp = (value >> 10) & 0x001F;
        uint32_t mant = value & 0x03FF;
        float f;
        if (exp == 31) { // NaN/Inf
            f = std::numeric_limits<float>::infinity();
        } else if (exp == 0) { // Denormalized
            f = (mant * std::pow(2.0f, 1 - 15)) * std::pow(2.0f, -14.0f);
        } else {
            f = ((mant | 0x0400) * std::pow(2.0f, exp - 15)) * std::pow(2.0f, -10.0f);
        }
        return (sign == 1) ? -f : f;
    }
};
#endif


//-------------------------------------------------------------------------------------------------
// 3. UNIFIED VECTOR & MATRIX TYPES (CPU and CUDA side)
//    MSL platform uses its native types which are aliased above.
//-------------------------------------------------------------------------------------------------

#ifndef PLATFORM_METAL // Metal uses its own types, these are for CPU/CUDA

// Forward declarations
struct int2; struct int3; struct int4;
struct uint2; struct uint3; struct uint4;
struct float2; struct float3; struct float4;
struct half2; struct half3; struct half4;
struct mat2; struct mat3; struct mat4;

// --- floatN ---
struct alignas(8) float2 { float x, y; };
struct alignas(16) float3 { float x, y, z; };
struct alignas(16) float4 { float x, y, z, w; };

// --- intN ---
struct alignas(8) int2 { int x, y; };
struct alignas(16) int3 { int x, y, z; };
struct alignas(16) int4 { int x, y, z, w; };

// --- uintN ---
struct alignas(8) uint2 { uint x, y; };
struct alignas(16) uint3 { uint x, y, z; };
struct alignas(16) uint4 { uint x, y, z, w; };

// --- halfN ---
struct alignas(4) half2 { half x, y; };
struct alignas(8) half3 { half x, y, z; };
struct alignas(8) half4 { half x, y, z, w; };

// --- matN ---
struct alignas(8)  mat2 { float2 cols[2]; };
struct alignas(16) mat3 { float3 cols[3]; }; // Use float3 to avoid padding issues
struct alignas(16) mat4 { float4 cols[4]; };

#endif // !PLATFORM_METAL

//-------------------------------------------------------------------------------------------------
// 4. TYPE CONSTRUCTORS
//-------------------------------------------------------------------------------------------------
#if defined(PLATFORM_METAL)
// FIX: Use native Metal vector constructors for MSL.
GPU_FUNC GPU_INLINE float2 make_float2(float s) { return float2(s); }
GPU_FUNC GPU_INLINE float2 make_float2(float x, float y) { return float2(x, y); }
GPU_FUNC GPU_INLINE float3 make_float3(float s) { return float3(s); }
GPU_FUNC GPU_INLINE float3 make_float3(float x, float y, float z) { return float3(x, y, z); }
GPU_FUNC GPU_INLINE float4 make_float4(float s) { return float4(s); }
GPU_FUNC GPU_INLINE float4 make_float4(float x, float y, float z, float w) { return float4(x, y, z, w); }

GPU_FUNC GPU_INLINE int2 make_int2(int s) { return int2(s); }
GPU_FUNC GPU_INLINE int2 make_int2(int x, int y) { return int2(x, y); }
GPU_FUNC GPU_INLINE int3 make_int3(int s) { return int3(s); }
GPU_FUNC GPU_INLINE int3 make_int3(int x, int y, int z) { return int3(x, y, z); }

GPU_FUNC GPU_INLINE uint2 make_uint2(uint s) { return uint2(s); }
GPU_FUNC GPU_INLINE uint2 make_uint2(uint x, uint y) { return uint2(x, y); }

GPU_FUNC GPU_INLINE half2 make_half2(float s) { return half2(s); }
GPU_FUNC GPU_INLINE half2 make_half2(float x, float y) { return half2(x, y); }

GPU_FUNC GPU_INLINE float3 make_float3(int3 v) { return float3(v); }

#else

// floatN constructors
GPU_FUNC GPU_INLINE float2 make_float2(float s) { float2 v = {s, s}; return v; }
GPU_FUNC GPU_INLINE float2 make_float2(float x, float y) { float2 v = {x, y}; return v; }
GPU_FUNC GPU_INLINE float3 make_float3(float s) { float3 v = {s, s, s}; return v; }
GPU_FUNC GPU_INLINE float3 make_float3(float x, float y, float z) { float3 v = {x, y, z}; return v; }
GPU_FUNC GPU_INLINE float4 make_float4(float s) { float4 v = {s, s, s, s}; return v; }
GPU_FUNC GPU_INLINE float4 make_float4(float x, float y, float z, float w) { float4 v = {x, y, z, w}; return v; }

// intN constructors
GPU_FUNC GPU_INLINE int2 make_int2(int s) { int2 v = {s, s}; return v; }
GPU_FUNC GPU_INLINE int2 make_int2(int x, int y) { int2 v = {x, y}; return v; }
GPU_FUNC GPU_INLINE int3 make_int3(int s) { int3 v = {s, s, s}; return v; }
GPU_FUNC GPU_INLINE int3 make_int3(int x, int y, int z) { int3 v = {x, y, z}; return v; }

// uintN constructors
GPU_FUNC GPU_INLINE uint2 make_uint2(uint s) { uint2 v = {s, s}; return v; }
GPU_FUNC GPU_INLINE uint2 make_uint2(uint x, uint y) { uint2 v = {x, y}; return v; }

// halfN constructors
GPU_FUNC GPU_INLINE half2 make_half2(float s) { half2 v = {half(s), half(s)}; return v; }
GPU_FUNC GPU_INLINE half2 make_half2(float x, float y) { half2 v = {half(x), half(y)}; return v; }

// Type casting constructors
GPU_FUNC GPU_INLINE float3 make_float3(int3 v) { return make_float3(F32(v.x), F32(v.y), F32(v.z)); }
#endif
//-------------------------------------------------------------------------------------------------
// 5. OPERATOR OVERLOADING
//-------------------------------------------------------------------------------------------------

#if defined(PLATFORM_CPU)

// --- float2 ---
inline float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
inline float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
inline float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
inline float2 operator/(float2 a, float2 b) { return make_float2(a.x / b.x, a.y / b.y); }
inline float2 operator*(float2 v, float s)   { return make_float2(v.x * s,   v.y * s); }
inline float2 operator/(float2 v, float s)   { return make_float2(v.x / s,   v.y / s); }

// --- float3 ---
inline float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline float3 operator/(float3 a, float3 b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
inline float3 operator*(float3 v, float s)   { return make_float3(v.x * s,   v.y * s,   v.z * s); }
inline float3 operator/(float3 v, float s)   { return make_float3(v.x / s,   v.y / s,   v.z / s); }

// --- float4 ---
inline float4 operator+(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
inline float4 operator-(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
inline float4 operator*(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
inline float4 operator/(float4 a, float4 b) { return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
inline float4 operator*(float4 v, float s)   { return make_float4(v.x * s,   v.y * s,   v.z * s,   v.w * s); }
inline float4 operator/(float4 v, float s)   { return make_float4(v.x / s,   v.y / s,   v.z / s,   v.w / s); }

// --- Integer-only operators ---
inline int2 operator%(int2 a, int2 b) { return make_int2(a.x % b.x, a.y % b.y); }
inline uint2 operator&(uint2 a, uint2 b) { return make_uint2(a.x & b.x, a.y & b.y); }

// --- Matrix multiplication ---
inline float4 operator*(mat4 m, float4 v) { return m.cols[0] * v.x + m.cols[1] * v.y + m.cols[2] * v.z + m.cols[3] * v.w; }
inline mat4 operator*(mat4 a, mat4 b) {
    mat4 result;
    result.cols[0] = a * b.cols[0];
    result.cols[1] = a * b.cols[1];
    result.cols[2] = a * b.cols[2];
    result.cols[3] = a * b.cols[3];
    return result;
}

#endif 


//-------------------------------------------------------------------------------------------------
// 6. COMMON MATH FUNCTIONS & INTRINSICS
//-------------------------------------------------------------------------------------------------

#if defined(PLATFORM_METAL)

#elif defined(PLATFORM_CUDA)
    #define sqrt(x)      sqrtf(x)
    #define sin(x)       sinf(x)
    #define cos(x)       cosf(x)
    #define floor(x)     floorf(x)
    #define abs(x)       ( (x > 0) ? x : -x ) // fabsf for float, abs for int
    #define fmax(x, y)   fmaxf(x, y)
    #define fmin(x, y)   fminf(x, y)
#else // CPU
    #define sqrt(x)      std::sqrt(x)
    #define sin(x)       std::sin(x)
    #define cos(x)       std::cos(x)
    #define floor(x)     std::floor(x)
    #define abs(x)       std::abs(x)
    #define fmax(x, y)   std::max(x, y)
    #define fmin(x, y)   std::min(x, y)
#endif

GPU_FUNC GPU_INLINE float lerp(float a, float b, float t) { return a + t * (b - a); }
GPU_FUNC GPU_INLINE float3 lerp(float3 a, float3 b, float t) { return a + (b - a) * t; }
GPU_FUNC GPU_INLINE float3 floor3(float3 v) { return make_float3(floor(v.x), floor(v.y), floor(v.z)); }

GPU_FUNC GPU_INLINE float3 abs3(float3 v) { return make_float3(abs(v.x), abs(v.y), abs(v.z)); }

GPU_FUNC GPU_INLINE float3 cross(float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
#if defined(PLATFORM_CPU)
#define GPU_FUNC_INLINE inline

GPU_FUNC_INLINE float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
GPU_FUNC_INLINE float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
GPU_FUNC_INLINE float length(float3 v) { return sqrt(dot(v, v)); }
GPU_FUNC_INLINE float3 normalize(float3 v) { float l = length(v); return (l > 0.0f) ? v / l : v; }
GPU_FUNC_INLINE int2 clamp(int2 v, int minVal, int maxVal) {
    // Note: On CPU, fmin/fmax are defined by the macro above to be std::min/std::max
    return make_int2(fmin(fmax(v.x, minVal), maxVal), fmin(fmax(v.y, minVal), maxVal));
}

GPU_FUNC GPU_FUNC_INLINE float3 clamp(const float3 v, float minVal, float maxVal) {
    float3 tv = make_float3(v.x < minVal ? minVal : v.x,
                            v.y < minVal ? minVal : v.y,
                            v.z < minVal ? minVal : v.z);

    return make_float3(tv.x > maxVal ? maxVal : tv.x,
                       tv.y > maxVal ? maxVal : tv.y,
                       tv.z > maxVal ? maxVal : tv.z);
}
GPU_FUNC GPU_FUNC_INLINE float clamp(const float v, float minVal, float maxVal) {
    float tv = v < minVal ? minVal : v;
    return v > maxVal ? maxVal : v;
}




#endif


// Atomic Operations
#if defined(PLATFORM_METAL)
// Overload for atomic operations on memory shared within a threadgroup
GPU_FUNC GPU_INLINE void atomicAdd(threadgroup atomic_uint* val, uint delta) {
    atomic_fetch_add_explicit(val, delta, memory_order_relaxed);
}
// Overload for atomic operations on global device memory
GPU_FUNC GPU_INLINE void atomicAdd(device atomic_uint* val, uint delta) {
    atomic_fetch_add_explicit(val, delta, memory_order_relaxed);
}
#elif defined(PLATFORM_CUDA)
GPU_FUNC GPU_INLINE void atomicAdd(unsigned int* address, unsigned int val) { atomicAdd(address, val); }
#else
// No-op or simulated for CPU for interface compatibility
GPU_FUNC GPU_INLINE void atomicAdd(unsigned int* address, unsigned int val) { *address += val; /* WARNING: NOT THREAD SAFE! */ }
#endif



#if defined(PLATFORM_CUDA)
#define CUDA_CHECK(err) \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA error"); \
    }
#endif

#define c_sunColor make_float3(1.0f * 10.0f, 0.9f * 10.0f, 0.2f * 10.0f) ;

#if defined(PLATFORM_METAL)
    // For MSL, use #define to ensure these are expanded as compile-time literals
    #define SHIX 12
    #define SHIY 9
    #define SHIZ 12
    #define MODX ((1u<<SHIX) - 1u)
    #define MODY ((1u<<SHIY) - 1u)
    #define MODZ ((1u<<SHIZ) - 1u)
    #define SIZEX (1u<<SHIX)
    #define SIZEY (1u<<SHIY)
    #define SIZEZ (1u<<SHIZ)
    #define BYTESIZE (SIZEX*SIZEY*SIZEZ/8u)
#else
    // For C++/CUDA, constexpr is typesafe and preferred
    constexpr uint64_t SHIX = 12;
    constexpr uint64_t SHIY = 9;
    constexpr uint64_t SHIZ = 12;
    constexpr uint64_t MODX = (1ULL << SHIX) - 1;
    constexpr uint64_t MODY = (1ULL << SHIY) - 1;
    constexpr uint64_t MODZ = (1ULL << SHIZ) - 1;
    constexpr uint64_t SIZEX = 1ULL << SHIX;
    constexpr uint64_t SIZEY = 1ULL << SHIY;
    constexpr uint64_t SIZEZ = 1ULL << SHIZ;
    constexpr uint64_t BYTESIZE = SIZEX * SIZEY * SIZEZ / 8;
#endif


GPU_FUNC GPU_INLINE uint64_t toIndex(int3 p) 
{
    return  (((uint64_t)p.x) & MODX) | 
           ((((uint64_t)p.y) & MODY) << SHIX) | 
           ((((uint64_t)p.z) & MODZ) << (SHIX + SHIY));
}

GPU_FUNC GPU_INLINE uint64_t toIndex(uint64_t x, uint64_t y, uint64_t z ) 
{
    return  (x & MODX) | 
           ((y & MODY) << SHIX) | 
           ((z & MODZ) << (SHIX + SHIY));
}


#undef F32
#undef SQRT
#undef SIN
#undef COS
#undef FLOOR
#undef ABS
#undef FMAX
#undef FMIN

#if defined(PLATFORM_CPU) || defined(PLATFORM_CUDA)
    #undef sqrt
    #undef sin
    #undef cos
    #undef floor
    #undef abs
    #undef fmax
    #undef fmin
#endif