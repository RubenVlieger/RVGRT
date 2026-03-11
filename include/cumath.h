#pragma once

// Include shared system configuration first
#include "renderer/SystemConfig.h"

// Indirection Grid Flags (32-bit uint)
// 0 = Empty (Implicit)
#define FLAG_SOLID_GENERIC 1 // Fully solid (optimization, e.g. bedrock)
#define IND_OFFSET 2         // Indices start here

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
#define GPU_FUNC static
#define GPU_INLINE inline
#define KERNEL_FUNC kernel
#define CONST_MEM constant
#define DEVICE_MEM device
#define THREAD_MEM thread
#define HOST_FUNC

#define DEVICE_PTR(type) device type
#define CONSTANT_PTR(type) constant type
#define THREAD_PTR(type) threadgroup type

#define RESTRICT
#define TEXTURE_OBJECT texture2d_array<float, access::sample>
#define ARRAY_OBJECT MTLTexture

#define TEX3D_U8_R texture3d<float, access::sample>
#define TEX3D_U8_W texture3d<float, access::write>
#define TEX3D_U32_R texture3d<uint, access::read>
#define TEX3D_U32_W texture3d<uint, access::write>
#define TEX3D_U32_RW texture3d<uint, access::read_write>

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

#elif defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(_WIN32)
/*******************************************************************************
 * CUDA (Host or Device Compilation)
 *******************************************************************************/
#include <cmath>
#include <cuda_fp16.h> // For half precision support - includes half2, half3, half4
#include <cuda_runtime.h>
#include <math_constants.h>

#define PLATFORM_CUDA

#if defined(__CUDA_ARCH__)
#define GPU_FUNC __device__
#define GPU_INLINE __forceinline__
#define GPU_CONST __constant__
#else
#define GPU_FUNC __host__ __device__
#define GPU_INLINE inline
#endif

#define KERNEL_FUNC extern "C" __global__
#define CONST_MEM __constant__
#define DEVICE_MEM __device__
#define THREAD_MEM __shared__
#define HOST_FUNC __host__

#define RESTRICT __restrict__
#define TEXTURE_OBJECT cudaTextureObject_t
#define ARRAY_OBJECT cudaArray_t

// CUDA provides vector types like float2, int2, etc.
// half2, half3, half4 are already defined in cuda_fp16.h (device only)
using uint = unsigned int;
using half = __half;
#define F32(x) float(x)

#if !defined(__CUDA_ARCH__)
// For host-side compilation with NVCC, use float3 as half3 (they're compatible for our purposes)
typedef float3 half3;
#endif

#define DEVICE_PTR(type) type
#define CONSTANT_PTR(type) const type
#define THREAD_PTR(type) __shared__ type

#define TEX3D_U8_R const unsigned char *RESTRICT
#define TEX3D_U8_W unsigned char *RESTRICT
#define TEX3D_U32_R const uint32_t *RESTRICT
#define TEX3D_U32_W uint32_t *RESTRICT
#define TEX3D_U32_RW uint32_t *RESTRICT

// Matrix types for CUDA (not provided by CUDA headers)
struct mat2 { float2 cols[2]; };
struct mat3 { float3 cols[3]; };
struct mat4 { float4 cols[4]; };

// make_float3 from int3 for CUDA
GPU_FUNC GPU_INLINE float3 make_float3(int3 v) {
    return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}

// make_float3 from single float for CUDA (scalar expansion)
GPU_FUNC GPU_INLINE float3 make_float3(float s) {
    return make_float3(s, s, s);
}

// The float3 operators are defined in shader_macros.h for shader files
// For other CUDA code, they're provided by cuda_fp16.h

// Define float3 operators ONLY if not already defined
#if defined(PLATFORM_CUDA) && !defined(FLOAT3_OPS_DEFINED)
#ifdef __CUDA_ARCH__
#define VECTOR_OPS_ALREADY_DEFINED 1
#else
// For host-side compilation with NVCC - use explicit __host__ __device__
__host__ __device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ inline float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__host__ __device__ inline float3 operator/(float3 a, float3 b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
__host__ __device__ inline float3 operator*(float3 v, float s) { return make_float3(v.x * s, v.y * s, v.z * s); }
__host__ __device__ inline float3 operator/(float3 v, float s) { return make_float3(v.x / s, v.y / s, v.z / s); }
__host__ __device__ inline float3 operator*(float s, float3 v) { return v * s; }
#define FLOAT3_OPS_DEFINED 1
#endif
#endif

#if !defined(__CUDA_ARCH__)
// make_half3 for host-side CUDA - uses float3
GPU_FUNC GPU_INLINE half3 make_half3(float x, float y, float z) {
    return make_float3(x, y, z);
}
GPU_FUNC GPU_INLINE half3 make_half3(float s) {
    return make_float3(s, s, s);
}
#endif

// float4 operators for CUDA
GPU_FUNC GPU_INLINE float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// float4 scalar multiplication for CUDA
GPU_FUNC GPU_INLINE float4 operator*(float4 v, float s) {
    return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}
GPU_FUNC GPU_INLINE float4 operator*(float s, float4 v) {
    return v * s;
}

// Matrix multiplication for CUDA
GPU_FUNC GPU_INLINE float4 operator*(mat4 m, float4 v) {
    return m.cols[0] * v.x + m.cols[1] * v.y + m.cols[2] * v.z + m.cols[3] * v.w;
}
GPU_FUNC GPU_INLINE mat4 operator*(mat4 a, mat4 b) {
    mat4 result;
    result.cols[0] = a * b.cols[0];
    result.cols[1] = a * b.cols[1];
    result.cols[2] = a * b.cols[2];
    result.cols[3] = a * b.cols[3];
    return result;
}

// int3 operators for CUDA
GPU_FUNC GPU_INLINE int3 operator-(int3 a, int3 b) { return make_int3(a.x - b.x, a.y - b.y, a.z - b.z); }

// uint3 bitwise operators for CUDA
GPU_FUNC GPU_INLINE uint3 operator>>(uint3 v, int shift) { return make_uint3(v.x >> shift, v.y >> shift, v.z >> shift); }
GPU_FUNC GPU_INLINE uint3 operator&(uint3 a, uint3 b) { return make_uint3(a.x & b.x, a.y & b.y, a.z & b.z); }
GPU_FUNC GPU_INLINE uint3 operator&(uint3 v, unsigned int mask) { return make_uint3(v.x & mask, v.y & mask, v.z & mask); }

// fmin/fmax for float3 in CUDA
GPU_FUNC GPU_INLINE float3 fmin_float3(float3 a, float3 b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
GPU_FUNC GPU_INLINE float3 fmax_float3(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

// float2 operators for CUDA (scalar multiplication)
GPU_FUNC GPU_INLINE float2 operator*(float2 v, float s) { return make_float2(v.x * s, v.y * s); }
GPU_FUNC GPU_INLINE float2 operator*(float s, float2 v) { return v * s; }
GPU_FUNC GPU_INLINE float2 operator/(float2 v, float s) { return make_float2(v.x / s, v.y / s); }
GPU_FUNC GPU_INLINE float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
GPU_FUNC GPU_INLINE float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
GPU_FUNC GPU_INLINE float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }

// int2 operators for CUDA
GPU_FUNC GPU_INLINE int2 operator+(int2 a, int2 b) { return make_int2(a.x + b.x, a.y + b.y); }
GPU_FUNC GPU_INLINE int2 operator-(int2 a, int2 b) { return make_int2(a.x - b.x, a.y - b.y); }
GPU_FUNC GPU_INLINE int2 operator*(int2 v, int s) { return make_int2(v.x * s, v.y * s); }

// uint2 operators for CUDA
GPU_FUNC GPU_INLINE uint2 operator*(uint2 v, unsigned int s) { return make_uint2(v.x * s, v.y * s); }
GPU_FUNC GPU_INLINE uint2 operator+(uint2 a, uint2 b) { return make_uint2(a.x + b.x, a.y + b.y); }

// saturate for float3 in CUDA (clamp to 0-1)
GPU_FUNC GPU_INLINE float3 saturate(float3 x) {
    return make_float3(
        fmaxf(0.0f, fminf(1.0f, x.x)),
        fmaxf(0.0f, fminf(1.0f, x.y)),
        fmaxf(0.0f, fminf(1.0f, x.z))
    );
}

// saturate for float
GPU_FUNC GPU_INLINE float saturate(float x) {
    return fmaxf(0.0f, fminf(1.0f, x));
}

// mix (lerp) for float3 in CUDA
GPU_FUNC GPU_INLINE float3 mix(float3 a, float3 b, float t) {
    return make_float3(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
    );
}

// pow for float3 (element-wise, exponent is scalar)
GPU_FUNC GPU_INLINE float3 pow(float3 x, float y) {
    return make_float3(powf(x.x, y), powf(x.y, y), powf(x.z, y));
}

// floor for float2 in CUDA
GPU_FUNC GPU_INLINE float2 floor(float2 x) {
    return make_float2(floorf(x.x), floorf(x.y));
}

// smoothstep for CUDA (edge0, edge1, x)
GPU_FUNC GPU_INLINE float smoothstep(float edge0, float edge1, float x) {
    float t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

// length for float2 in CUDA
GPU_FUNC GPU_INLINE float length(float2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}

// bool3 type for CUDA
struct bool3 { bool x, y, z; GPU_FUNC GPU_INLINE bool3(bool x_, bool y_, bool z_) : x(x_), y(y_), z(z_) {} };

// mat4 * float3 for CUDA
GPU_FUNC GPU_INLINE float3 mat4_mul_float3(mat4 m, float3 v, bool translation) {
    float4 v4 = make_float4(v.x, v.y, v.z, translation ? 1.0f : 0.0f);
    float4 result = m * v4;
    return make_float3(result.x, result.y, result.z);
}

#else
/*******************************************************************************
 * C++ (CPU-Side Compilation)
 *******************************************************************************/
#include <algorithm> // for std::min/max
#include <cmath>
#include <limits> // for infinity

#define PLATFORM_CPU

#define GPU_FUNC
#define HOST_FUNC

#define GPU_INLINE inline
#define KERNEL_FUNC /* Not Applicable */
#define CONST_MEM   /* Not Applicable */
#define DEVICE_MEM  /* Not Applicable */
#define THREAD_MEM  /* Not Applicable */

#define DEVICE_PTR(type) type
#define CONSTANT_PTR(type) const type
#define THREAD_PTR(type) type

#define RESTRICT
#define TEXTURE_OBJECT void *
#define ARRAY_OBJECT void *

#define TEX3D_U8_R const unsigned char *
#define TEX3D_U8_W unsigned char *
#define TEX3D_U32_R const unsigned int *
#define TEX3D_U32_W unsigned int *
#define TEX3D_U32_RW unsigned int *

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
      exp = 31;
      mant = 0;
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

#if !defined(PLATFORM_METAL) && !defined(PLATFORM_CUDA) // Metal uses its own types, CPU uses custom types

// Forward declarations
struct int2;
struct int3;
struct int4;
struct uint2;
struct uint3;
struct uint4;
struct float2;
struct float3;
struct float4;
struct half2;
struct half3;
struct half4;
struct mat2;
struct mat3;
struct mat4;

// --- floatN ---
struct alignas(8) float2 {
  float x, y;
};
struct alignas(16) float3 {
  float x, y, z;
};
struct alignas(16) float4 {
  float x, y, z, w;
};

// --- intN ---
struct alignas(8) int2 {
  int x, y;
};
struct alignas(16) int3 {
  int x, y, z;
};
struct alignas(16) int4 {
  int x, y, z, w;
};

// --- uintN ---
struct alignas(8) uint2 {
  uint x, y;
};
struct alignas(16) uint3 {
  uint x, y, z;
};
struct alignas(16) uint4 {
  uint x, y, z, w;
};

// --- halfN ---
struct alignas(4) half2 {
  half x, y;
};
struct alignas(8) half3 {
  half x, y, z;
};
struct alignas(8) half4 {
  half x, y, z, w;
};

// --- matN ---
struct alignas(8) mat2 {
  float2 cols[2];
};
struct alignas(16) mat3 {
  float3 cols[3];
}; // Use float3 to avoid padding issues
struct alignas(16) mat4 {
  float4 cols[4];
};

GPU_FUNC GPU_INLINE half3 make_half3(float s) {
  half3 v = {half(s), half(s), half(s)};
  return v;
}
GPU_FUNC GPU_INLINE half3 make_half3(float x, float y, float z) {
  half3 v = {half(x), half(y), half(z)};
  return v;
}
// Helper to convert float3 -> half3
GPU_FUNC GPU_INLINE half3 make_half3(float3 v) {
  half3 r = {half(v.x), half(v.y), half(v.z)};
  return r;
}

// 2. Arithmetic Operators for half3
// Note: We convert to float for the math, then cast back to half
inline half3 operator+(half3 a, half3 b) {
  return make_half3((float)a.x + (float)b.x, (float)a.y + (float)b.y,
                    (float)a.z + (float)b.z);
}
inline half3 operator-(half3 a, half3 b) {
  return make_half3((float)a.x - (float)b.x, (float)a.y - (float)b.y,
                    (float)a.z - (float)b.z);
}
inline half3 operator*(half3 a, half3 b) {
  return make_half3((float)a.x * (float)b.x, (float)a.y * (float)b.y,
                    (float)a.z * (float)b.z);
}
inline half3 operator*(half3 v, half s) {
  float sf = (float)s;
  return make_half3((float)v.x * sf, (float)v.y * sf, (float)v.z * sf);
}
inline half3 operator*(half s, half3 v) { return v * s; } // Commutative

#endif // !PLATFORM_METAL

//-------------------------------------------------------------------------------------------------
// 4. TYPE CONSTRUCTORS
//-------------------------------------------------------------------------------------------------
#if defined(PLATFORM_METAL)
// FIX: Use native Metal vector constructors for MSL.
GPU_FUNC GPU_INLINE float2 make_float2(float s) { return float2(s); }
GPU_FUNC GPU_INLINE float2 make_float2(float x, float y) {
  return float2(x, y);
}
GPU_FUNC GPU_INLINE float3 make_float3(float s) { return float3(s); }
GPU_FUNC GPU_INLINE float3 make_float3(float x, float y, float z) {
  return float3(x, y, z);
}
GPU_FUNC GPU_INLINE float4 make_float4(float s) { return float4(s); }
GPU_FUNC GPU_INLINE float4 make_float4(float x, float y, float z, float w) {
  return float4(x, y, z, w);
}

GPU_FUNC GPU_INLINE int2 make_int2(int s) { return int2(s); }
GPU_FUNC GPU_INLINE int2 make_int2(int x, int y) { return int2(x, y); }
GPU_FUNC GPU_INLINE int3 make_int3(int s) { return int3(s); }
GPU_FUNC GPU_INLINE int3 make_int3(int x, int y, int z) {
  return int3(x, y, z);
}

GPU_FUNC GPU_INLINE uint2 make_uint2(uint s) { return uint2(s); }
GPU_FUNC GPU_INLINE uint2 make_uint2(uint x, uint y) { return uint2(x, y); }

GPU_FUNC GPU_INLINE half2 make_half2(float s) { return half2(s); }
GPU_FUNC GPU_INLINE half2 make_half2(float x, float y) { return half2(x, y); }

GPU_FUNC GPU_INLINE half3 make_half3(half s) { return half3(s); }
GPU_FUNC GPU_INLINE half3 make_half3(half x, half y, half z) {
  return half3(x, y, z);
}

GPU_FUNC GPU_INLINE float3 make_float3(int3 v) { return float3(v); }
GPU_FUNC GPU_INLINE float3 make_float3(half3 v) { return float3(v); }

#elif !defined(PLATFORM_CUDA)

// floatN constructors
GPU_FUNC GPU_INLINE float2 make_float2(float s) {
  float2 v = {s, s};
  return v;
}
GPU_FUNC GPU_INLINE float2 make_float2(float x, float y) {
  float2 v = {x, y};
  return v;
}
GPU_FUNC GPU_INLINE float3 make_float3(float s) {
  float3 v = {s, s, s};
  return v;
}
GPU_FUNC GPU_INLINE float3 make_float3(float x, float y, float z) {
  float3 v = {x, y, z};
  return v;
}
GPU_FUNC GPU_INLINE float4 make_float4(float s) {
  float4 v = {s, s, s, s};
  return v;
}
GPU_FUNC GPU_INLINE float4 make_float4(float x, float y, float z, float w) {
  float4 v = {x, y, z, w};
  return v;
}

// intN constructors
GPU_FUNC GPU_INLINE int2 make_int2(int s) {
  int2 v = {s, s};
  return v;
}
GPU_FUNC GPU_INLINE int2 make_int2(int x, int y) {
  int2 v = {x, y};
  return v;
}
GPU_FUNC GPU_INLINE int3 make_int3(int s) {
  int3 v = {s, s, s};
  return v;
}
GPU_FUNC GPU_INLINE int3 make_int3(int x, int y, int z) {
  int3 v = {x, y, z};
  return v;
}

// uintN constructors
GPU_FUNC GPU_INLINE uint2 make_uint2(uint s) {
  uint2 v = {s, s};
  return v;
}
GPU_FUNC GPU_INLINE uint2 make_uint2(uint x, uint y) {
  uint2 v = {x, y};
  return v;
}

// halfN constructors
GPU_FUNC GPU_INLINE half2 make_half2(float s) {
  half2 v = {half(s), half(s)};
  return v;
}
GPU_FUNC GPU_INLINE half2 make_half2(float x, float y) {
  half2 v = {half(x), half(y)};
  return v;
}

// Type casting constructors
GPU_FUNC GPU_INLINE float3 make_float3(int3 v) {
  return make_float3(F32(v.x), F32(v.y), F32(v.z));
}
#endif
//-------------------------------------------------------------------------------------------------
// 5. OPERATOR OVERLOADING
//-------------------------------------------------------------------------------------------------

#if defined(PLATFORM_CPU)

// --- float2 ---
inline float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
inline float2 operator-(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}
inline float2 operator*(float2 a, float2 b) {
  return make_float2(a.x * b.x, a.y * b.y);
}
inline float2 operator/(float2 a, float2 b) {
  return make_float2(a.x / b.x, a.y / b.y);
}
inline float2 operator*(float2 v, float s) {
  return make_float2(v.x * s, v.y * s);
}
inline float2 operator/(float2 v, float s) {
  return make_float2(v.x / s, v.y / s);
}

// --- float3 ---
inline float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline float3 operator*(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline float3 operator/(float3 a, float3 b) {
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline float3 operator*(float3 v, float s) {
  return make_float3(v.x * s, v.y * s, v.z * s);
}
inline float3 operator/(float3 v, float s) {
  return make_float3(v.x / s, v.y / s, v.z / s);
}

// --- float4 ---
inline float4 operator+(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline float4 operator-(float4 a, float4 b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline float4 operator*(float4 a, float4 b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline float4 operator/(float4 a, float4 b) {
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline float4 operator*(float4 v, float s) {
  return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}
inline float4 operator/(float4 v, float s) {
  return make_float4(v.x / s, v.y / s, v.z / s, v.w / s);
}

// --- Integer-only operators ---
inline int2 operator%(int2 a, int2 b) {
  return make_int2(a.x % b.x, a.y % b.y);
}
inline uint2 operator&(uint2 a, uint2 b) {
  return make_uint2(a.x & b.x, a.y & b.y);
}

// --- Matrix multiplication (CPU only, CUDA has its own version) ---
#if defined(PLATFORM_CPU)
inline float4 operator*(mat4 m, float4 v) {
  return m.cols[0] * v.x + m.cols[1] * v.y + m.cols[2] * v.z + m.cols[3] * v.w;
}
inline mat4 operator*(mat4 a, mat4 b) {
  mat4 result;
  result.cols[0] = a * b.cols[0];
  result.cols[1] = a * b.cols[1];
  result.cols[2] = a * b.cols[2];
  result.cols[3] = a * b.cols[3];
  return result;
}
#endif

#endif

//-------------------------------------------------------------------------------------------------
// 6. COMMON MATH FUNCTIONS & INTRINSICS
//-------------------------------------------------------------------------------------------------

#if defined(PLATFORM_METAL)

#elif defined(PLATFORM_CUDA)
#define sqrt(x) sqrtf(x)
#define sin(x) sinf(x)
#define cos(x) cosf(x)
#define floor(x) floorf(x)
#define abs(x) ((x > 0) ? x : -x) // fabsf for float, abs for int
#define fmax(x, y) fmaxf(x, y)
#define fmin(x, y) fminf(x, y)
#else // CPU
#define sqrt(x) std::sqrt(x)
#define sin(x) std::sin(x)
#define cos(x) std::cos(x)
#define floor(x) std::floor(x)
#define abs(x) std::abs(x)
#define fmax(x, y) std::max(x, y)
#define fmin(x, y) std::min(x, y)
#endif

GPU_FUNC GPU_INLINE float lerp(float a, float b, float t) {
  return a + t * (b - a);
}
GPU_FUNC GPU_INLINE float3 lerp(float3 a, float3 b, float t) {
  return make_float3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
}

GPU_FUNC GPU_INLINE float3 floor3(float3 v) {
  return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

GPU_FUNC GPU_INLINE float3 abs3(float3 v) {
  return make_float3(abs(v.x), abs(v.y), abs(v.z));
}

#if defined(PLATFORM_CUDA)
#define GPU_FUNC_INLINE __host__ __device__ inline
#elif defined(PLATFORM_CPU)
#define GPU_FUNC_INLINE inline
#endif

#if defined(PLATFORM_CPU) || defined(PLATFORM_CUDA)
GPU_FUNC_INLINE float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
GPU_FUNC_INLINE float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
GPU_FUNC_INLINE float length(float3 v) { return sqrt(dot(v, v)); }
GPU_FUNC_INLINE float3 normalize(float3 v) {
  float l = length(v);
  float invL = (l > 0.0f) ? 1.0f / l : 0.0f;
  return make_float3(v.x * invL, v.y * invL, v.z * invL);
}
GPU_FUNC_INLINE int2 clamp(int2 v, int minVal, int maxVal) {

  return make_int2(fmin(fmax(v.x, minVal), maxVal),
                   fmin(fmax(v.y, minVal), maxVal));
}
GPU_FUNC GPU_INLINE float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

GPU_FUNC GPU_INLINE float3 reflect(float3 i, float3 n) {
  float d = 2.0f * (i.x * n.x + i.y * n.y + i.z * n.z);
  return make_float3(i.x - d * n.x, i.y - d * n.y, i.z - d * n.z);
}

GPU_FUNC GPU_FUNC_INLINE float3 clamp(const float3 v, float minVal,
                                      float maxVal) {
  float3 tv =
      make_float3(v.x < minVal ? minVal : v.x, v.y < minVal ? minVal : v.y,
                  v.z < minVal ? minVal : v.z);

  return make_float3(tv.x > maxVal ? maxVal : tv.x,
                     tv.y > maxVal ? maxVal : tv.y,
                     tv.z > maxVal ? maxVal : tv.z);
}
GPU_FUNC GPU_FUNC_INLINE float clamp(const float v, float minVal,
                                     float maxVal) {
  float tv = v < minVal ? minVal : v;
  return v > maxVal ? maxVal : v;
}

#endif

// Atomic Operations
#if defined(PLATFORM_METAL)
// Overload for atomic operations on memory shared within a threadgroup
GPU_FUNC GPU_INLINE void atomicAdd(threadgroup atomic_uint *val, uint delta) {
  atomic_fetch_add_explicit(val, delta, memory_order_relaxed);
}
// Overload for atomic operations on global device memory
GPU_FUNC GPU_INLINE void atomicAdd(device atomic_uint *val, uint delta) {
  atomic_fetch_add_explicit(val, delta, memory_order_relaxed);
}
#elif defined(PLATFORM_CUDA)
// CUDA has built-in atomicAdd - don't redefine it
// Only define wrapper for device code
#if defined(__CUDA_ARCH__)
GPU_FUNC GPU_INLINE void localAtomicAdd(unsigned int *address, unsigned int val) {
  atomicAdd(address, val);
}
#endif
#else
// No-op or simulated for CPU for interface compatibility
GPU_FUNC GPU_INLINE void atomicAdd(unsigned int *address, unsigned int val) {
  *address += val; /* WARNING: NOT THREAD SAFE! */
}
#endif

#if defined(PLATFORM_CUDA)
#define CUDA_CHECK(err)                                                        \
  if (err != cudaSuccess) {                                                    \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;       \
    throw std::runtime_error("CUDA error");                                    \
  }
#endif

#if defined(PLATFORM_METAL)
#define c_sunColor make_half3(4.5h, 3.6h, 3.0h)
#else
// c_sunColor is defined in the shader files for CUDA
#endif

#if defined(PLATFORM_METAL)
// For MSL, use #define to ensure these are expanded as compile-time literals
#define SHIX 12
#define SHIY 9
#define SHIZ 12
// World dimension aliases from SystemConfig.h
#define SHIX WORLD_SHIFT_X
#define SHIY WORLD_SHIFT_Y
#define SHIZ WORLD_SHIFT_Z
#define MODX WORLD_MASK_X
#define MODY WORLD_MASK_Y
#define MODZ WORLD_MASK_Z
#define SIZEX WORLD_SIZE_X
#define SIZEY WORLD_SIZE_Y
#define SIZEZ WORLD_SIZE_Z
#define BYTESIZE WORLD_BYTE_SIZE
#else
// For C++/CUDA, constexpr aliases from SystemConfig.h
constexpr uint64_t SHIX = WORLD_SHIFT_X;
constexpr uint64_t SHIY = WORLD_SHIFT_Y;
constexpr uint64_t SHIZ = WORLD_SHIFT_Z;
constexpr uint64_t MODX = WORLD_MASK_X;
constexpr uint64_t MODY = WORLD_MASK_Y;
constexpr uint64_t MODZ = WORLD_MASK_Z;
constexpr uint64_t SIZEX = WORLD_SIZE_X;
constexpr uint64_t SIZEY = WORLD_SIZE_Y;
constexpr uint64_t SIZEZ = WORLD_SIZE_Z;
constexpr uint64_t BYTESIZE = WORLD_BYTE_SIZE;
#endif

GPU_FUNC GPU_INLINE uint64_t toIndex(int3 p) {
  return (((uint64_t)p.x) & MODX) | ((((uint64_t)p.y) & MODY) << SHIX) |
         ((((uint64_t)p.z) & MODZ) << (SHIX + SHIY));
}

GPU_FUNC GPU_INLINE uint64_t toIndex(uint64_t x, uint64_t y, uint64_t z) {
  return (x & MODX) | ((y & MODY) << SHIX) | ((z & MODZ) << (SHIX + SHIY));
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

// =========================================================
// TEXTURE ATLAS INDICES (Additional texture-specific indices)
// =========================================================
// row 0
#define TEX_SLAB_SIDE 5
#define TEX_SLAB_TOP 6
#define TEX_BRICK 7
#define TEX_TNT_SIDE 8
#define TEX_TNT_TOP 9
#define TEX_TNT_BOT 10
#define TEX_WEB 11
#define TEX_ROSE 12
#define TEX_FLOWER 13
#define TEX_WATER 14
#define TEX_SAPLING 15

// --- Row 1 ---
#define TEX_COBBLE 16
#define TEX_BEDROCK 17
#define TEX_SAND 18
#define TEX_GRAVEL 19
#define TEX_LOG_SIDE 20
#define TEX_LOG_TOP 21
#define TEX_IRON_BLK 22
#define TEX_GOLD_BLK 23
#define TEX_DIAM_BLK 24
#define TEX_CHEST_TOP 25
#define TEX_CHEST_SIDE 26
#define TEX_CHEST_FRONT 27
#define TEX_MUSHROOM_RED 28
#define TEX_MUSHROOM_BRN 29

// --- Row 2 ---
#define TEX_GOLD_ORE 32
#define TEX_IRON_ORE 33
#define TEX_COAL_ORE 34
#define TEX_BOOKSHELF 35
#define TEX_MOSSY 36
#define TEX_OBSIDIAN 37
#define TEX_GRID 38
#define TEX_TALLGRASS 39
#define TEX_CRAFT_TOP 43
#define TEX_CRAFT_FRONT 44
#define TEX_CRAFT_SIDE 45

// --- Row 3 ---
#define TEX_SPONGE 48
#define TEX_GLASS 49
#define TEX_DIAM_ORE 50
#define TEX_REDSTONE_ORE 51
#define TEX_LEAVES 52
#define TEX_LEAVES_OPAQUE 53
#define TEX_STONE_BRICK 54
#define TEX_DEAD_BUSH 55
#define TEX_FERN 56

// --- Row 4 ---
#define TEX_WOOL_WHITE 64
#define TEX_SPAWNER 65
#define TEX_SNOW 66
#define TEX_ICE 67
#define TEX_GRASS_SNOW 68
#define TEX_CACTUS_TOP 69
#define TEX_CACTUS_SIDE 70
#define TEX_CACTUS_IN 71
#define TEX_CLAY 72
#define TEX_REEDS 73
#define TEX_NOTEBLOCK 74
#define TEX_JUKEBOX 75

// --- Row 5 ---
#define TEX_TORCH 80
#define TEX_DOOR_W_UP 81
#define TEX_DOOR_I_UP 82
#define TEX_LADDER 83
#define TEX_TRAPDOOR 84
#define TEX_IRON_BARS 85
#define TEX_FARMLAND_WET 86
#define TEX_FARMLAND_DRY 87
#define TEX_WHEAT_0 88
#define TEX_WHEAT_7 95

// --- Row 6 ---
#define TEX_LEVER 96
#define TEX_DOOR_W_DN 97
#define TEX_DOOR_I_DN 98
#define TEX_REDTORCH_ON 99
#define TEX_PUMPKIN_TOP 102
#define TEX_PUMPKIN_SIDE 103
#define TEX_PUMPKIN_FACE 104
#define TEX_PUMPKIN_OFF 119

// --- Row 7 ---
#define TEX_RAIL_CORNER 112
#define TEX_WOOL_BLACK 113
#define TEX_WOOL_GRAY 114
#define TEX_RAIL_STR 128

// --- Row 8 ---
#define TEX_LAPIS_BLK 144

// --- Row 9 ---
#define TEX_LAPIS_ORE 160

// --- Row 12 ---
#define TEX_SANDSTONE_TOP 192
#define TEX_SANDSTONE_SID 193
#define TEX_SANDSTONE_BOT 194

// --- Row 14 ---
#define TEX_NETHERRACK 224
#define TEX_SOULSAND 225
#define TEX_GLOWSTONE 226
#define TEX_PISTON_TOP 227
#define TEX_PISTON_SIDE 228
#define TEX_PISTON_BOT 229
#define TEX_PISTON_IN 230
