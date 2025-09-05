#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include "cuda_fp16.h"
#include <algorithm>

#define CUDA_CHECK(err) \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA error"); \
    }


const uint64_t SHIX = 10;
const uint64_t SHIY = 9;
const uint64_t SHIZ = 10;

const uint64_t MODX = (1<<SHIX) - 1;
const uint64_t MODY = (1<<SHIY) - 1;
const uint64_t MODZ = (1<<SHIZ) - 1;

const uint64_t SIZEX = 1<<SHIX;
const uint64_t SIZEY = 1<<SHIY;
const uint64_t SIZEZ = 1<<SHIZ;

const uint64_t BYTESIZE = SIZEX*SIZEY*SIZEZ/8;

__device__ __forceinline__ uint64_t toIndex(int3 p) 
{
    return  (((uint64_t)p.x) & MODX) | 
           ((((uint64_t)p.y) & MODY) << SHIX) | 
           ((((uint64_t)p.z) & MODZ) << (SHIX + SHIY));
}

__device__ __forceinline__ uint64_t toIndex(uint64_t x, uint64_t y, uint64_t z ) 
{
    return  (x & MODX) | 
           ((y & MODY) << SHIX) | 
           ((z & MODZ) << (SHIX + SHIY));
}

// __device__ __forceinline__ uint64_t splitBy3(uint64_t a) {
//     // keep only lower 21 bits
//     a &= 0x1fffffULL;
//     a = (a | (a << 32)) & 0x1f00000000ffffULL;
//     a = (a | (a << 16)) & 0x1f0000ff0000ffULL;
//     a = (a | (a << 8))  & 0x100f00f00f00f00fULL;
//     a = (a | (a << 4))  & 0x10c30c30c30c30c3ULL; // note: slight variant of constants to cover masks
//     a = (a | (a << 2))  & 0x1249249249249249ULL;
//     return a;
// }

// __device__ __forceinline__ uint64_t mortonEncode(uint64_t x, uint64_t y, uint64_t z) {
//     return (splitBy3(x) | (splitBy3(y) << 1) | (splitBy3(z) << 2));
// }

// // compact helper: reverse splitBy3
// __device__ __forceinline__ uint64_t compactBy3(uint64_t x) {
//     x &= 0x1249249249249249ULL;
//     x = (x ^ (x >> 2))  & 0x10c30c30c30c30c3ULL;
//     x = (x ^ (x >> 4))  & 0x100f00f00f00f00fULL;
//     x = (x ^ (x >> 8))  & 0x1f0000ff0000ffULL;
//     x = (x ^ (x >> 16)) & 0x1f00000000ffffULL;
//     x = (x ^ (x >> 32)) & 0x1fffffULL;
//     return x;
// }

// // decode morton -> x,y,z
// __device__ __forceinline__ void mortonDecode(uint64_t morton, uint32_t &outX, uint32_t &outY, uint32_t &outZ) {
//     outX = (uint32_t)compactBy3(morton);
//     outY = (uint32_t)compactBy3(morton >> 1);
//     outZ = (uint32_t)compactBy3(morton >> 2);
// }

// // --- IsSolid using Morton mapping ---
// // Replace your old IsSolid with this. It now converts (x,y,z) -> morton index.
// __device__ __forceinline__ bool IsSolid(int3 p, const uint32_t* __restrict__ bits) {
//     // bounds check (important if SIZEX, SIZEY, SIZEZ are not power-of-two)
// // #ifdef SIZEX
// //     if (p.x < 0 || p.y < 0 || p.z < 0) return false;
// //     if ((uint32_t)p.x >= (uint32_t)SIZEX || (uint32_t)p.y >= (uint32_t)SIZEY || (uint32_t)p.z >= (uint32_t)SIZEZ) return false;
// // #endif

//     uint64_t idx = mortonEncode((uint64_t)p.x, (uint64_t)p.y, (uint64_t)p.z);
//     // same bit unpacking as before:
//     return (bits[idx >> 5] >> (idx & 31)) & 1u;
// }


struct half3 {
    __half x, y, z;
};

// --- Constructors ---

__host__ __device__ inline half3 make_half3(__half r, __half g, __half b) {
    return { r, g, b };
}

__host__ __device__ inline half3 make_half3(float r, float g, float b) {
    return { __float2half(r), __float2half(g), __float2half(b) };
}

__host__ __device__ inline half3 make_half3(float3 v) {
    return { __float2half(v.x), __float2half(v.y), __float2half(v.z) };
}

__host__ __device__ inline float3 make_float3(half3 v) {
    return make_float3(__half2float(v.x), __half2float(v.y), __half2float(v.z));
}

// --- Arithmetic ops in half precision ---

__host__ __device__ inline half3 operator+(const half3& a, const half3& b) {
    return make_half3(__hadd(a.x, b.x),
                      __hadd(a.y, b.y),
                      __hadd(a.z, b.z));
}

__host__ __device__ inline half3 operator-(const half3& a, const half3& b) {
    return make_half3(__hsub(a.x, b.x),
                      __hsub(a.y, b.y),
                      __hsub(a.z, b.z));
}

__host__ __device__ inline half3 operator*(const half3& a, const half3& b) {
    return make_half3(__hmul(a.x, b.x),
                      __hmul(a.y, b.y),
                      __hmul(a.z, b.z));
}

__host__ __device__ inline half3 operator*(const half3& a, __half s) {
    return make_half3(__hmul(a.x, s),
                      __hmul(a.y, s),
                      __hmul(a.z, s));
}

__host__ __device__ inline half3 operator/(const half3& a, __half s) {
    return make_half3(__hdiv(a.x, s),
                      __hdiv(a.y, s),
                      __hdiv(a.z, s));
}

// --- Dot product & length (returns float, since sum needs higher precision) ---

__host__ __device__ inline float dot(const half3& a, const half3& b) {
    __half hx = __hmul(a.x, b.x);
    __half hy = __hmul(a.y, b.y);
    __half hz = __hmul(a.z, b.z);
    return __half2float(__hadd(__hadd(hx, hy), hz));
}

__host__ __device__ inline float length(const half3& v) {
    return sqrtf(dot(v, v));
}

// --- Normalization (returns half3, scales in float for safety) ---

__host__ __device__ inline half3 normalize(const half3& v) {
    float len = length(v);
    if (len > 0.0f) {
        float inv = 1.0f / len;
        return make_half3(__hmul(v.x, __float2half(inv)),
                          __hmul(v.y, __float2half(inv)),
                          __hmul(v.z, __float2half(inv)));
    }
    return make_half3(__float2half(0.f), __float2half(0.f), __float2half(0.f));
}


__device__ inline float clampf(float v, float a, float b) {
    return fmaxf(a, fminf(b, v));
}
__device__ inline float smoothstepf(float edge0, float edge1, float x) {
    float t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
__host__ __device__ inline float3 float3f(float x, float y, float z) { return make_float3(x, y, z); }
__host__ __device__ inline float3 float3f(float v) { return make_float3(v, v, v); }
__host__ __device__ inline float3 float3f(const float2 &xy, float z) { return make_float3(xy.x, xy.y, z); }
__host__ __device__ inline float3 float3f(float x, const float2 &yz) { return make_float3(x, yz.x, yz.y); }

__host__ __device__ inline int3 int3i(int x, int y, int z) { return make_int3(x, y, z); }
__host__ __device__ inline int3 int3i(int v) { return make_int3(v, v, v); }

__host__ __device__ inline uint3 uint3u(unsigned int x, unsigned int y, unsigned int z) { return make_uint3(x, y, z); }
__host__ __device__ inline uint3 uint3u(unsigned int v) { return make_uint3(v, v, v); }


__host__ __device__ inline int3 operator+(const int3 &a, const int3 &b) { return make_int3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline int3 operator-(const int3 &a, const int3 &b) { return make_int3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ inline int3 operator*(const int3 &a, const int3 &b) { return make_int3(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ inline int3 operator*(const int3 &a, int s) { return make_int3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ inline int3 operator*(int s, const int3 &a) { return a*s; }
__host__ __device__ inline int3 operator/(const int3 &a, const int3 &b) { return make_int3(a.x/b.x, a.y/b.y, a.z/b.z); }
__host__ __device__ inline int3 operator/(const int3 &a, int s) { return make_int3(a.x/s, a.y/s, a.z/s); }
__host__ __device__ inline int3 operator-(const int3 &a) { return make_int3(-a.x, -a.y, -a.z); }

// uint3
__host__ __device__ inline uint3 operator+(const uint3 &a, const uint3 &b) { return make_uint3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline uint3 operator-(const uint3 &a, const uint3 &b) { return make_uint3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ inline uint3 operator*(const uint3 &a, const uint3 &b) { return make_uint3(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ inline uint3 operator*(const uint3 &a, unsigned int s) { return make_uint3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ inline uint3 operator*(unsigned int s, const uint3 &a) { return a*s; }
__host__ __device__ inline uint3 operator/(const uint3 &a, const uint3 &b) { return make_uint3(a.x/b.x, a.y/b.y, a.z/b.z); }
__host__ __device__ inline uint3 operator/(const uint3 &a, unsigned int s) { return make_uint3(a.x/s, a.y/s, a.z/s); }

// -------------------
// Arithmetic operators
// -------------------
__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ inline float3 operator*(const float3 &a, const float3 &b) { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ inline float3 operator*(const float3 &a, float s) { return make_float3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ inline float3 operator*(float s, const float3 &a) { return a*s; }
__host__ __device__ inline float3 operator/(const float3 &a, const float3 &b) { return make_float3(a.x/b.x, a.y/b.y, a.z/b.z); }
__host__ __device__ inline float3 operator/(const float3 &a, float s) { return make_float3(a.x/s, a.y/s, a.z/s); }
__host__ __device__ inline float3 operator-(const float3 &a) { return make_float3(-a.x, -a.y, -a.z); }
__host__ __device__ inline bool operator==(const float3 &a, const float3 &b) { return (a.x==b.x && a.y==b.y && a.z==b.z); }
__host__ __device__ inline bool operator!=(const float3 &a, const float3 &b) { return !(a==b); }

// -------------------
// Vector functions
// -------------------
__host__ __device__ inline int dot(const int3 &a, const int3 &b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline unsigned int dot(const uint3 &a, const uint3 &b) { return a.x*b.x + a.y*b.y + a.z*b.z; }


__host__ __device__ inline float length(const float3 &v) { return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }
__host__ __device__ inline float lengthsq(const float3 &v) { return v.x*v.x + v.y*v.y + v.z*v.z; }
__host__ __device__ inline float3 normalize(const float3 &v) { float len = length(v); return v*(1.0f/len); }
__host__ __device__ inline float dot(const float3 &a, const float3 &b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline float3 cross(const float3 &a, const float3 &b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__host__ __device__ inline float distance(const float3 &a, const float3 &b) { return length(a-b); }
__host__ __device__ inline float distancesq(const float3 &a, const float3 &b) { return lengthsq(a-b); }

__host__ __device__ inline float angle(const float3 &a, const float3 &b) {
    float d = dot(normalize(a), normalize(b));
    return acosf(fminf(fmaxf(d, -1.0f), 1.0f));
}
__host__ __device__ inline float3 reflect(const float3 &I, const float3 &N) {
    return I - 2.0f * dot(I, N) * N;
}
__host__ __device__ inline float lengthsquared(const float3 &v) {
    return v.x*v.x + v.y*v.y + v.z*v.z;
}
__host__ __device__ inline float3 project(const float3 &a, const float3 &b) { return dot(a,b)/lengthsquared(b)*b; }
__host__ __device__ inline float3 reject(const float3 &a, const float3 &b) { return a - project(a,b); }

// -------------------
// Component-wise math
// -------------------
__host__ __device__ inline float3 abs(const float3 &v) { return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
__host__ __device__ inline float3 floor(const float3 &v) { return make_float3(floorf(v.x), floorf(v.y), floorf(v.z)); }
__host__ __device__ inline float3 ceil(const float3 &v) { return make_float3(ceilf(v.x), ceilf(v.y), ceilf(v.z)); }
__host__ __device__ inline float3 round(const float3 &v) { return make_float3(roundf(v.x), roundf(v.y), roundf(v.z)); }
__host__ __device__ inline float3 frac(const float3 &v) { return v - floor(v); }
__host__ __device__ inline float3 sign(const float3 &v) { return make_float3(((float)(v.x>0.0f)-(v.x<0.f)), (float)((v.y>0.f)-(v.y<0.f)), (float)((v.z>0.f)-(v.z<0.f))); }

__host__ __device__ inline float3 min3(const float3 &a, const float3 &b) { return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z)); }
__host__ __device__ inline float3 max3(const float3 &a, const float3 &b) { return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z)); }
__host__ __device__ inline float3 clamp(const float3 &v, const float3 &mn, const float3 &mx) { return min3(max3(v,mn), mx); }
__host__ __device__ inline float3 lerp(const float3 &a, const float3 &b, float t) { return a + t*(b-a); }
__host__ __device__ inline int3 min3(const int3 &a, const int3 &b) { return make_int3(std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z)); }
__host__ __device__ inline int3 max3(const int3 &a, const int3 &b) { return make_int3(std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z)); }
__host__ __device__ inline int3 clamp3(const int3 &v, int minVal, int maxVal) { return max3(min3(v,int3i(maxVal)), int3i(minVal)); }

__host__ __device__ inline uint3 min3(const uint3 &a, const uint3 &b) { return make_uint3(std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z)); }
__host__ __device__ inline uint3 max3(const uint3 &a, const uint3 &b) { return make_uint3(std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z)); }


__host__ __device__ inline float3 sqrtf3(const float3 &v) { return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z)); }
__host__ __device__ inline float3 expf3(const float3 &v) { return make_float3(expf(v.x), expf(v.y), expf(v.z)); }
__host__ __device__ inline float3 logf3(const float3 &v) { return make_float3(logf(v.x), logf(v.y), logf(v.z)); }
__host__ __device__ inline float3 sinf3(const float3 &v) { return make_float3(sinf(v.x), sinf(v.y), sinf(v.z)); }
__host__ __device__ inline float3 cosf3(const float3 &v) { return make_float3(cosf(v.x), cosf(v.y), cosf(v.z)); }


__device__ inline float3 sign3(const float3 &v) {
    return make_float3((float)((v.x > 0.f) - (v.x < 0.f)),
                       (float)((v.y > 0.f) - (v.y < 0.f)),
                       (float)((v.z > 0.f) - (v.z < 0.f)));
}

__device__ inline float3 abs3(const float3 &v) {
    return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}


__device__ __forceinline__ int3 mod256(int3 v) {
    return make_int3(v.x & 255, v.y & 255, v.z & 255);
}
