#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>


#define CUDA_CHECK(err) \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA error"); \
    }

const unsigned int SHIX = 11;
const unsigned int SHIY = 9;
const unsigned int SHIZ = 11;

const unsigned int MODX = (1<<SHIX) - 1;
const unsigned int MODY = (1<<SHIY) - 1;
const unsigned int MODZ = (1<<SHIZ) - 1;

const unsigned int SIZEX = 1<<SHIX;
const unsigned int SIZEY = 1<<SHIY;
const unsigned int SIZEZ = 1<<SHIZ;

const uint64_t BYTESIZE = SIZEX*SIZEY*SIZEZ/8;

// -------------------
// Constructors
// -------------------
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
__host__ __device__ inline uint3 operator-(const uint3 &a) { return make_uint3(-a.x, -a.y, -a.z); }

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
__host__ __device__ inline float3 sign(const float3 &v) { return make_float3((v.x>0)-(v.x<0), (v.y>0)-(v.y<0), (v.z>0)-(v.z<0)); }

__host__ __device__ inline float3 min(const float3 &a, const float3 &b) { return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z)); }
__host__ __device__ inline float3 max(const float3 &a, const float3 &b) { return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z)); }
__host__ __device__ inline float3 clamp(const float3 &v, const float3 &mn, const float3 &mx) { return min(max(v,mn), mx); }
__host__ __device__ inline float3 lerp(const float3 &a, const float3 &b, float t) { return a + t*(b-a); }

__host__ __device__ inline int3 min3(const int3 &a, const int3 &b) { return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z)); }
__host__ __device__ inline int3 max3(const int3 &a, const int3 &b) { return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)); }
__host__ __device__ inline int3 clamp3(const int3 &v, int minVal, int maxVal) { return max3(min3(v,int3i(maxVal)), int3i(minVal)); }

__host__ __device__ inline uint3 min3(const uint3 &a, const uint3 &b) { return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z)); }
__host__ __device__ inline uint3 max3(const uint3 &a, const uint3 &b) { return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)); }


__host__ __device__ inline float3 sqrtf3(const float3 &v) { return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z)); }
__host__ __device__ inline float3 expf3(const float3 &v) { return make_float3(expf(v.x), expf(v.y), expf(v.z)); }
__host__ __device__ inline float3 logf3(const float3 &v) { return make_float3(logf(v.x), logf(v.y), logf(v.z)); }
__host__ __device__ inline float3 sinf3(const float3 &v) { return make_float3(sinf(v.x), sinf(v.y), sinf(v.z)); }
__host__ __device__ inline float3 cosf3(const float3 &v) { return make_float3(cosf(v.x), cosf(v.y), cosf(v.z)); }


__device__ inline float3 sign3(const float3 &v) {
    return make_float3((v.x > 0) - (v.x < 0),
                       (v.y > 0) - (v.y < 0),
                       (v.z > 0) - (v.z < 0));
}

__device__ inline float3 abs3(const float3 &v) {
    return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}


__device__ __forceinline__ int3 mod256(int3 v) {
    return make_int3(v.x & 255, v.y & 255, v.z & 255);
}
