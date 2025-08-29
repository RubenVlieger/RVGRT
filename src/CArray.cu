#include "CArray.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <math_constants.h> // for CUDA constants like CUDART_PI_F

#include "cumath.cuh"


// helpers
__device__ inline float fractf_dev(float x) { return x - floorf(x); }
__device__ inline float dot3(float ax, float ay, float az, float bx, float by, float bz) {
    return ax*bx + ay*by + az*bz;
}
__device__ inline float length3(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z);
}
__device__ inline void normalize3(float &x, float &y, float &z) {
    float L = length3(x,y,z);
    if (L > 0.0f) { x /= L; y /= L; z /= L; }
}

// device noise3D matching your CPU version (returns -1..1)
__device__ float noise3D(float x, float y, float z) {
    float d = sinf(x*12.9898f + y*78.233f + z*128.852f) * 43758.5453f;
    return fractf_dev(d) * 2.0f - 1.0f;
}

// device simplex3D port (returns same scale as CPU)
__device__ float simplex3D(float px, float py, float pz) {
    const float f3 = 1.0f / 3.0f;
    float s = (px + py + pz) * f3;
    int i = int(floorf(px + s));
    int j = int(floorf(py + s));
    int k = int(floorf(pz + s));

    const float g3 = 1.0f / 6.0f;
    float t = float(i + j + k) * g3;
    float x0 = float(i) - t; x0 = px - x0;
    float y0 = float(j) - t; y0 = py - y0;
    float z0 = float(k) - t; z0 = pz - z0;

    int i1, j1, k1;
    int i2, j2, k2;

    if (x0 >= y0) {
        if (y0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
        else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
        else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    } else {
        if (y0 < z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
        else if (x0 < z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
        else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }

    float x1 = x0 - float(i1) + g3;
    float y1 = y0 - float(j1) + g3;
    float z1 = z0 - float(k1) + g3;
    float x2 = x0 - float(i2) + 2.0f*g3;
    float y2 = y0 - float(j2) + 2.0f*g3;
    float z2 = z0 - float(k2) + 2.0f*g3;
    float x3 = x0 - 1.0f + 3.0f*g3;
    float y3 = y0 - 1.0f + 3.0f*g3;
    float z3 = z0 - 1.0f + 3.0f*g3;

    // integer lattice points
    int i3 = i+0, j3 = j+0, k3 = k+0;
    int i_1 = i + i1, j_1 = j + j1, k_1 = k + k1;
    int i_2 = i + i2, j_2 = j + j2, k_2 = k + k2;
    int i_3 = i + 1, j_3 = j + 1, k_3 = k + 1;

    // gradient vectors using device noise3D_dev to build pseudo-random gradients
    float gx0 = noise3D(float(i3), float(j3), float(k3));
    float gy0 = noise3D(float(i3)*2.01f, float(j3)*2.01f, float(k3)*2.01f);
    float gz0 = noise3D(float(i3)*2.02f, float(j3)*2.02f, float(k3)*2.02f);
    normalize3(gx0, gy0, gz0);

    float gx1 = noise3D(float(i_1), float(j_1), float(k_1));
    float gy1 = noise3D(float(i_1)*2.01f, float(j_1)*2.01f, float(k_1)*2.01f);
    float gz1 = noise3D(float(i_1)*2.02f, float(j_1)*2.02f, float(k_1)*2.02f);
    normalize3(gx1, gy1, gz1);

    float gx2 = noise3D(float(i_2), float(j_2), float(k_2));
    float gy2 = noise3D(float(i_2)*2.01f, float(j_2)*2.01f, float(k_2)*2.01f);
    float gz2 = noise3D(float(i_2)*2.02f, float(j_2)*2.02f, float(k_2)*2.02f);
    normalize3(gx2, gy2, gz2);

    float gx3 = noise3D(float(i_3), float(j_3), float(k_3));
    float gy3 = noise3D(float(i_3)*2.01f, float(j_3)*2.01f, float(k_3)*2.01f);
    float gz3 = noise3D(float(i_3)*2.02f, float(j_3)*2.02f, float(k_3)*2.02f);
    normalize3(gx3, gy3, gz3);

    float n0 = 0.0f, n1 = 0.0f, n2 = 0.0f, n3 = 0.0f;

    float t0 = 0.5f - x0*x0 - y0*y0 - z0*z0;
    if (t0 >= 0.0f) {
        t0 *= t0;
        n0 = t0 * t0 * dot3(gx0, gy0, gz0, x0, y0, z0);
    }
    float t1 = 0.5f - x1*x1 - y1*y1 - z1*z1;
    if (t1 >= 0.0f) {
        t1 *= t1;
        n1 = t1 * t1 * dot3(gx1, gy1, gz1, x1, y1, z1);
    }
    float t2 = 0.5f - x2*x2 - y2*y2 - z2*z2;
    if (t2 >= 0.0f) {
        t2 *= t2;
        n2 = t2 * t2 * dot3(gx2, gy2, gz2, x2, y2, z2);
    }
    float t3 = 0.5f - x3*x3 - y3*y3 - z3*z3;
    if (t3 >= 0.0f) {
        t3 *= t3;
        n3 = t3 * t3 * dot3(gx3, gy3, gz3, x3, y3, z3);
    }

    return 96.0f * (n0 + n1 + n2 + n3);
}

__device__ float Evaluate(float x, float y, float z) {
    return simplex3D(x * 0.005f, y * 0.005f, z * 0.005f);
}


extern "C" __global__
void fillKernelWords(uint32_t* __restrict__ data, uint64_t numWords, uint64_t totalBits)
{
    // 64-bit idx to support very large worlds
    uint64_t wordIdx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (wordIdx >= numWords) return;

    // compute base bit index for this 32-bit word
    uint64_t baseBit = wordIdx * 32ull;

    // build the 32-bit word locally (no atomics)
    uint32_t w = 0u;

    for (uint64_t bit = 0; bit < 32; ++bit) {
        uint64_t bitIndex = baseBit + (uint64_t)bit;
        if (bitIndex >= totalBits) break; // in case totalBits isn't divisible by 32

        // convert bitIndex -> (x,y,z) using your shifts/masks or SIZEX,SIZEY,SIZEZ
        // Example if SIZEX, SIZEY, SIZEZ are powers of two and you use shifts:
        // Assuming you have SHIX, SHIY, SHIZ and MODX/MODY/MODZ defined appropriately
        uint64_t idx = (uint64_t)bitIndex; // safe if totalBits < 2^32; else compute using 64bit math below

        // If your world might exceed 2^32 bits, compute x,y,z using 64-bit ops:
        // uint64_t z = bitIndex >> (SHIX + SHIY);
        // uint64_t y = (bitIndex >> SHIX) & MODY;
        // uint64_t x = bitIndex & MODX;

        // I'll show the 64-bit variant:
        uint64_t z = bitIndex >> (SHIX + SHIY);
        uint64_t y = (bitIndex >> SHIX) & (uint64_t)MODY;
        uint64_t x = bitIndex & (uint64_t)MODX;

        // call Evaluate (adapted to accept float/double as you have)
        float v = Evaluate((float)x, (float)y, (float)z);

        bool solid = v > 0.7f;
        if (solid) {
            w |= (1u << bit);
        }
    }

    // store with a single write
    data[wordIdx] = w;
}

// // Kernel to fill the 3D bits array based on Evaluate
// __global__ void fillKernel(uint32_t* data) 
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if(idx >= (1 << (SHIX + SHIY + SHIX))) return;

//     int z = idx >> (SHIX + SHIY); // idx / (256*256*SHIX >> SHIY;          // idx / (256*256)
//     int y = (idx >> SHIX) & MODY;   // (idx / 256) % 256
//     int x = idx & MODX;          // idx % 256

//     float value = Evaluate((float)x, (float)y, (float)z);
//     bool solid = value > 0.7f;

//     if(solid) {
//         int uintIdx = idx >> 5;      // divide by 32
//         int bitIdx  = idx & 31;      // modulo 32
//         atomicOr(&data[uintIdx], 1u << bitIdx);
//     }
// }

uint32_t* CArray::getPtr() {
    return dev_data;
}

uint64_t CArray::getSize() {
    return SIZE;
}

void CArray::fill() 
{
    if(SIZE == -1)  {
        std::cout << "ERROR CARRAY NOT ALLOCATED" << std::endl;
        exit(1);
    }

    CUDA_CHECK(cudaMemset(dev_data, 0, SIZE));



    uint64_t totalBits = (uint64_t)SIZEX * (uint64_t)SIZEY * (uint64_t)SIZEZ;
    uint64_t numWords  = (totalBits + 31ull) / 32ull;

    uint64_t threads = 256;
    uint64_t blocks64 = (numWords + threads - 1) / threads;

    std::cout << blocks64 << std::endl;

    // If blocks64 fits in a single grid dimension:
    fillKernelWords<<<blocks64, threads>>>(dev_data, numWords, totalBits);


    // int threads = 256;
    // int blocks = (SIZE*8 + threads - 1) / threads;
    // fillKernel<<<blocks, threads>>>(dev_data);
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}


void CArray::readback(uint32_t* buffer) 
{
    if(!buffer) {
        std::cout << "NO BUFFER IS INITIALIZED" << std::endl;
        exit(1);
    }
    cudaMemcpy(buffer, dev_data, SIZE, cudaMemcpyDeviceToHost);
}

void CArray::Allocate(uint64_t _size)
{
    SIZE = _size;
    if(!dev_data)
    {
        std::cout << "CREATING ARRAY" << std::endl;
        CUDA_CHECK(cudaMalloc(&dev_data, SIZE));
    }
}

void CArray::Free()
{
    if(dev_data)
    {
        cudaFree(dev_data);
        dev_data = nullptr;
    }
}
CArray::CArray() 
{

}

CArray::~CArray() 
{
  
}