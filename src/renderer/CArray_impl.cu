#include "CArray.h"
#include <cuda_runtime.h>
#include <iostream>
#include "cumath.h"
#include "TerrainGeneration.h"

extern "C" __global__ void fillKernel(uint32_t* __restrict__ data, uint64_t numWords)
{
    // compute *linear* block index (works for any gridDim.x/y/z)
    uint64_t linearBlockIdx = (uint64_t)blockIdx.x
                            + (uint64_t)blockIdx.y * (uint64_t)gridDim.x
                            + (uint64_t)blockIdx.z * (uint64_t)gridDim.x * (uint64_t)gridDim.y;

    uint64_t wordIdx = linearBlockIdx * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (wordIdx >= numWords) return;

    uint64_t baseBit = wordIdx * 32ull;
    uint32_t w = 0u;
    for (uint64_t bit = 0; bit < 32; ++bit) {
        uint64_t bitIndex = baseBit + bit;
        uint64_t z = bitIndex >> (SHIX + SHIY);
        uint64_t y = (bitIndex >> SHIX) & (uint64_t)MODY;
        uint64_t x = bitIndex & (uint64_t)MODX;
        float v = Evaluate((float)x, (float)y, (float)z);
        if (v > 0.7f) w |= (1u << bit);
    }
    data[wordIdx] = w;
}



void CArray::fill() 
{
    if(SIZE == -1)  {
        std::cout << "ERROR CARRAY NOT ALLOCATED" << std::endl;
        exit(1);
    }
    CUDA_CHECK(cudaMemset(dev_data, 0, SIZE));
    CUDA_CHECK(cudaGetLastError());

    uint64_t totalBits = (uint64_t)SIZEX * (uint64_t)SIZEY * (uint64_t)SIZEZ;
    uint64_t numWords  = (totalBits + 31ull) / 32ull;

    const unsigned int threads = 256;
    unsigned int blocks64 = (unsigned int)((numWords + (uint64_t)threads - 1ull) / (uint64_t)threads);

    fillKernel<<<blocks64, threads>>>(dev_data, numWords);
    CUDA_CHECK(cudaGetLastError());
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
