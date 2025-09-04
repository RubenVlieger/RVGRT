#include "CArray.cuh"
#include <cuda_runtime.h>
#include <iostream>

#include "cumath.cuh"
#include "TerrainGeneration.cuh"


// Kernel to fill every value inside the CArray, the for loop is made such that it does not need atomics to write. 
extern "C" __global__
void fillKernel(uint32_t* __restrict__ data, uint64_t numWords, uint64_t totalBits)
{
    // 64-bit idx to support very large worlds
    uint64_t wordIdx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (wordIdx >= numWords) return;

    uint64_t baseBit = wordIdx * 32ull;

    uint32_t w = 0u;
    for (uint64_t bit = 0; bit < 32; ++bit) 
    {
        uint64_t bitIndex = baseBit + (uint64_t)bit;

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

uint32_t* CArray::getPtr() 
{
    return dev_data;
}

uint64_t CArray::getSize() 
{
    return SIZE;
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

    fillKernel<<<blocks64, threads>>>(dev_data, numWords, totalBits);
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
CArray::CArray() 
{

}

CArray::~CArray() 
{
  
}