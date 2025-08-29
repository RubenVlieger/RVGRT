#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
    

#include "cumath.cuh"

class CArray {
public:
    
    CArray();
    ~CArray();
    
    // Fill the array with bits using CUDA
    uint64_t getSize();

    void fill();
    void Allocate(uint64_t size);
    void Free();


    // Read back into CPU buffer
    void readback(uint32_t* buffer);
    uint32_t* getPtr();

private:
    uint32_t* dev_data = nullptr;
    uint64_t SIZE = 0;      // Total bytes
};
