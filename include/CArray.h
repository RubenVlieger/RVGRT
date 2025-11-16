#pragma once

#include "cumath.h"

#ifndef __METAL_VERSION__
class CArray
{
public:

    CArray() {};
    ~CArray() {};

    uint64_t getSize()  { return SIZE; }

    void fill();
    void Allocate(uint64_t size);
    void Free();

    void readback(uint32_t* buffer);
    // This is defined inside the class, so it's hidden from Metal automatically.
    uint32_t* getPtr() { return dev_data; }

private:
    uint32_t* dev_data = nullptr;
    uint64_t SIZE = 0;      // Total bytes
};

#endif