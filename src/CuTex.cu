#include "CuTex.cuh"
#include <cstring>   // std::memset
#include <cmath>     // std::abs
#include <iostream>
#include "cumath.cuh"

static inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(err) << "\n";
    }
}

CuTex::CuTex()
{
    tex_obj = 0;
    dev_ptr = nullptr;
    width = height = 0;
}

// Constructor to create linear memory and a texture object from it.
CuTex::CuTex(int _width, int _height,
            cudaChannelFormatDesc _cudaChannelFormatDesc,
             cudaTextureAddressMode _cudaTextureAddressMode,
             cudaTextureFilterMode _cudaTextureFilterMode)
{
    tex_obj = 0;
    dev_ptr = nullptr;

    width = _width;
    height = _height;

    if (width <= 0 || height <= 0) {
        std::cerr << "CuTex: width/height must be > 0\n";
        return;
    }

    // Allocate linear device memory for the float buffer.
    cudaError_t err = cudaMalloc(&dev_ptr, static_cast<size_t>(width) * height * sizeof(float));
    if (err != cudaSuccess) {
        checkCudaError(err, "cudaMalloc");
        dev_ptr = nullptr;
        return;
    }

    // Describe the linear resource for texture creation.
    cudaResourceDesc resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = dev_ptr;
    resDesc.res.linear.desc = _cudaChannelFormatDesc;
    resDesc.res.linear.sizeInBytes = static_cast<size_t>(width) * height * sizeof(float);

    // Create texture object from the linear memory.
    cudaTextureDesc texDesc;
    std::memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = _cudaTextureAddressMode;
    texDesc.addressMode[1] = _cudaTextureAddressMode;
    texDesc.filterMode = _cudaTextureFilterMode;
    texDesc.readMode = cudaReadModeElementType; // Read as element type (float)
    texDesc.normalizedCoords = 0; // Use unnormalized coordinates for direct mapping
    
    err = cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        checkCudaError(err, "cudaCreateTextureObject");
        cudaFree(dev_ptr);
        dev_ptr = nullptr;
        tex_obj = 0;
        return;
    }
}

CuTex::~CuTex()
{
    if (tex_obj != 0) {
        cudaError_t e = cudaDestroyTextureObject(tex_obj);
        if (e != cudaSuccess) checkCudaError(e, "cudaDestroyTextureObject");
        tex_obj = 0;
    }
    if (dev_ptr != nullptr) {
        cudaError_t e = cudaFree(dev_ptr);
        if (e != cudaSuccess) checkCudaError(e, "cudaFree");
        dev_ptr = nullptr;
    }
}

void CuTex::readback(float* buffer)
{
    if (!buffer) {
        std::cerr << "CuTex::readback - null host buffer\n";
        return;
    }
    if (!dev_ptr) {
        std::cerr << "CuTex::readback - no dev_ptr allocation\n";
        return;
    }
    size_t totalBytes = static_cast<size_t>(width) * height * sizeof(float);
    cudaError_t err = cudaMemcpy(
        buffer,
        dev_ptr,
        totalBytes,
        cudaMemcpyDeviceToHost
    );
    if (err != cudaSuccess) {
        checkCudaError(err, "cudaMemcpy (readback)");
    }
}
