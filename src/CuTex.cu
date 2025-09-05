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

CuTex::CuTex(CuTex&& other) noexcept
{
    tex_obj = other.tex_obj;
    dev_ptr = other.dev_ptr;
    width = other.width;
    height = other.height;

    // leave other in a safe-to-destruct state
    other.tex_obj = 0;
    other.dev_ptr = nullptr;
    other.width = other.height = 0;
}

// Move assignment
CuTex& CuTex::operator=(CuTex&& other) noexcept
{
    if (this != &other) {
        // free our resources first
        if (tex_obj != 0) { cudaDestroyTextureObject(tex_obj); tex_obj = 0; }
        if (dev_ptr != nullptr) { cudaFree(dev_ptr); dev_ptr = nullptr; }

        // steal
        tex_obj = other.tex_obj;
        dev_ptr = other.dev_ptr;
        width = other.width;
        height = other.height;

        // null out other
        other.tex_obj = 0;
        other.dev_ptr = nullptr;
        other.width = other.height = 0;
    }
    return *this;
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

    size_t pitch = 0;
    // Allocate pitched device memory
    cudaError_t err = cudaMallocPitch(reinterpret_cast<void**>(&dev_ptr),
                                      &pitch,
                                      width * sizeof(float),
                                      height);
    if (err != cudaSuccess) {
        checkCudaError(err, "cudaMallocPitch");
        dev_ptr = nullptr;
        return;
    }

    // Zero the memory
    err = cudaMemset2D(dev_ptr, pitch, 0, width * sizeof(float), height);
    if (err != cudaSuccess) {
        checkCudaError(err, "cudaMemset2D");
        cudaFree(dev_ptr);
        dev_ptr = nullptr;
        return;
    }

    // Describe the pitched 2D resource for texture creation.
    cudaResourceDesc resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dev_ptr;
    resDesc.res.pitch2D.desc = _cudaChannelFormatDesc;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = pitch;

    // Texture description
    cudaTextureDesc texDesc;
    std::memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = _cudaTextureAddressMode;
    texDesc.addressMode[1] = _cudaTextureAddressMode;
    texDesc.filterMode = _cudaTextureFilterMode;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1; // use unnormalized coords (x in [0,width), y in [0,height))

    // Create texture object
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
