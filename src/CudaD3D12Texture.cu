#include "CudaD3D12Texture.cuh"
#include <stdexcept>
#include <iostream>
#include <cuda_fp16.h>

static inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(err) << "\n";
        throw std::runtime_error(msg);
    }
}

static void checkHresult(HRESULT hr, const char* msg) {
    if (FAILED(hr)) {
        std::cerr << "D3D12/DXGI Error (" << msg << ")" << std::endl;
        throw std::runtime_error(msg);
    }
}


CudaD3D12Texture::CudaD3D12Texture() = default;

CudaD3D12Texture::~CudaD3D12Texture() {
    Release();
}

void CudaD3D12Texture::Release() {
    if (m_cudaSurfaceObj) {
        cudaDestroySurfaceObject(m_cudaSurfaceObj);
        m_cudaSurfaceObj = 0;
    }
    if (m_cudaTextureObj) {
        cudaDestroyTextureObject(m_cudaTextureObj);
        m_cudaTextureObj = 0;
    }
    if (m_cudaMipmappedArray) {
        cudaFreeMipmappedArray(m_cudaMipmappedArray);
        m_cudaMipmappedArray = nullptr;
    }
    m_cudaArray = nullptr;
    
    if (m_cudaDevPtr) {
        // Device pointer is from external memory mapping, don't free directly
        m_cudaDevPtr = nullptr;
    }
    
    if (m_cudaExtMem) {
        cudaDestroyExternalMemory(m_cudaExtMem);
        m_cudaExtMem = nullptr;
    }
    if (m_sharedHandle) {
        CloseHandle(m_sharedHandle);
        m_sharedHandle = nullptr;
    }
    m_d3dResource.Reset();
    m_d3dHeap.Reset();
    m_pitch = 0;
}

static cudaChannelFormatDesc GetCudaChannelDesc(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            return cudaCreateChannelDesc<uchar4>();
        case DXGI_FORMAT_R16G16_FLOAT:
            return cudaCreateChannelDesc<half2>();
        case DXGI_FORMAT_R16_FLOAT:
            return cudaCreateChannelDesc<half>();
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            return cudaCreateChannelDescHalf4();
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            return cudaCreateChannelDesc<float4>();
        case DXGI_FORMAT_R32G32_FLOAT:
            return cudaCreateChannelDesc<float2>();
        case DXGI_FORMAT_R32_FLOAT:
            return cudaCreateChannelDesc<float>();
        // Add other formats as needed
        default:
            throw std::runtime_error("Unsupported DXGI_FORMAT for CUDA texture mapping.");
    }
}

// Move constructor and assignment operator (for placing in containers if needed)
CudaD3D12Texture::CudaD3D12Texture(CudaD3D12Texture&& other) noexcept
    : m_d3dResource(std::move(other.m_d3dResource)),
      m_d3dHeap(std::move(other.m_d3dHeap)),
      m_cudaSurfaceObj(other.m_cudaSurfaceObj),
      m_cudaTextureObj(other.m_cudaTextureObj),
      m_cudaExtMem(other.m_cudaExtMem),
      m_sharedHandle(other.m_sharedHandle),
      m_cudaDevPtr(other.m_cudaDevPtr),
      m_pitch(other.m_pitch),
      m_cudaMipmappedArray(other.m_cudaMipmappedArray),
      m_cudaArray(other.m_cudaArray),
      width(other.width),
      height(other.height) {
    // Leave the moved-from object in a safe state
    other.m_cudaSurfaceObj = 0;
    other.m_cudaTextureObj = 0;
    other.m_cudaExtMem = nullptr;
    other.m_sharedHandle = nullptr;
    other.m_cudaDevPtr = nullptr;
    other.m_pitch = 0;
    other.m_cudaMipmappedArray = nullptr;
    other.m_cudaArray = nullptr;
}

CudaD3D12Texture& CudaD3D12Texture::operator=(CudaD3D12Texture&& other) noexcept {
    if (this != &other) {
        Release();
        m_d3dResource = std::move(other.m_d3dResource);
        m_d3dHeap = std::move(other.m_d3dHeap);
        m_cudaSurfaceObj = other.m_cudaSurfaceObj;
        m_cudaTextureObj = other.m_cudaTextureObj;
        m_cudaExtMem = other.m_cudaExtMem;
        m_sharedHandle = other.m_sharedHandle;
        m_cudaDevPtr = other.m_cudaDevPtr;
        m_pitch = other.m_pitch;
        m_cudaMipmappedArray = other.m_cudaMipmappedArray;
        m_cudaArray = other.m_cudaArray;
        width = other.width;
        height = other.height;

        other.m_cudaSurfaceObj = 0;
        other.m_cudaTextureObj = 0;
        other.m_cudaExtMem = nullptr;
        other.m_sharedHandle = nullptr;
        other.m_cudaDevPtr = nullptr;
        other.m_pitch = 0;
        other.m_cudaMipmappedArray = nullptr;
        other.m_cudaArray = nullptr;
    }
    return *this;
}

void CudaD3D12Texture::Initialize_Cuda_Array(UINT _width, UINT _height, const cudaChannelFormatDesc& formatDesc, cudaTextureFilterMode filterMode)
{
    if (IsValid()) {
        Release();
    }
    width = _width;
    height = _height;

    // 1. Allocate the cudaArray with the specified format. This creates the swizzled storage.
    checkCudaError(cudaMallocArray(&m_cudaArray, &formatDesc, width, height, cudaArraySurfaceLoadStore), "cudaMallocArray failed.");

    // 2. Create the Surface Object (for writing to the array)
    cudaResourceDesc resDescSurf = {};
    resDescSurf.resType = cudaResourceTypeArray;
    resDescSurf.res.array.array = m_cudaArray;
    checkCudaError(cudaCreateSurfaceObject(&m_cudaSurfaceObj, &resDescSurf), "cudaCreateSurfaceObject failed.");

    // 3. Create the Texture Object (for reading from the array with filtering)
    cudaResourceDesc resDescTex = {};
    resDescTex.resType = cudaResourceTypeArray;
    resDescTex.res.array.array = m_cudaArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp; // Clamp to edge
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = filterMode; // Bilinear filtering
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1; // Use normalized UV coordinates [0, 1]
    
    checkCudaError(cudaCreateTextureObject(&m_cudaTextureObj, &resDescTex, &texDesc, nullptr), "cudaCreateTextureObject failed.");
}
void CudaD3D12Texture::Initialize_D3D12_Only(ID3D12Device* device, UINT _width, UINT _height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, const wchar_t* debugName)
{
    if (IsValid()) {
        Release();
    }
    width = _width;
    height = _height;

    // 1. Define the properties for a standard GPU-private heap (not shared).
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    // 2. Create the D3D12 Resource Description.
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = (unsigned int)width;
    desc.Height = (unsigned int)height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    // CRITICAL PERFORMANCE NOTE: Use UNKNOWN layout for GPU-only textures.
    // This allows the driver to use optimized swizzled layouts and compression.
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; 
    desc.Flags = flags; // This is where ALLOW_UNORDERED_ACCESS will be passed in.

    // 3. Create a Committed Resource.
    // This is the standard way to create a standalone resource in D3D12.
    // It creates the resource and allocates its memory in one call.
    checkHresult(device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON, // Start in the common state
        nullptr, // No optimized clear value needed
        IID_PPV_ARGS(&m_d3dResource)), "CreateCommittedResource failed for D3D12-only texture.");

    m_d3dResource->SetName(debugName);

    // NOTE: We intentionally DO NOT initialize any CUDA members (m_d3dHeap, m_sharedHandle, m_cudaExtMem, etc.)
    // because this resource will never be touched by CUDA.
}

void CudaD3D12Texture::Initialize(ID3D12Device* device, UINT _width, UINT _height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, const wchar_t* debugName) {
    (void)flags; // Flags are ignored for shared interop - ROW_MAJOR forbids UAV
    
    if (IsValid()) {
        Release();
    }
    width = _width;
    height = _height;

    // 1. Create D3D12 Resource Description with ROW_MAJOR layout (required for CUDA interop)
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = (unsigned int)width;
    desc.Height = (unsigned int)height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;  // Required for CUDA interop via heap
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;  // Required when using SHARED_CROSS_ADAPTER heap

    // 2. Get allocation info for the resource
    D3D12_RESOURCE_ALLOCATION_INFO allocInfo = device->GetResourceAllocationInfo(0, 1, &desc);

    // 3. Create shared heap
    D3D12_HEAP_DESC heapDesc = {};
    heapDesc.SizeInBytes = allocInfo.SizeInBytes;
    heapDesc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapDesc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapDesc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapDesc.Alignment = allocInfo.Alignment;
    heapDesc.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;

    checkHresult(device->CreateHeap(&heapDesc, IID_PPV_ARGS(&m_d3dHeap)),
        "CreateHeap failed for shared CUDA-D3D12 interop.");

    // 4. Create placed resource on the heap
    checkHresult(device->CreatePlacedResource(
        m_d3dHeap.Get(),
        0,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&m_d3dResource)),
        "CreatePlacedResource failed for CUDA-D3D12 interop.");
    m_d3dResource->SetName(debugName);

    // 5. Create shared handle for the heap (not the resource)
    checkHresult(device->CreateSharedHandle(
        m_d3dHeap.Get(),
        nullptr, GENERIC_ALL, nullptr, &m_sharedHandle),
        "CreateSharedHandle for heap failed.");

    // 6. Import heap into CUDA
    cudaExternalMemoryHandleDesc extMemHandleDesc = {};
    extMemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    extMemHandleDesc.handle.win32.handle = m_sharedHandle;
    extMemHandleDesc.size = allocInfo.SizeInBytes;
    extMemHandleDesc.flags = cudaExternalMemoryDedicated;
    checkCudaError(cudaImportExternalMemory(&m_cudaExtMem, &extMemHandleDesc),
        "cudaImportExternalMemory from heap failed.");

    // 7. Map as linear buffer
    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = allocInfo.SizeInBytes;
    bufferDesc.flags = 0;
    checkCudaError(cudaExternalMemoryGetMappedBuffer(&m_cudaDevPtr, m_cudaExtMem, &bufferDesc),
        "cudaExternalMemoryGetMappedBuffer failed.");

    // 8. Get row pitch using GetCopyableFootprints
    UINT64 totalSize = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &layout, nullptr, nullptr, &totalSize);
    m_pitch = layout.Footprint.RowPitch;

    // 9. Map as mipmapped array (if format is supported)
    if (format != DXGI_FORMAT_R16G16_FLOAT && format != DXGI_FORMAT_R16_FLOAT) {
        cudaExternalMemoryMipmappedArrayDesc mipmappedDesc = {};
        mipmappedDesc.extent = make_cudaExtent(width, height, 0);
        mipmappedDesc.formatDesc = GetCudaChannelDesc(format);
        mipmappedDesc.numLevels = 1;
        mipmappedDesc.flags = cudaArraySurfaceLoadStore;

        checkCudaError(cudaExternalMemoryGetMappedMipmappedArray(&m_cudaMipmappedArray, m_cudaExtMem, &mipmappedDesc),
            "cudaExternalMemoryGetMappedMipmappedArray failed.");
        checkCudaError(cudaGetMipmappedArrayLevel(&m_cudaArray, m_cudaMipmappedArray, 0),
            "cudaGetMipmappedArrayLevel failed.");

        // 10. Create Surface Object (for writing)
        cudaResourceDesc resDescSurf = {};
        resDescSurf.resType = cudaResourceTypeArray;
        resDescSurf.res.array.array = m_cudaArray;
        checkCudaError(cudaCreateSurfaceObject(&m_cudaSurfaceObj, &resDescSurf),
            "cudaCreateSurfaceObject failed.");

        // 11. Create Texture Object (for reading with filtering)
        cudaResourceDesc resDescTex = {};
        resDescTex.resType = cudaResourceTypeArray;
        resDescTex.res.array.array = m_cudaArray;
        
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;
        
        checkCudaError(cudaCreateTextureObject(&m_cudaTextureObj, &resDescTex, &texDesc, nullptr),
            "cudaCreateTextureObject failed.");
    } else {
        throw std::runtime_error("Format not supported for CUDA interop. Use DXGI_FORMAT_R8G8B8A8_UNORM, R32G32B32A32_FLOAT, or R32_FLOAT.");
    }

    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    checkCudaError(cudaGetLastError(), "cudaGetLastError");
}
