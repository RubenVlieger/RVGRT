#include "CudaD3D12Texture.cuh"
#include <stdexcept>
#include <iostream>
#include <aclapi.h> // For security attributes
#include <cumath.cuh>

// (You can reuse your existing WindowsSecurityAttributes class here)
class WindowsSecurityAttributes {
protected:
    SECURITY_ATTRIBUTES m_winSecurityAttributes;
    PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;
public:
    WindowsSecurityAttributes() {
        m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**));
        if (!m_winPSecurityDescriptor) throw std::runtime_error("Failed to allocate security descriptor.");
        
        PSID* ppSID = (PSID*)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
        PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));
        InitializeSecurityDescriptor(m_winPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);
        SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
        AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, ppSID);
        EXPLICIT_ACCESS explicitAccess;
        ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
        explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
        explicitAccess.grfAccessMode = SET_ACCESS;
        explicitAccess.grfInheritance = INHERIT_ONLY;
        explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
        explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
        explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;
        SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);
        SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);
        m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
        m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
        m_winSecurityAttributes.bInheritHandle = TRUE;
    }
    ~WindowsSecurityAttributes() {
        PSID* ppSID = (PSID*)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
        PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));
        if (*ppSID) FreeSid(*ppSID);
        if (*ppACL) LocalFree(*ppACL);
        free(m_winPSecurityDescriptor);
    }
    SECURITY_ATTRIBUTES* operator&() { return &m_winSecurityAttributes; }
};

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
    
    // These are derived from m_cudaExtMem, so just null them out
    m_cudaDevPtr = nullptr;
    m_cudaMipmappedArray = nullptr;
    m_cudaArray = nullptr;
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
      m_cudaExtMem(other.m_cudaExtMem),
      m_sharedHandle(other.m_sharedHandle),
      m_cudaDevPtr(other.m_cudaDevPtr),
      m_cudaMipmappedArray(other.m_cudaMipmappedArray),
      m_cudaArray(other.m_cudaArray),
      m_pitch(other.m_pitch) {
    // Leave the moved-from object in a safe state
    other.m_cudaExtMem = nullptr;
    other.m_sharedHandle = nullptr;
}

CudaD3D12Texture& CudaD3D12Texture::operator=(CudaD3D12Texture&& other) noexcept {
    if (this != &other) {
        Release();
        m_d3dResource = std::move(other.m_d3dResource);
        m_d3dHeap = std::move(other.m_d3dHeap);
        m_cudaExtMem = other.m_cudaExtMem;
        m_sharedHandle = other.m_sharedHandle;
        m_cudaDevPtr = other.m_cudaDevPtr;
        m_cudaMipmappedArray = other.m_cudaMipmappedArray;
        m_cudaArray = other.m_cudaArray;
        m_pitch = other.m_pitch;
        width = other.width;
        height = other.height;

        other.m_cudaExtMem = nullptr;
        other.m_sharedHandle = nullptr;
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
    if (IsValid()) {
        Release();
    }
    width = _width;
    height = _height;

    // --- Steps 1-5 remain exactly the same ---
    
    // 1. Create D3D12 Resource Description
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = (unsigned int)width;
    desc.Height = (unsigned int)height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;

    D3D12_RESOURCE_ALLOCATION_INFO allocInfo = device->GetResourceAllocationInfo(0, 1, &desc);
    if (allocInfo.SizeInBytes == 0) {
        throw std::runtime_error("Resource allocation size is zero.");
    }

    // 2. Create a Shared Heap
    D3D12_HEAP_DESC heapDesc = {};
    heapDesc.SizeInBytes = allocInfo.SizeInBytes;
    heapDesc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapDesc.Alignment = allocInfo.Alignment;
    heapDesc.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;
    checkHresult(device->CreateHeap(&heapDesc, IID_PPV_ARGS(&m_d3dHeap)), "CreateHeap failed for shared texture.");
    // 3. Create a Placed Resource on the Heap
    checkHresult(device->CreatePlacedResource(
        m_d3dHeap.Get(), 0, &desc, D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&m_d3dResource)), "CreatePlacedResource failed for shared texture.");
    m_d3dResource->SetName(debugName);
    
    // 4. Create a Shared Handle for the Heap
    checkHresult(device->CreateSharedHandle(m_d3dHeap.Get(), nullptr, GENERIC_ALL, nullptr, &m_sharedHandle), "CreateSharedHandle failed for D3D12 heap.");
    
    // 5. Import Heap into CUDA as External Memory
    cudaExternalMemoryHandleDesc extMemHandleDesc = {};
    extMemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    extMemHandleDesc.handle.win32.handle = m_sharedHandle;
    extMemHandleDesc.size = allocInfo.SizeInBytes;
    extMemHandleDesc.flags = cudaExternalMemoryDedicated;
    checkCudaError(cudaImportExternalMemory(&m_cudaExtMem, &extMemHandleDesc), "cudaImportExternalMemory from heap.");

    // --- 6. Map to Buffers (MODIFIED LOGIC) ---
    
    // Always map to a linear buffer to get a raw device pointer for kernels
    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = allocInfo.SizeInBytes;
    checkCudaError(cudaExternalMemoryGetMappedBuffer(&m_cudaDevPtr, m_cudaExtMem, &bufferDesc), "cudaExternalMemoryGetMappedBuffer");

    // MODIFICATION: Only map to a mipmapped array if the format is supported by CUDA textures.
    // DXGI_FORMAT_R16G16_FLOAT is not supported.
    if (format != DXGI_FORMAT_R16G16_FLOAT && 
        format != DXGI_FORMAT_R16_FLOAT) 
    {
        cudaExternalMemoryMipmappedArrayDesc mipmappedDesc = {};
        mipmappedDesc.extent = make_cudaExtent(width, height, 0);
        mipmappedDesc.formatDesc = GetCudaChannelDesc(format);
        mipmappedDesc.numLevels = 1;
        mipmappedDesc.flags = cudaArraySurfaceLoadStore;

        cudaError_t mapArrayResult = cudaExternalMemoryGetMappedMipmappedArray(&m_cudaMipmappedArray, m_cudaExtMem, &mipmappedDesc);
        if (mapArrayResult == cudaSuccess) {
            checkCudaError(cudaGetMipmappedArrayLevel(&m_cudaArray, m_cudaMipmappedArray, 0), "cudaGetMipmappedArrayLevel");
        } else {
            // Log a warning if it fails for an unexpected format
            std::cerr << "Warning: cudaExternalMemoryGetMappedMipmappedArray failed for format " << format
                      << " with error: " << cudaGetErrorString(mapArrayResult) << ". Fallback to linear memory access." << std::endl;
        }
    } else {
        std::cout << "Info: Skipping cudaMipmappedArray for DXGI_FORMAT_R16G16_FLOAT. Use linear device pointer (m_cudaDevPtr) instead." << std::endl;
    }
    
    // --- 7. Get the correct row pitch from D3D12 (Crucial for linear access) ---
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placedFootprint;
    UINT numRows;
    UINT64 rowSizeInBytes;
    UINT64 totalBytes;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &placedFootprint, &numRows, &rowSizeInBytes, &totalBytes);
    m_pitch = placedFootprint.Footprint.RowPitch;

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
}