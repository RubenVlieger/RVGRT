#ifdef D3D12
#include "Framebuffer.cuh"
#define NOMINMAX
#include <windows.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <aclapi.h> // Required for security functions

class WindowsSecurityAttributes {
protected:
    SECURITY_ATTRIBUTES m_winSecurityAttributes;
    PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
    WindowsSecurityAttributes() {
        m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**));
        if (!m_winPSecurityDescriptor) {
            // Handle allocation error
            return;
        }

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


Framebuffer::Framebuffer() = default;

Framebuffer::~Framebuffer() 
{
    // Ensure all resources are freed on destruction
    CleanupInterop();
}

void Framebuffer::InitializeInterop(ID3D12Device* device, int _w, int _h) 
{
    width = _w;
    height = _h;

    // --- Step 1: Create the D3D12 ROW_MAJOR Resource ---
    // This part is correct and remains the same.
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; 

    HRESULT hr = device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&d_texture));
    if (FAILED(hr)) { std::cerr << "CreateCommittedResource failed\n"; return; }

    // --- Step 2: Get the Pitch/Layout of the D3D12 resource ---
    device->GetCopyableFootprints(&desc, 0, 1, 0, nullptr, nullptr, nullptr, &rowPitchInBytes);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placedFootprint;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &placedFootprint, nullptr, nullptr, nullptr);
    rowPitchInBytes = placedFootprint.Footprint.RowPitch;

    // --- Step 3: Create and Import the Shared Handle ---
    // This is also correct and remains the same.
    WindowsSecurityAttributes securityAttributes;
    hr = device->CreateSharedHandle(d_texture.Get(), &securityAttributes, GENERIC_ALL, nullptr, &sharedHandle);
    if (FAILED(hr) || sharedHandle == nullptr) { std::cerr << "CreateSharedHandle failed\n"; return; }
    
    cudaExternalMemoryHandleDesc extMemHandleDesc = {};
    memset(&extMemHandleDesc, 0, sizeof(extMemHandleDesc));
    extMemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    extMemHandleDesc.handle.win32.handle = sharedHandle;
    D3D12_RESOURCE_ALLOCATION_INFO allocInfo = device->GetResourceAllocationInfo(0, 1, &desc);
    extMemHandleDesc.size = allocInfo.SizeInBytes;
    extMemHandleDesc.flags = cudaExternalMemoryDedicated; 
    
    cudaError_t err = cudaImportExternalMemory(&cudaExtMem, &extMemHandleDesc);
    if (err != cudaSuccess) { std::cerr << "cudaImportExternalMemory failed: " << cudaGetErrorString(err) << std::endl; return; }

    // --- Step 4 (THE NEW FIX): Map the external memory to a linear device pointer ---
    cudaExternalMemoryBufferDesc bufferDesc = {};
    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.size = allocInfo.SizeInBytes;
    
    err = cudaExternalMemoryGetMappedBuffer((void**)&d_pixels, cudaExtMem, &bufferDesc);
    if (err != cudaSuccess) {
        std::cerr << "cudaExternalMemoryGetMappedBuffer failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

void Framebuffer::CleanupInterop() 
{
    // d_pixels is a mapped pointer, not a malloc'd one, so we don't cudaFree it.
    // Destroying the external memory object is what unmaps it.
    d_pixels = nullptr;

    if (cudaExtMem) {
        cudaDestroyExternalMemory(cudaExtMem);
        cudaExtMem = 0;
    }
    if (sharedHandle) {
        CloseHandle(sharedHandle);
        sharedHandle = nullptr;
    }
    d_texture.Reset();
}


#else
#include "Framebuffer.cuh"
#define NOMINMAX
#include <windowsx.h>
#include "d3d11.h"

#include <cuda_runtime.h>
#include <iostream>
#include <cuda_d3d11_interop.h>


Framebuffer::Framebuffer()
    : width(0), height(0), d_pixels(nullptr) {
    
}   

Framebuffer::~Framebuffer() 
{

}

void Framebuffer::InitializeInterop(ID3D11Device* device, ID3D11DeviceContext* context) 
{
    d_context = context;
    // Create the D3D11 texture
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width; desc.Height = height;
    desc.MipLevels = 1; desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.CPUAccessFlags = 0; desc.MiscFlags = 0;
    HRESULT hr = device->CreateTexture2D(&desc, nullptr, &d_texture);
    if (FAILED(hr)) {
        std::cerr << "CreateTexture2D failed\n";
    }
    // Register with CUDA (allows mapping)
    cudaGraphicsD3D11RegisterResource(&cudaResource, 
                                      (ID3D11Resource*)d_texture, 
                                      cudaGraphicsRegisterFlagsNone);
}

void Framebuffer::CleanupInterop() 
{
    if (cudaResource) {
        cudaGraphicsUnregisterResource(cudaResource);
        cudaResource = nullptr;
    }
    if (d_texture) {
        d_texture->Release();
        d_texture = nullptr;
    }
}

void Framebuffer::CopyDeviceToTexture() 
{
    if (!cudaResource) return;
    // Map the D3D resource for CUDA access
    cudaGraphicsMapResources(1, &cudaResource, 0);
    // Get CUDA array for the texture
    cudaArray* dstArray = nullptr;
    cudaGraphicsSubResourceGetMappedArray(&dstArray, cudaResource, 0, 0);
    // Copy device memory into the array (2D copy, no host)
    size_t pitch = width * sizeof(uint32_t);
    cudaMemcpy2DToArray(dstArray,
                        0, 0,
                        d_pixels, pitch,
                        pitch, height,
                        cudaMemcpyDeviceToDevice);
    // Unmap so D3D can use it
    cudaGraphicsUnmapResources(1, &cudaResource, 0);
}



void Framebuffer::Allocate(int _w, int _h) {
    if (d_pixels) 
    {
        Free(); // free old buffer
    }
    width = _w;
    height = _h;

    size_t size = width * height * sizeof(uint32_t);
    cudaError_t err = cudaMalloc(&d_pixels, size);
    if (err != cudaSuccess) 
    {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        d_pixels = nullptr;
    } else 
    {
        // Optionally clear framebuffer
        cudaMemset(d_pixels, 0, size);
    }
}

void Framebuffer::Free() 
{
    if (d_pixels) {
        cudaFree(d_pixels);
        d_pixels = nullptr;
    }
}

std::vector<uint32_t> Framebuffer::readback() const 
{
    std::vector<uint32_t> cpuBuffer(width * height);
    if (d_pixels) {
        cudaMemcpy(cpuBuffer.data(), d_pixels, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    return cpuBuffer;
}

void Framebuffer::readback(uint32_t* buffer) const 
{
    if (d_pixels) {
        cudaMemcpy(buffer, d_pixels, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
}
#endif