#pragma once

// Use __OBJC__ to guard Objective-C specific forward declarations
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandQueue;
#endif

// The C++ compiler will only see the void* declarations for D3D12
#ifdef _WIN32
struct ID3D12Device;
struct ID3D12CommandQueue;
#endif

class GraphicsDevice
{
public:
    virtual ~GraphicsDevice() = default;

    virtual void Initialize(void* windowHandle) = 0;
    virtual void BeginFrame() = 0;
    virtual void EndFrame() = 0;

#if defined(_WIN32)
    virtual ID3D12Device* GetD3D12Device() = 0;
    virtual ID3D12CommandQueue* GetCommandQueue() = 0;
#elif defined(__APPLE__)
    // For Objective-C++ files, the compiler sees the real Metal types.
    #ifdef __OBJC__
    virtual id<MTLDevice> GetMetalDevice() = 0;
    virtual id<MTLCommandQueue> GetMetalCommandQueue() = 0;
    #else
    // For pure C++ files, we provide an opaque void* pointer as a placeholder.
    // This allows C++ code to include the header without knowing about Metal.
    virtual void* GetMetalDevice() = 0;
    virtual void* GetMetalCommandQueue() = 0;
    #endif
#endif
};