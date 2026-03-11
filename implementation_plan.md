# Fix `CreateHeap failed for shared texture` Error

## Root Cause

`CudaD3D12Texture::Initialize()` uses a **placed resource on a shared heap** with `D3D12_TEXTURE_LAYOUT_ROW_MAJOR`. This approach has two problems:

1. **`ROW_MAJOR` layout for 2D textures** requires `CrossAdapterRowMajorTextureSupported` — a hardware feature most single-GPU laptops (including RTX 3050 Ti) do **not** support.
2. **`D3D12_HEAP_FLAG_SHARED` on custom heaps** with 2D textures is restricted; NVIDIA's CUDA interop samples use committed resources instead.

The standard CUDA↔D3D12 interop pattern (per NVIDIA's `simpleD3D12` sample) is:
- Create a **committed resource** with `D3D12_HEAP_FLAG_SHARED`
- Share the **resource** handle (not a heap handle)
- Import using `cudaExternalMemoryHandleTypeD3D12Resource` (not `D3D12Heap`)

## Proposed Changes

### D3D12↔CUDA Interop

#### [MODIFY] [CudaD3D12Texture.cu](file:///c:/Users/RC1ki/Documents/rvgrt/RVGRT/src/CudaD3D12Texture.cu)

Rewrite `Initialize()` (lines 215–306) to use committed resource interop:

1. **Change layout** from `D3D12_TEXTURE_LAYOUT_ROW_MAJOR` → `D3D12_TEXTURE_LAYOUT_UNKNOWN` (allows GPU-optimized swizzled layout)

2. **Replace heap-based allocation** with `CreateCommittedResource`:
```cpp
D3D12_HEAP_PROPERTIES heapProps = {};
heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

checkHresult(device->CreateCommittedResource(
    &heapProps,
    D3D12_HEAP_FLAG_SHARED,          // Key: shared committed resource
    &desc,
    D3D12_RESOURCE_STATE_COMMON,
    nullptr,
    IID_PPV_ARGS(&m_d3dResource)));
```

3. **Share the resource** (not heap):
```cpp
checkHresult(device->CreateSharedHandle(
    m_d3dResource.Get(),             // Share the RESOURCE, not a heap
    nullptr, GENERIC_ALL, nullptr, &m_sharedHandle));
```

4. **Import as D3D12Resource** (not D3D12Heap):
```cpp
extMemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;  // Changed
```

5. **Remove** the heap-based buffer mapping (`cudaExternalMemoryGetMappedBuffer`) — not applicable for texture resources. Keep only the mipmapped array path for CUDA surface/texture access.

6. **Remove** the `GetCopyableFootprints` pitch calculation — not needed with swizzled layout (CUDA uses array-based access, not linear).

#### [MODIFY] [CudaD3D12Texture.cuh](file:///c:/Users/RC1ki/Documents/rvgrt/RVGRT/include/CudaD3D12Texture.cuh)

- Remove `m_d3dHeap` member (no longer used)
- Remove `m_cudaDevPtr` and `m_pitch` members (linear access not used)
- Update `Release()` — remove heap cleanup

> [!IMPORTANT]
> The `win32_main.cpp` render loop calls `interopTexture.getCudaSurfObject()` to pass to `PostDraw()`. The surface object is currently created in `Initialize_Cuda_Array()` only. The rewritten `Initialize()` must also create surface/texture objects from the imported mipmapped array so `getCudaSurfObject()` works.

## Verification Plan

### Manual Verification
1. Build with `cmake --build . --config Release`
2. Run `RVGRT.exe` — should get past startup without the `CreateHeap` error
3. Verify the window opens and begins rendering
