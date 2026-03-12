# Fix CudaD3D12Texture CUDA-D3D12 Interop

The current `Initialize()` uses `CreateCommittedResource` with `D3D12_TEXTURE_LAYOUT_ROW_MAJOR` + `ALLOW_UNORDERED_ACCESS`, which D3D12 rejects. The fix is to switch to the **shared heap + placed resource** approach from the old working code.

## Problem

D3D12 rule: row-major textures **cannot** have `ALLOW_RENDER_TARGET` or `ALLOW_UNORDERED_ACCESS` flags. But CUDA interop via `cudaExternalMemoryHandleTypeD3D12Heap` requires a shared heap with `ROW_MAJOR` layout. The old working code solves this by:

1. Creating a `D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER` heap
2. Creating a **placed resource** on that heap with `ROW_MAJOR` layout and **no UAV flags**
3. Sharing the **heap** (not resource) with CUDA
4. Mapping it as both a linear buffer (`cudaExternalMemoryGetMappedBuffer`) and optionally a mipmapped array

The CUDA kernels then write via `surf2Dwrite` to the mipmapped array view, and D3D12 copies the placed resource to the backbuffer.

> [!IMPORTANT]
> The placed resource **must not** have UAV flags since `ROW_MAJOR` layout forbids them. The old code correctly omits UAV from `desc.Flags` for the shared interop path.

## Proposed Changes

### CudaD3D12Texture

#### [MODIFY] [CudaD3D12Texture.cuh](file:///c:/Users/RC1ki/Documents/rvgrt/RVGRT/include/CudaD3D12Texture.cuh)

- Add `m_d3dHeap` (`ComPtr<ID3D12Heap>`) member
- Add `m_cudaDevPtr` (`void*`) and `m_pitch` (`size_t`) members
- Update `GetCudaDevicePtr()` and `getPitch()` to return actual values
- Update `IsValid()` — for CUDA-only arrays, `m_cudaExtMem` may be null

#### [MODIFY] [CudaD3D12Texture.cu](file:///c:/Users/RC1ki/Documents/rvgrt/RVGRT/src/CudaD3D12Texture.cu)

**`Initialize()` rewrite** — switch to the shared heap + placed resource approach:
1. Create `D3D12_RESOURCE_DESC` with `ROW_MAJOR` layout and **no UAV flags**
2. Call `GetResourceAllocationInfo` to get size/alignment
3. Create shared heap with `D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER`
4. Create placed resource via `CreatePlacedResource`
5. Share the **heap** handle with CUDA via `cudaExternalMemoryHandleTypeD3D12Heap`
6. Map as both linear buffer (`cudaExternalMemoryGetMappedBuffer`) and mipmapped array (where format is supported)
7. Create surface/texture objects from the mipmapped array
8. Get row pitch via `GetCopyableFootprints`

**`Release()` update** — free `m_d3dHeap`, `m_cudaDevPtr`

**Move constructor/assignment** — transfer `m_d3dHeap`, `m_cudaDevPtr`, `m_pitch`

Remove the `WindowsSecurityAttributes` class (per user request, not needed).

---

### Render Loop (no changes expected)

#### [NO CHANGE] [win32_main.cpp](file:///c:/Users/RC1ki/Documents/rvgrt/RVGRT/src/platform/win32_main.cpp)

The render loop already calls `Initialize()` with `DXGI_FORMAT_R16G16B16A16_FLOAT` and `ALLOW_UNORDERED_ACCESS`. The UAV flag will now be **ignored** inside `Initialize()` since shared interop resources can't use it — but the API signature stays the same. No caller changes needed.

> [!NOTE]
> The `flags` parameter in `Initialize()` will be silently dropped for the shared interop path. This matches the old working behavior. The D3D12 copy path doesn't need UAV since it only uses `CopyResource`.

## Verification Plan

### Manual Verification
1. Build the project (presumably via CMake + your existing build setup)
2. Run the application and check the console output:
   - Should see `[Render] Interop texture initialized` (our earlier diagnostic)
   - Should see `[Render] About to Draw` and `[Render] Draw complete`
   - No abort/crash dialog
3. Confirm the window appears and doesn't immediately crash

> [!TIP]
> If it crashes *after* `Initialize` succeeds, the issue is downstream in the CUDA kernels or the D3D12 copy path — but that would be a separate problem from this interop fix.
