# Slang → CUDA Build Path Implementation

This document describes the complete Slang-to-CUDA build pipeline for RVGRT on Windows.

## Overview

The build system uses **Slang** (`slangc`) to transpile `.slang` shader files to CUDA C++, then post-processes them with a Python script to generate host-side wrapper functions.

## Build Flow

```
.slang files → slangc -target cuda → .cu files → gen_cuda_wrappers.py → post-processed .cu + .cuh header → nvcc → .obj → link
```

## Step-by-Step Process

### 1. Slang Transpilation (CMakeLists.txt)

Each kernel is transpiled individually (per-kernel compilation avoids duplicate symbol conflicts):

```cmake
slangc -target cuda \
    -entry distApproximationKernel \
    -stage compute \
    -I include \
    -I src/renderer/kernels/slang \
    -D__SLANG__ \
    src/renderer/kernels/slang/dist_approx.slang \
    -o slang_generated/distApproximationKernel.cu
```

**Why per-kernel?** Each Slang kernel generates its own `__constant__ SLANG_globalParams` variable. If multiple kernels are compiled together, these duplicate symbols cause linker errors.

### 2. Python Post-Processing (gen_cuda_wrappers.py)

The Python script (`gen_cuda_wrappers.py`) processes each generated `.cu` file:

1. **Renames the GlobalParams struct** to a unique name per kernel:
   ```cpp
   struct GlobalParams_distApproximationKernel { ... };
   ```

2. **Renames the constant variable** to avoid conflicts:
   ```cpp
   __constant__ GlobalParams_distApproximationKernel SLANG_gp_distApproximationKernel;
   ```

3. **Generates a Launch_* wrapper function** that:
   - Takes host-friendly parameters (cudaSurfaceObject_t, cudaTextureObject_t, etc.)
   - Fills the GlobalParams struct
   - Uploads to __constant__ memory via `cudaMemcpyToSymbolAsync`
   - Launches the kernel

Example generated wrapper:
```cpp
extern "C" void Launch_distApproximationKernel(
    cudaStream_t stream, dim3 grid, dim3 block,
    cudaSurfaceObject_t distTex, const CameraData& camera, 
    const FrameData& frame, cudaTextureObject_t indirection,
    void* sectorBuffer, void* occupancyBuffer, void* dataBuffer,
    void* sectorMaskBuffer, void* charData
) {
    GlobalParams_distApproximationKernel gp = {};
    gp.distTex_0 = distTex;
    memcpy(&gp.camera_0, &camera, sizeof(gp.camera_0));
    // ... fill other fields ...
    cudaMemcpyToSymbolAsync(SLANG_gp_distApproximationKernel, &gp, sizeof(gp), 
                            0, cudaMemcpyHostToDevice, stream);
    distApproximationKernel<<<grid, block, 0, stream>>>();
}
```

4. **Generates a header file** (`CudaShaderLaunchers.generated.cuh`) with all Launch_* declarations.

### 3. CUDA Compilation

The post-processed `.cu` files are compiled by nvcc as regular CUDA source files:

```cmake
set_source_files_properties(${CUDA_WRAPPER_SOURCES} PROPERTIES LANGUAGE CUDA)
```

### 4. Host Code Usage (CudaRenderer.cu)

`CudaRenderer.cu` includes the generated header and calls the Launch_* functions:

```cpp
#include "CudaShaderLaunchers.generated.cuh"

// In Draw():
Launch_distApproximationKernel(
    _cudaStream, gridSizeHalf, groupSize8,
    _halfDistTexture.surface,
    camData,
    frameData,
    _materialMap.GetIndirectionTexture(),
    _materialMap.GetSectorBufferPtr(),
    _materialMap.GetOccupancyPtr(),
    _materialMap.GetDataPtr(),
    _materialMap.GetSectorMaskPtr(),
    _characterBuffer
);
```

## Resource Mapping

| Slang Type | CUDA Type | Host Parameter Type |
|------------|-----------|---------------------|
| `RWTexture2D<float4>` | `CUsurfObject` | `cudaSurfaceObject_t` |
| `Texture2D<float4>` | `CUtexObject` | `cudaTextureObject_t` |
| `Texture3D<uint>` | `CUtexObject` | `cudaTextureObject_t` |
| `StructuredBuffer<T>` | `StructuredBuffer<T>` | `void*` (data pointer) |
| `RWStructuredBuffer<T>` | `RWStructuredBuffer<T>` | `void*` (data pointer) |
| `ConstantBuffer<T>` | Inline in GlobalParams | `const T&` (copied) |
| `uniform CameraData` | Inline in GlobalParams | `const CameraData&` |
| `uniform FrameData` | Inline in GlobalParams | `const FrameData&` |
| `SamplerState` | `SamplerState` (dummy) | Ignored |

## Kernels and Entry Points

All 15 kernels are transpiled:

| Entry Point | Source File | Purpose |
|-------------|-------------|---------|
| `distApproximationKernel` | dist_approx.slang | Pass 0: Half-res LOD distance approximation |
| `GBufferAndDirectLight` | direct_light.slang | Pass 1: Primary rays + direct lighting |
| `IndirectBounce` | indirect_bounce.slang | Pass 2: 1-bounce global illumination |
| `BilateralUpsample` | bilateral_upsample.slang | Pass 2.5: Edge-aware upsample |
| `TemporalAccumulation` | temporal_acc.slang | Pass 3: Temporal reprojection |
| `BilateralDenoise` | denoise.slang | Pass 4: A-Trous denoiser |
| `VolumetricFog` | volumetric.slang | Pass 5: Volumetric fog |
| `ComputeExposure` | exposure.slang | Pass 6: Auto-exposure |
| `Composite` | composite.slang | Pass 7: Final composite |
| `FallbackBlit` | composite.slang | Pass 7b: Fallback upscale |
| `XMap_AnalyzeSectors` | material_gen.slang | Terrain: Sector analysis |
| `XMap_AnalyzeStreaming` | material_gen.slang | Terrain: Streaming analysis |
| `XMap_FillBricks` | material_gen.slang | Terrain: Brick filling |
| `FillDynamicAtlases` | material_gen.slang | Terrain: Atlas filling |
| `TextOverlay` | text_overlay.slang | Pass 8: SDF text rendering |

## Files Involved

### Input (Slang source)
- `src/renderer/kernels/slang/*.slang` - All shader source files

### Generated (build output)
- `build/slang_generated/<EntryPoint>.cu` - Raw Slang output
- `build/slang_generated/cuda_postproc/<EntryPoint>.cu` - Post-processed with wrappers
- `build/CudaShaderLaunchers.generated.cuh` - Header with Launch_* declarations

### Build scripts
- `CMakeLists.txt` - Orchestrates the build (lines 174-242 for Windows)
- `gen_cuda_wrappers.py` - Post-processes Slang output

### Host code
- `src/renderer/CUDA/CudaRenderer.cu` - Uses Launch_* functions
- `src/renderer/CUDA/CudaMaterialMap.cu` - Provides GPU buffers/textures

## Testing

The build path was verified on macOS by:
1. Running `slangc -target cuda` for each kernel (all 15 succeeded)
2. Running `gen_cuda_wrappers.py` on the generated files (success)
3. Verifying the generated wrapper functions match expected signatures

## Known Limitations

1. **Struct layout compatibility**: The Slang-generated structs (e.g., `CameraData_natural_0`) must be layout-compatible with the C++ structs (`CameraData`). Both use standard layout with matching field order.

2. **3D Texture for indirection**: Slang expects `CUtexObject` (cudaTextureObject_t) for the indirection texture. The CudaMaterialMap already creates a proper CUDA 3D texture for this.

3. **SamplerState ignored**: Slang generates `SamplerState` fields but CUDA bakes sampling state into the texture object. The wrapper sets these to nullptr.

## Future Improvements

- Consider caching GlobalParams uploads across frames for resources that don't change
- Profile __constant__ memory bandwidth vs kernel parameter passing
- Add automatic struct layout verification between C++ and Slang