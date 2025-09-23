# 🟩 Voxel World Engine (WIP)

This is an attempt at creating a **Minecraft-like voxel world engine**, built mostly with CUDA for accelerated rendering. This is a platform which allows me to try certain algorithms and datastructures and optimization techniques for realtime rendering.  

---

## Features (so far)
- Procedural voxel world generation.
- CUDA + DirectX 12 interop for GPU-accelerated rendering.
- GPU accelerated world generation
- GPU accelerated coarse signed distance field creation and usage
- Lower resolution estimation of primary ray distance and shadows.
- Hybrid based voxel raytracing algorithm consisting of distance estimation and DDA for analytical normals.
- Implemenation of global illumination, with voxel cone tracing for smooth shadows
- Usage of a texturepack, shadows and reflections
- Usage of DLSS image upscaling
---

## Requirements
- **OS**: Windows only currently (uses the Win32 API)  
- **Graphics**: Direct3D 12 + CUDA-capable GPU (e.g., NVIDIA)  


##  Build Instructions
Make sure you have the following installed:
- [CMake](https://cmake.org/) **3.18+**
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) **12.6+** (tested)
- [Visual Studio 2022](https://visualstudio.microsoft.com/vs/) (MSVC compiler)

### Steps
```bash
# Clone this repository
git clone https://github.com/RubenVlieger/RVGRT.git
cd RVGRT

# Create build directory
mkdir build && cd build

# Generate project files
cmake ..

# Build & run (or open in Visual Studio)



For further questions, please reach me by my email: ruben.vlieger@ru.nl
