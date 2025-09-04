# 🟩 Voxel World Engine (WIP)

This is yet another attempt at creating a **Minecraft-like voxel world engine**, built with CUDA for efficient rendering, a platform which allows me to try certain algorithms and datastructures and optimization techniques for realtime rendering.  

---

## Features (so far)
- Procedural voxel world generation (Minecraft-like).
- CUDA + DirectX 11 interop for GPU-accelerated rendering.
- GPU accelerated world generation
- GPU accelerated coarse signed distance field creation
- Lower resolution estimation of z-buffer (as optimization technique)
- Dual based voxel raytracing algorithm consisting of distance estimation and final DDA for precise hits.
- Usage of a texturepack, shadows and reflections
- Currently a 600kb executable of which half is the texturepack.
- Experimental engine

---

## Requirements
- **OS**: Windows only currently (uses the Win32 API)  
- **Graphics**: Direct3D 11 + CUDA-capable GPU (e.g., NVIDIA)  


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
