# 🟩 Voxel World Engine (WIP)

This is an attempt at creating a **Minecraft-like voxel world engine**, built mostly with CUDA and Metal for accelerated rendering, in a mostly cross platform way with a big shared codebase. This is a platform which allows me to try certain algorithms and datastructures and optimization techniques for realtime rendering.  

---

## Features (so far)
- Procedural voxel world generation.
- CUDA + DirectX 12 interop for GPU-accelerated rendering.
- GPU accelerated world generation
- GPU accelerated signed distance field (SDF) creation and usage
- Lower resolution estimation of primary ray distance.
- Hybrid based voxel raytracing algorithm: consisting of distance estimation (SDF accelerated) and DDA for precise hits and normals. 
- Path tracing approach with multiple samples per pixel, which produces global illumination
- Deferred rendering pipeline consisting of multiple kernels
- A-Trous edge aware denoising of the indirect lighting 
- Usage of the orignial Minecraft texturepack
- Implemented shadows and reflections
- Volumetric lighting
- Automatic exposure
- Usage of DLSS image upscaling for windows 


## To-do list:
- Clouds
- Realistic lens effects
- A plane to fly around in
- Ability to upload voxel scenes to render
- Ability to walk around and possibly interact with the world
- A compact and fast data structure for storing materialID's


## Requirements
- **OS**: Windows, and MacOS
- **Graphics**: Discrete GPU with >4gb video memory recommended   


##  Build Instructions
Make sure you have the following installed:
- [CMake](https://cmake.org/) **3.18+**
- Either the MSVC compiler (Windows) or Xcode development tools (MacOS) and Metal shader compiler required
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
