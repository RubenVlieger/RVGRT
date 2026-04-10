# 🟩 Voxel World Engine

This is an attempt at creating a **Minecraft-like voxel world engine**, built mostly with CUDA and Metal (in this dev phase only Metal supported!) for accelerated rendering, in a mostly cross platform way with a shared codebase. This is a platform which allows me to try certain algorithms and datastructures and optimization techniques for realtime rendering.  

---

## Features (so far)
- Procedural voxel world generation.
- Either Metal + Metal Compute or CUDA + DirectX 12 interop for GPU-accelerated rendering.
- GPU accelerated world generation
- Realistic shadows and reflections
- Realistic volumetric lighting
- Color grading
- Automatic exposure
- Usage of DLSS image upscaling for windows and MetalFX upscaling on Mac
- 3 Level grid sparse data structure chosen for storing material ID's and accelerated rendering
- Lower resolution estimation of primary ray distance.
- Path tracing approach for with indirect lighting with multiple samples per pixel, which produces global illumination and realistic results
- Procedurally generated world, where new sections are swapped in memory as you fly around
- Deferred rendering pipeline consisting of multiple kernels
- A-Trous edge aware denoising of the noisy indirect lighting
- A-Trous edge aware denoising of noisy volumetric lighting data
- Usage of the orignial Minecraft texturepack
- Animated character model
- SDF (Signed Distance Field) text rendering via compute shader overlay
- GPU-accelerated text with tile-culled rendering for performance
- **In-game console with 20 commands** — press `T` to chat or `/` for commands
  - Navigation: `/spawn`, `/home`, `/sethome`, `/tp <x> <y> <z>`, `/jump <dx> <dy> <dz>`
  - Settings: `/speed`, `/gravity`, `/fly`, `/noclip`, `/fov`, `/sensitivity`, `/reset`
  - Info: `/pos`, `/fps`, `/players`, `/me`, `/time`, `/clear`
  - Adding commands: see `src/console/RegisterCommands.cpp`


## To-do list:
- Clouds
- Realistic lens effects
- A plane to fly around in
- Ability to upload voxel scenes to render
- Ability to walk around and possibly interact with the world
- Full fly / noclip mode



## Requirements
- **OS**: MacOS or Windows
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
