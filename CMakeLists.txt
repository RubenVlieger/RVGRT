cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

SET(APP_NAME RVGRT)
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum OS X deployment version")

# Set the project languages based on the target platform
if(WIN32)
    project(RVGRT C CXX CUDA)
else()
    project(RVGRT C CXX OBJCXX)
endif()

message("Project name: " ${CMAKE_PROJECT_NAME})

# ------------------------------------------------------------------
#  Slang Compiler Setup
# ------------------------------------------------------------------
include(FetchContent)

set(SLANG_VERSION "2026.7.1")
if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-macos-aarch64.zip")
    else()
        set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-macos-x86_64.zip")
    endif()
elseif(WIN32)
    set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-windows-x86_64.zip")
else()
    set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64.zip")
endif()

FetchContent_Declare(
    slang_binary
    URL ${SLANG_URL}
)
FetchContent_Populate(slang_binary)

find_program(SLANGC slangc PATHS ${slang_binary_SOURCE_DIR}/bin NO_DEFAULT_PATH REQUIRED)
message(STATUS "Found slangc: ${SLANGC}")

set(SLANG_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/renderer/kernels/slang")
set(SLANG_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/slang_generated")
file(MAKE_DIRECTORY ${SLANG_GENERATED_DIR})

set(SLANG_LIB_FILES
    ${SLANG_SOURCE_DIR}/ShaderTypes.slang
    ${SLANG_SOURCE_DIR}/Tables.slang
    ${SLANG_SOURCE_DIR}/Math.slang
    ${SLANG_SOURCE_DIR}/TerrainGen.slang
    ${SLANG_SOURCE_DIR}/Intersection.slang
    ${SLANG_SOURCE_DIR}/Raytracing.slang
    ${SLANG_SOURCE_DIR}/Settings.slang
)

set(SLANG_KERNEL_FILES
    ${SLANG_SOURCE_DIR}/dist_approx.slang
    ${SLANG_SOURCE_DIR}/direct_light.slang
    ${SLANG_SOURCE_DIR}/indirect_bounce.slang
    ${SLANG_SOURCE_DIR}/bilateral_upsample.slang
    ${SLANG_SOURCE_DIR}/temporal_acc.slang
    ${SLANG_SOURCE_DIR}/denoise.slang
    ${SLANG_SOURCE_DIR}/volumetric.slang
    ${SLANG_SOURCE_DIR}/exposure.slang
    ${SLANG_SOURCE_DIR}/composite.slang
    ${SLANG_SOURCE_DIR}/material_gen.slang
    ${SLANG_SOURCE_DIR}/text_overlay.slang
)

set(SLANG_ENTRY_POINTS
    -entry distApproximationKernel -stage compute
    -entry GBufferAndDirectLight -stage compute
    -entry IndirectBounce -stage compute
    -entry BilateralUpsample -stage compute
    -entry TemporalAccumulation -stage compute
    -entry BilateralDenoise -stage compute
    -entry VolumetricFog -stage compute
    -entry ComputeExposure -stage compute
    -entry Composite -stage compute
    -entry FallbackBlit -stage compute
    -entry XMap_AnalyzeSectors -stage compute
    -entry XMap_AnalyzeStreaming -stage compute
    -entry XMap_FillBricks -stage compute
    -entry FillDynamicAtlases -stage compute
    -entry TextOverlay -stage compute
)

# Set language standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)

# Add the 'include' directory for headers
include_directories(include)

# ------------------------------------------------------------------
#  Platform-Specific Build Configuration
# ------------------------------------------------------------------

if(WIN32)
    #================================================#
    #======= WINDOWS (D3D12 + CUDA) BUILD PATH ======#
    #================================================#
    message(STATUS "Configuring for Windows (D3D12 + CUDA)")

    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_INCLUDE_DIRS}")

    # --- ALL SOURCES for the Windows build ---
    set(WIN_SOURCES
        src/main.cpp
        src/State.cpp
        src/Character.cpp
        src/Camera.cpp
        src/renderer/FrameDataManager.cpp
        src/platform/win32_main.cpp
        src/platform/WindowsPlatform.cpp
        src/renderer/D3D12/D3D12Device.cpp
        src/renderer/TexturePack_impl.cu
        src/platform/Win32NetworkClient.cpp
        src/console/ConsoleBuffer.cpp
        src/console/ConsoleInput.cpp
        src/console/CommandRegistry.cpp
        src/console/GameConsole.cpp
        src/console/RegisterCommands.cpp
    )
    
    # Core CUDA renderer files
    set(WIN_CUDA_SOURCES
        src/renderer/CUDA/CudaRenderer.cu
        src/renderer/CUDA/CudaTraits.cu
        src/renderer/CUDA/CudaMaterialMap.cu
        src/renderer/CUDA/BrickPool_CUDA.cu
        src/CudaD3D12Texture.cu
    )
    
    # --- Slang shader compilation ---
    set(SLANG_GENERATED_CUDA "${SLANG_GENERATED_DIR}/Generated.cu")
    
    add_custom_command(
        OUTPUT "${SLANG_GENERATED_CUDA}"
        COMMAND ${SLANGC}
            -target cuda
            ${SLANG_ENTRY_POINTS}
            -I "${CMAKE_SOURCE_DIR}/include"
            -I "${CMAKE_SOURCE_DIR}/src/renderer/kernels/slang"
            -D__SLANG__
            ${SLANG_KERNEL_FILES}
            -o "${SLANG_GENERATED_CUDA}"
        DEPENDS ${SLANG_LIB_FILES} ${SLANG_KERNEL_FILES}
        COMMENT "Transpiling Slang shaders to CUDA C++..."
    )
    
    list(APPEND WIN_CUDA_SOURCES "${SLANG_GENERATED_CUDA}")
    set_source_files_properties("${SLANG_GENERATED_CUDA}" PROPERTIES LANGUAGE CUDA)
    
    message(STATUS "Using Slang shader sources: ${SLANG_KERNEL_FILES}")

    # --- Create the Windows Executable ---
    add_executable(${APP_NAME} WIN32
        ${WIN_SOURCES}
        ${WIN_CUDA_SOURCES}
    )
    
    target_sources(${APP_NAME} PRIVATE exports.def)

    # --- CUDA Properties ---
    set_property(TARGET ${APP_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${APP_NAME} PROPERTY CUDA_ARCHITECTURES native)

    # --- DirectX Linking ---
    target_link_libraries(${APP_NAME} PRIVATE d3d12 dxgi dxguid)
    add_compile_definitions(D3D12)
    
    # --- Optional Streamline SDK ---
    option(USE_STREAMLINE "Enable NVIDIA Streamline SDK (DLSS)" OFF)
    if(USE_STREAMLINE)
        set(STREAMLINE_SDK_PATH "${CMAKE_SOURCE_DIR}/vendor/streamline-sdk-v2.9.0")
        if(EXISTS "${STREAMLINE_SDK_PATH}/include/sl.h")
            message(STATUS "Streamline SDK found, enabling DLSS support")
            target_compile_definitions(${APP_NAME} PRIVATE HAS_STREAMLINE)
            target_include_directories(${APP_NAME} PRIVATE "${STREAMLINE_SDK_PATH}/include")
            target_link_directories(${APP_NAME} PRIVATE "${STREAMLINE_SDK_PATH}/lib/x64")
            target_link_libraries(${APP_NAME} PRIVATE sl.interposer)
            
            add_custom_command(
                TARGET ${APP_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                    "${STREAMLINE_SDK_PATH}/bin/x64/sl.interposer.dll" 
                    $<TARGET_FILE_DIR:${APP_NAME}>
                COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                    "${STREAMLINE_SDK_PATH}/bin/x64/sl.dlss.dll" 
                    $<TARGET_FILE_DIR:${APP_NAME}>
                COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                    "${STREAMLINE_SDK_PATH}/bin/x64/nvngx_dlss.dll" 
                    $<TARGET_FILE_DIR:${APP_NAME}>
                COMMENT "Copying Streamline DLLs to output directory..."
            )
        else()
            message(WARNING "Streamline SDK not found at ${STREAMLINE_SDK_PATH}, DLSS disabled")
        endif()
    endif()

elseif(APPLE)
    #======================================#
    #======= MACOS (METAL) BUILD PATH ======#
    #======================================#
    message(STATUS "Configuring for macOS (Metal)")

    set(MACOS_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/AppIcon.icns")

    set_source_files_properties(${MACOS_ICON} PROPERTIES
        MACOSX_PACKAGE_LOCATION Resources
    )

    set(MAC_SOURCES
        src/main.cpp
        src/State.cpp
        src/Character.cpp
        src/Camera.cpp
        src/renderer/FrameDataManager.cpp
        src/renderer/FontAtlas.cpp
        src/renderer/RendererBase.mm
        src/platform/macos_main.mm
        src/platform/MacOSPlatform.mm
        src/renderer/Metal/MetalDevice.mm
        src/renderer/Metal/MetalRenderer.mm
        src/renderer/Metal/MetalTraits.mm
        src/renderer/Metal/MetalBuffer.mm
        src/renderer/TexturePack_impl.mm
        src/platform/GameView.mm
        src/renderer/MaterialMap.mm
        src/renderer/Metal/BrickPool_Metal.mm
        src/renderer/TextRenderer.mm
        src/platform/MacOSNetworkClient.mm
        src/console/ConsoleBuffer.cpp
        src/console/ConsoleInput.cpp
        src/console/CommandRegistry.cpp
        src/console/GameConsole.cpp
        src/console/RegisterCommands.cpp
        ${MACOS_ICON}
    )
    
    # --- Slang to Metal shader compilation ---
    find_program(XCRUN_TOOL xcrun REQUIRED)
    message(STATUS "Found xcrun for Metal shader compilation")

    set(MAC_METAL_LIBRARY_SRC "${CMAKE_CURRENT_BINARY_DIR}/default.metallib")

    # Kernel-specific entry points and source files
    set(KERNEL_SOURCES
        dist_approx.slang
        direct_light.slang
        indirect_bounce.slang
        bilateral_upsample.slang
        temporal_acc.slang
        denoise.slang
        volumetric.slang
        exposure.slang
        composite.slang
        composite.slang
        material_gen.slang
        material_gen.slang
        material_gen.slang
        material_gen.slang
        text_overlay.slang
    )

    set(KERNEL_ENTRY_POINTS
        distApproximationKernel
        GBufferAndDirectLight
        IndirectBounce
        BilateralUpsample
        TemporalAccumulation
        BilateralDenoise
        VolumetricFog
        ComputeExposure
        Composite
        FallbackBlit
        XMap_AnalyzeSectors
        XMap_AnalyzeStreaming
        XMap_FillBricks
        FillDynamicAtlases
        TextOverlay
    )

    list(LENGTH KERNEL_SOURCES KERNEL_COUNT)
    math(EXPR KERNEL_LAST "${KERNEL_COUNT} - 1")

    set(SLANG_GENERATED_METAL_FILES "")
    set(SLANG_GENERATED_AIR_FILES "")

    foreach(IDX RANGE 0 ${KERNEL_LAST})
        list(GET KERNEL_SOURCES ${IDX} KERNEL_SOURCE)
        list(GET KERNEL_ENTRY_POINTS ${IDX} KERNEL_ENTRY_POINT)

        set(GENERATED_METAL_FILE "${SLANG_GENERATED_DIR}/${KERNEL_ENTRY_POINT}.metal")
        set(GENERATED_AIR_FILE "${SLANG_GENERATED_DIR}/${KERNEL_ENTRY_POINT}.air")
        list(APPEND SLANG_GENERATED_METAL_FILES "${GENERATED_METAL_FILE}")
        list(APPEND SLANG_GENERATED_AIR_FILES "${GENERATED_AIR_FILE}")

        add_custom_command(
            OUTPUT "${GENERATED_METAL_FILE}"
            COMMAND ${SLANGC}
                -target metal
                -entry ${KERNEL_ENTRY_POINT}
                -stage compute
                -I "${CMAKE_SOURCE_DIR}/include"
                -I "${CMAKE_SOURCE_DIR}/src/renderer/kernels/slang"
                -D__SLANG__
                "${CMAKE_SOURCE_DIR}/src/renderer/kernels/slang/${KERNEL_SOURCE}"
                -o "${GENERATED_METAL_FILE}"
            DEPENDS "${CMAKE_SOURCE_DIR}/src/renderer/kernels/slang/${KERNEL_SOURCE}"
                   ${SLANG_LIB_FILES}
            COMMENT "Transpiling ${KERNEL_ENTRY_POINT} to MSL..."
        )

        add_custom_command(
            OUTPUT "${GENERATED_AIR_FILE}"
            COMMAND ${XCRUN_TOOL} metal -c
                    -x metal
                    -frecord-sources
                    -gline-tables-only
                    -I "${CMAKE_SOURCE_DIR}/include"
                    "${GENERATED_METAL_FILE}"
                    -o "${GENERATED_AIR_FILE}"
            DEPENDS "${GENERATED_METAL_FILE}"
            COMMENT "Compiling ${KERNEL_ENTRY_POINT} MSL to AIR..."
        )
    endforeach()

    add_custom_command(
        OUTPUT "${MAC_METAL_LIBRARY_SRC}"
        COMMAND ${XCRUN_TOOL} metallib -o "${MAC_METAL_LIBRARY_SRC}" ${SLANG_GENERATED_AIR_FILES}
        DEPENDS ${SLANG_GENERATED_AIR_FILES}
        COMMENT "Linking AIR files to metallib..."
    )

    # --- Generate shader binding constants header from compiled .metal files ---
    set(SHADER_BINDINGS_HEADER "${CMAKE_CURRENT_BINARY_DIR}/ShaderBindings.generated.hpp")
    add_custom_command(
        OUTPUT "${SHADER_BINDINGS_HEADER}"
        COMMAND python3 "${CMAKE_SOURCE_DIR}/gen_shader_bindings.py" "${SLANG_GENERATED_DIR}" "${SHADER_BINDINGS_HEADER}"
        DEPENDS ${SLANG_GENERATED_METAL_FILES}
        COMMENT "Generating shader binding constants..."
    )
    add_custom_target(GenerateShaderBindings DEPENDS "${SHADER_BINDINGS_HEADER}")

    add_custom_target(BuildMetalShaders DEPENDS "${MAC_METAL_LIBRARY_SRC}")
    
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CudaRenderer_mac_stub.cpp"
        "#include \"renderer/Renderer.hpp\"
        class CudaRenderer : public Renderer {
        public:
            CudaRenderer() {}
            ~CudaRenderer() {}
            void Draw(const Character&, unsigned int) override {}
        };"
    )

    set_source_files_properties("${MAC_METAL_LIBRARY_SRC}" PROPERTIES
        MACOSX_PACKAGE_LOCATION Resources
    )

    add_executable(${APP_NAME} MACOSX_BUNDLE
        ${MAC_SOURCES}
        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CUDA/CudaRenderer_mac_stub.cpp"
        "${MAC_METAL_LIBRARY_SRC}" 
    )
    add_custom_command(TARGET ${APP_NAME} POST_BUILD
    COMMAND xattr -cr "$<TARGET_BUNDLE_DIR:${APP_NAME}>"
    COMMENT "Stripping extended attributes to prevent CodeSign detritus error"
)

    add_dependencies(${APP_NAME} BuildMetalShaders GenerateShaderBindings)
    target_include_directories(${APP_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    #set_target_properties(${APP_NAME} PROPERTIES UNITY_BUILD ON)
    target_compile_options(${APP_NAME} PRIVATE -fobjc-arc)
    
    set_target_properties(${APP_NAME} PROPERTIES
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.rvgrt.voxelengine"
        MACOSX_BUNDLE_BUNDLE_NAME "RVGRT"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
        MACOSX_BUNDLE_ICON_FILE "AppIcon.icns" 
        MACOSX_BUNDLE_BUNDLE_VERSION "1"
        PRODUCT_BUNDLE_IDENTIFIER "com.rvgrt.voxelengine"
        RESOURCE "${MAC_METAL_LIBRARY_SRC}"

        XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "-"
        XCODE_ATTRIBUTE_DEVELOPMENT_TEAM ""
        XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Manual"
    )
    target_link_libraries(${APP_NAME} PRIVATE
        "-framework Cocoa"
        "-framework Metal"
        "-framework MetalKit"
        "-framework QuartzCore"
        "-framework MetalFX" 
    )
endif()


set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
set(CMAKE_OBJCXX_COMPILER_LAUNCHER ccache)


# ------------------------------------------------------------------
#  Common Compile Options (Applied to both platforms)
# ------------------------------------------------------------------
target_compile_options(${APP_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${APP_NAME} PRIVATE _DEBUG)
    if(WIN32)
        target_compile_options(${APP_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${APP_NAME} PRIVATE -O3 -DNDEBUG)
    if(WIN32)
        target_compile_options(${APP_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    endif()
endif()