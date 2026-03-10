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

    # --- Find C preprocessor for shader preprocessing ---
    find_program(CPP_TOOL cpp NAMES cpp clang-cpp 
        PATHS "C:/Program Files/LLVM/bin" 
              "C:/msys64/mingw64/bin"
              "C:/mingw64/bin"
        DOC "C preprocessor for shader files")
    
    if(NOT CPP_TOOL)
        message(WARNING "C preprocessor (cpp) not found. Shader preprocessing disabled.")
        message(STATUS "Install LLVM or MinGW to enable shader preprocessing")
    else()
        message(STATUS "Found C preprocessor: ${CPP_TOOL}")
    endif()

        # --- Shader preprocessing ---
        if(CPP_TOOL)
        set(SHADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/renderer/kernels/shaders")
        set(CUDA_KERNEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda_kernels")
        file(MAKE_DIRECTORY ${CUDA_KERNEL_DIR})

        set(SHADER_FILES
            tables.shader
            dist_approx.shader
            direct_light.shader
            indirect_bounce.shader
            temporal_acc.shader
            denoise.shader
            volumetric.shader
            exposure.shader
            composite.shader
            material_gen.shader
        )

        set(CUDA_KERNEL_FILES "")
        foreach(SHADER ${SHADER_FILES})
            set(SHADER_PATH "${SHADER_SOURCE_DIR}/${SHADER}")
            set(CUDA_PATH "${CUDA_KERNEL_DIR}/${SHADER}.cu")
            list(APPEND CUDA_KERNEL_FILES ${CUDA_PATH})
            
            # Preprocess shader: define PLATFORM_CUDA, include shader_macros.h first
            # Note: shader_macros.h handles platform detection and conditional metal_stdlib include
            add_custom_command(
                OUTPUT ${CUDA_PATH}
                COMMAND ${CPP_TOOL} -E -P 
                    -D__CUDA_ARCH__=750 
                    -DPLATFORM_CUDA=1
                    -I${CMAKE_SOURCE_DIR}/include 
                    "${SHADER_PATH}" > "${CUDA_PATH}"
                DEPENDS "${SHADER_PATH}" "${CMAKE_SOURCE_DIR}/include/shader_macros.h"
                COMMENT "Preprocessing ${SHADER} for CUDA"
            )
        endforeach()

        add_custom_target(GenerateCudaKernels DEPENDS ${CUDA_KERNEL_FILES})
    endif()

    # --- ALL SOURCES for the Windows build ---
    set(WIN_SOURCES
        src/main.cpp
        src/State.cpp
        src/Character.cpp
        src/Camera.cpp
        src/platform/win32_main.cpp
        src/platform/WindowsPlatform.cpp
        src/renderer/D3D12Device.cpp
        src/renderer/TexturePack_impl.cu
        src/platform/Win32NetworkClient.cpp
    )
    
    # Core CUDA renderer files
    set(WIN_CUDA_SOURCES
        src/renderer/CudaRenderer.cu
        src/renderer/CudaMaterialMap.cu
        src/renderer/BrickPool_CUDA.cu
        src/CudaD3D12Texture.cu
    )
    
    # Add preprocessed kernel files if available
    if(CUDA_KERNEL_FILES)
        list(APPEND WIN_CUDA_SOURCES ${CUDA_KERNEL_FILES})
        message(STATUS "Using preprocessed CUDA kernels from shaders")
    else()
        message(WARNING "Shader preprocessing disabled. Kernels must be provided manually.")
    endif()

    # --- Create the Windows Executable ---
    add_executable(${APP_NAME} WIN32
        ${WIN_SOURCES}
        ${WIN_CUDA_SOURCES}
    )
    
    # Add dependency on shader preprocessing
    if(CPP_TOOL AND CUDA_KERNEL_FILES)
        add_dependencies(${APP_NAME} GenerateCudaKernels)
    endif()
    
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
        src/platform/macos_main.mm
        src/platform/MacOSPlatform.mm
        src/renderer/MetalDevice.mm
        src/renderer/MetalRenderer.mm
        src/renderer/MetalBuffer.mm
        src/renderer/TexturePack_impl.mm
        src/platform/GameView.mm
        src/renderer/MaterialMap.mm
        src/renderer/BrickPool_Metal.mm
        src/platform/MacOSNetworkClient.mm
        ${MACOS_ICON} 
    )
    
    # --- Metal shader compilation ---
    # Compile .shader files directly (they contain PLATFORM_METAL preprocessor directives)
    set(SHADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/renderer/kernels/shaders")
    
    set(SHADER_FILES
        tables.shader
        dist_approx.shader
        direct_light.shader
        indirect_bounce.shader
        temporal_acc.shader
        denoise.shader
        volumetric.shader
        exposure.shader
        composite.shader
        material_gen.shader
    )
    
    find_program(XCRUN_TOOL xcrun REQUIRED)
    message(STATUS "Found xcrun for Metal shader compilation")
    
    set(MAC_METAL_SOURCES "")
    foreach(SHADER ${SHADER_FILES})
        set(SHADER_PATH "${SHADER_SOURCE_DIR}/${SHADER}")
        list(APPEND MAC_METAL_SOURCES ${SHADER_PATH})
    endforeach()    
    set(MAC_METAL_AIR_FILES "")
    set(MAC_METAL_LIBRARY_SRC "${CMAKE_CURRENT_BINARY_DIR}/default.metallib") 

    find_program(XCRUN_TOOL xcrun REQUIRED)

    foreach(METAL_FILE ${MAC_METAL_SOURCES})
        get_filename_component(METAL_FILENAME ${METAL_FILE} NAME_WE)
        set(AIR_FILE "${CMAKE_CURRENT_BINARY_DIR}/${METAL_FILENAME}.air")
        list(APPEND MAC_METAL_AIR_FILES ${AIR_FILE})

        add_custom_command(
            OUTPUT "${AIR_FILE}"
            COMMAND ${XCRUN_TOOL} metal -c
                    -x metal
                    -DPLATFORM_METAL=1
                    -frecord-sources
                    -gline-tables-only
                    -I "${CMAKE_SOURCE_DIR}/include"
                    "${METAL_FILE}"
                    -o "${AIR_FILE}"
            DEPENDS "${METAL_FILE}" "${CMAKE_SOURCE_DIR}/include/shader_macros.h"
            COMMENT "Compiling Metal shader ${METAL_FILE}"
        )
    endforeach()

    add_custom_command(
        OUTPUT "${MAC_METAL_LIBRARY_SRC}"
        COMMAND ${XCRUN_TOOL} metallib -o "${MAC_METAL_LIBRARY_SRC}" ${MAC_METAL_AIR_FILES}
        DEPENDS ${MAC_METAL_AIR_FILES}
        COMMENT "Creating Metal library from ${MAC_METAL_AIR_FILES}"
    )

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
        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CudaRenderer_mac_stub.cpp"
        "${MAC_METAL_LIBRARY_SRC}" 
    )
    add_custom_command(TARGET ${APP_NAME} POST_BUILD
    COMMAND xattr -cr "$<TARGET_BUNDLE_DIR:${APP_NAME}>"
    COMMENT "Stripping extended attributes to prevent CodeSign detritus error"
)

    add_dependencies(${APP_NAME} BuildMetalShaders)

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
