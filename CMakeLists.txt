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

    # --- ALL SOURCES for the Windows build ---
    set(WIN_SOURCES
        src/main.cpp
        src/State.cpp
        src/Character.cpp
        src/Camera.cpp
        src/platform/win32_main.cpp
        src/platform/WindowsPlatform.cpp
        src/renderer/D3D12Device.cpp
        src/platform/Win32NetworkClient.cpp
    )
    file(GLOB_RECURSE WIN_CUDA_SOURCES CONFIGURE_DEPENDS "src/*.cu")

    # --- Create the Windows Executable ---
    add_executable(${APP_NAME} WIN32
        ${WIN_SOURCES}
        ${WIN_CUDA_SOURCES}
    )
    target_sources(${APP_NAME} PRIVATE exports.def)

    # --- CUDA Properties ---
    set_property(TARGET ${APP_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${APP_NAME} PROPERTY CUDA_ARCHITECTURES native)

    # --- DirectX and Streamline Linking ---
    target_link_libraries(${APP_NAME} PRIVATE d3d12 dxgi dxguid)
    add_compile_definitions(D3D12)
    
    set(STREAMLINE_SDK_PATH "${CMAKE_SOURCE_DIR}/vendor/streamline-sdk-v2.9.0")
    target_include_directories(${APP_NAME} PRIVATE "${STREAMLINE_SDK_PATH}/include")
    target_link_directories(${APP_NAME} PRIVATE "${STREAMLINE_SDK_PATH}/lib/x64")
    target_link_libraries(${APP_NAME} PRIVATE sl.interposer)
    
    add_custom_command(
        TARGET ${APP_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${STREAMLINE_SDK_PATH}/bin/x64/sl.interposer.dll" $<TARGET_FILE_DIR:${APP_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${STREAMLINE_SDK_PATH}/bin/x64/sl.dlss.dll" $<TARGET_FILE_DIR:${APP_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${STREAMLINE_SDK_PATH}/bin/x64/nvngx_dlss.dll" $<TARGET_FILE_DIR:${APP_NAME}>
        COMMENT "Copying Streamline DLLs to output directory..."
    )

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
        src/renderer/BrickPool.mm
        src/platform/MacOSNetworkClient.mm
        ${MACOS_ICON} 
    )
    
    file(GLOB MAC_METAL_SOURCES CONFIGURE_DEPENDS 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/*.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/kernels/*.metal"
    )    
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
                    -frecord-sources
                    -gline-tables-only
                    -I "${CMAKE_SOURCE_DIR}/include"
                    "${METAL_FILE}"
                    -o "${AIR_FILE}"
            DEPENDS "${METAL_FILE}"
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
