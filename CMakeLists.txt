cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

SET(APP_NAME RVGRT)

# Set the project languages based on the target platform
if(WIN32)
    project(Programma C CXX CUDA)
else()
    project(Programma C CXX OBJCXX)
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
        src/renderer/CArray_impl.mm
        src/renderer/CoarseArray_impl.mm
        src/renderer/TexturePack_impl.mm
    )
    
    file(GLOB MAC_METAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/*.metal")
    set(MAC_METAL_AIR_FILES "")
    set(MAC_METAL_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/default.metallib")

    find_program(XCRUN_TOOL xcrun REQUIRED)

    # --- FIX: Loop over each .metal file and create a compile command for it ---
    foreach(METAL_FILE ${MAC_METAL_SOURCES})
        get_filename_component(METAL_FILENAME ${METAL_FILE} NAME_WE)
        set(AIR_FILE "${CMAKE_CURRENT_BINARY_DIR}/${METAL_FILENAME}.air")
        list(APPEND MAC_METAL_AIR_FILES ${AIR_FILE})

        add_custom_command(
            OUTPUT "${AIR_FILE}"
            COMMAND ${XCRUN_TOOL} metal -c
                    -I "${CMAKE_SOURCE_DIR}/include"
                    "${METAL_FILE}"
                    -o "${AIR_FILE}"
            DEPENDS "${METAL_FILE}"
            COMMENT "Compiling Metal shader ${METAL_FILE}"
        )
    endforeach()

    # --- FIX: Create the final library using ALL the compiled .air files ---
    add_custom_command(
        OUTPUT "${MAC_METAL_LIBRARY}"
        COMMAND ${XCRUN_TOOL} metallib -o "${MAC_METAL_LIBRARY}" ${MAC_METAL_AIR_FILES}
        DEPENDS ${MAC_METAL_AIR_FILES}
        COMMENT "Creating Metal library from ${MAC_METAL_AIR_FILES}"
    )
    
    add_custom_target(Shaders ALL DEPENDS "${MAC_METAL_LIBRARY}")

    # --- Dummy Implementation for CudaRenderer (contents unchanged) ---
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CudaRenderer_mac_stub.cpp"
        "
        #include \"renderer/Renderer.hpp\"
        class CudaRenderer : public Renderer {
        public:
            CudaRenderer() {}
            ~CudaRenderer() {}
            void Draw(const Character&, unsigned int) override {}
        };
        // This is necessary to satisfy the `new CudaRenderer()` in the Windows code path
        // which, although guarded, might still be seen by some analysis tools.
        // A better approach is a factory function, but this works for now.
        "
    )

    # --- Create the macOS Executable ---
    add_executable(${APP_NAME} MACOSX_BUNDLE
        ${MAC_SOURCES}
        "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CudaRenderer_mac_stub.cpp"
    )

    add_dependencies(${APP_NAME} Shaders)

        add_custom_command(
        TARGET ${APP_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "${MAC_METAL_LIBRARY}"
                "$<TARGET_FILE_DIR:${APP_NAME}>/../Resources/default.metallib"
        COMMENT "Copying default.metallib to app bundle..."
    )

    # --- Link against macOS Frameworks ---
    target_link_libraries(${APP_NAME} PRIVATE
        "-framework Cocoa"
        "-framework Metal"
        "-framework MetalKit"
        "-framework QuartzCore"
    )
endif()

# ------------------------------------------------------------------
#  Common Compile Options (Applied to both platforms)
# ------------------------------------------------------------------
target_compile_options(${APP_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr>
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