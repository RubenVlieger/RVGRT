cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
SET(APP_NAME Programma)
project (Programma C CXX CUDA)

option(USE_D3D12 "Use Direct3D 12 instead of Direct3D 11" ON)

if (MSVC)
  set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe")
  set(CMAKE_CXX_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.38.33130/bin/Hostx64/x64/cl.exe")
endif()

if (UNIX)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

if(WIN32)
    message(STATUS "Project is building for Windows")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3")
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
endif()

message("Project name: " ${CMAKE_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

if (MSVC)
    message(STATUS "Building for Windows with MSVC")
    set(_CRT_SECURE_NO_WARNINGS 1)
    set(CURL_STATICLIB 1)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif()

include_directories(include)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(STREAMLINE_SDK_PATH "${CMAKE_SOURCE_DIR}/vendor/streamline-sdk-v2.9.0")
set(STREAMLINE_INCLUDE_DIR "${STREAMLINE_SDK_PATH}/include")
set(STREAMLINE_LIB_DIR "${STREAMLINE_SDK_PATH}/lib/x64")
set(STREAMLINE_BIN_DIR "${STREAMLINE_SDK_PATH}/bin/x64")

include_directories(${STREAMLINE_INCLUDE_DIR})

file(GLOB SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.c" "src/*.cu") 

if(WIN32)
    add_executable(${APP_NAME} WIN32 ${SOURCES})
else()
    add_executable(${APP_NAME} ${SOURCES})
endif()

target_sources(${APP_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/exports.def)

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr --ptxas-options=-allow-expensive-optimizations=true>
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61 75 87) # Running on GTX 1080 Ti, GTX 1660 Ti, and Jetson AGX Orin
set_property(TARGET ${APP_NAME} PROPERTY CUDA_ARCHITECTURES native)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _DEBUG)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


if(USE_D3D12)
    link_directories(${})
    target_link_libraries(${APP_NAME} PRIVATE d3d12 dxgi dxguid)
    add_compile_definitions(D3D12)
else()
    target_link_libraries(${APP_NAME} PRIVATE d3d11 dxgi dxguid)
endif()

target_include_directories(${APP_NAME} PRIVATE "${STREAMLINE_INCLUDE_DIR}")
target_link_directories(${APP_NAME} PRIVATE "${STREAMLINE_LIB_DIR}")
# Note: CMake automatically finds 'sl.interposer.lib' from 'sl.interposer'
target_link_libraries(${APP_NAME} PRIVATE sl.interposer)

# 2. Link DirectX

#target_link_libraries(${APP_NAME} PRIVATE CUDA::cudart)

message(STATUS "Setting up post-build command to copy Streamline DLLs")
add_custom_command(
    TARGET ${APP_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${STREAMLINE_BIN_DIR}/sl.interposer.dll"
    $<TARGET_FILE_DIR:${APP_NAME}>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${STREAMLINE_BIN_DIR}/sl.dlss.dll"
    $<TARGET_FILE_DIR:${APP_NAME}>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${STREAMLINE_BIN_DIR}/nvngx_dlss.dll"

    $<TARGET_FILE_DIR:${APP_NAME}>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${STREAMLINE_BIN_DIR}/sl.dlss_g.dll" # Optional: for debugging
    $<TARGET_FILE_DIR:${APP_NAME}>
    COMMENT "Copying Streamline DLLs to output directory..."
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Building in Release mode")
    if(MSVC)
        target_compile_options(${APP_NAME} PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
        target_compile_options(${APP_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    else()
        target_compile_options(${APP_NAME} PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
    endif()
endif()

if(NN_HALF_FLOAT)
    message(STATUS "Using half precision floating for neural network")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE NN_HALF_FLOAT)
else()
    message(STATUS "Using full precision floating for neural network")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE NN_FULL_FLOAT)
endif()

if(ACTIVATION_NOTHING)
    message(STATUS "Not using an activation function")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ACTIVATION_NOTHING)
elseif(ACTIVATION_MICK_RELU)
    message(STATUS "Using the Mick ReLU activation function")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ACTIVATION_MICK_RELU)
endif()
# add_custom_command(TARGET ${APP_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     ${D3D12_SDK_DLLS} $<TARGET_FILE_DIR:${APP_NAME}>
# )

# SET(APP_NAME Programma)
# project (Programma)

# cmake_minimum_required (VERSION 3.2)

# set(CMAKE_CXX_STANDARD 20)

# list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda")

# find_package(CUDAToolkit REQUIRED)
# set(CMAKE_CUDA_STANDARD 11)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# include(protect_nvcc_flags)
# include(protect_pthread_flags)

# file(GLOB SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*cu")
# include_directories(include)

# # add_definitions(-DUNICODE -D_UNICODE)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

# add_executable(${APP_NAME} ${SOURCES})

# target_link_libraries(${APP_NAME} PUBLIC d2d1.lib)

# install(TARGETS ${APP_NAME} DESTINATION bin)
