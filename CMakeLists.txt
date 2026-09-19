cmake_minimum_required(VERSION 3.20)
project(rmab_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ensure Python 3.12 is found
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Find pybind11 via CMake config, python -m pybind11 --cmake, or FetchContent fallback
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmake
        OUTPUT_VARIABLE pybind11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_EXIT_CODE
    )
    if(pybind11_EXIT_CODE EQUAL 0 AND EXISTS "${pybind11_CMAKE_DIR}")
        list(APPEND CMAKE_PREFIX_PATH "${pybind11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# pybind11 module rmab_cpp
pybind11_add_module(rmab_cpp src/bindings.cpp)

target_include_directories(rmab_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/hardware
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(rmab_cpp PRIVATE
    -O3
    -fPIC
    -Wall
    -Wextra
)

# Output compiled .so directly to repository root
set_target_properties(rmab_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Standalone microsecond latency & dwell timing verification binary
add_executable(test_timing hardware/test_timing.cpp)
target_include_directories(test_timing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hardware)
target_compile_options(test_timing PRIVATE -O3)
set_target_properties(test_timing PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/hardware"
)
