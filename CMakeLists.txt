cmake_minimum_required(VERSION 3.21)
project(libmorton_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- nanobind ---
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# --- libmorton via FetchContent ---
include(FetchContent)

FetchContent_Declare(
    libmorton
    GIT_REPOSITORY https://github.com/Forceflow/libmorton.git
    GIT_TAG        v0.2.12
    GIT_SHALLOW    TRUE
)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(libmorton)

# --- binding module ---
nanobind_add_module(
    _morton
    STABLE_ABI
    NB_STATIC
    ext/morton_ext.cpp
)

target_link_libraries(_morton PRIVATE libmorton::libmorton)

# --- BMI2 intrinsics (opt out with -DMORTON_USE_BMI2=OFF) ---
option(MORTON_USE_BMI2 "Enable BMI2 instructions for faster Morton codes" ON)
if(MORTON_USE_BMI2)
    if(MSVC)
        target_compile_options(_morton PRIVATE /arch:AVX2)
    else()
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mbmi2" HAS_BMI2)
        if(HAS_BMI2)
            target_compile_options(_morton PRIVATE -mbmi2)
        endif()
    endif()
endif()

install(TARGETS _morton LIBRARY DESTINATION libmorton)
