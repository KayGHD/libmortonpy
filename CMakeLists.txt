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
    GIT_TAG        main
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

install(TARGETS _morton LIBRARY DESTINATION libmorton)
