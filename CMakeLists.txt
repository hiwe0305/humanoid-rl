cmake_minimum_required(VERSION 3.18)
project(humanoid_rl_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release build optimizations
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Find HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)

# Find OpenMP for parallel stepping
find_package(OpenMP REQUIRED)

# ─── C++ sources ─────────────────────────────────────────────────────────────
set(CPP_SOURCES
    csrc/envpool/rollout_buffer.cpp
    csrc/io/hdf5_writer.cpp
    csrc/bindings.cpp
)

# ─── pybind11 module ─────────────────────────────────────────────────────────
pybind11_add_module(humanoid_rl_cpp ${CPP_SOURCES})

target_include_directories(humanoid_rl_cpp PRIVATE
    csrc/
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(humanoid_rl_cpp PRIVATE
    ${HDF5_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Thread support
find_package(Threads REQUIRED)
target_link_libraries(humanoid_rl_cpp PRIVATE Threads::Threads)

# Compile definitions
target_compile_definitions(humanoid_rl_cpp PRIVATE
    VERSION_INFO="${PROJECT_VERSION}"
)

# ─── Install into venv site-packages ─────────────────────────────────────────
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

install(TARGETS humanoid_rl_cpp
    LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Python site-packages: ${PYTHON_SITE_PACKAGES}")
message(STATUS "HDF5 libraries: ${HDF5_CXX_LIBRARIES}")
