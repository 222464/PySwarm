# CMake version 3.1+ is required to enable C++14 features cleanly
cmake_minimum_required(VERSION 3.1)

project(PySwarm)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/")

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(PYTHON_VERSION)
    message("Python target version: ${PYTHON_VERSION}")            
endif()


############################################################################
# Add the OpenMP library

find_package(OpenMP REQUIRED)
 
include_directories(${OpenMP_CXX_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") # This links -fopenmp


############################################################################
# Add the Swarm library

find_package(Swarm REQUIRED)

include_directories(${SWARM_INCLUDE_DIR})


############################################################################
# Add the PySwarm library

set(PYSWARM_INCLUDE_DIR "source/pyswarm;")

include_directories(${PYSWARM_INCLUDE_DIR})

file(GLOB_RECURSE PYSWARM_SRC
    "source/pyswarm/*.h"
    "source/pyswarm/*.cpp"
)

add_library(PySwarm ${PYSWARM_SRC})

target_link_libraries(PySwarm ${OpenMP_CXX_LIBRARIES})
target_link_libraries(PySwarm ${SWARM_LIBRARIES})


############################################################################
# Find SWIG and setup building the Python bindings to PySwarm library

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
endif()


############################################################################
# Find Python package

find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIR})


############################################################################
# Setup SWIG bindings

set(CMAKE_SWIG_OUTDIR "${CMAKE_BINARY_DIR}")

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

set_source_files_properties(pyswarm.i PROPERTIES CPLUSPLUS ON)

swig_add_module(pyswarm python pyswarm.i)

swig_link_libraries(pyswarm PySwarm ${PYTHON_LIBRARIES})

set(CMAKE_INSTALL_PREFIX ${CMAKE_SWIG_OUTDIR})
install(TARGETS _pyswarm DESTINATION ${CMAKE_SWIG_OUTDIR})
install(FILES ${CMAKE_BINARY_DIR}/pyswarm.py DESTINATION ${CMAKE_SWIG_OUTDIR})