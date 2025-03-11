cmake_minimum_required(VERSION 3.10)
project(oottvm)

cmake_policy(SET CMP0077 NEW)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_STATIC_RUNTIME "Build with static runtime" OFF)
option(CMAKE_POSITION_INDEPENDENT_CODE "Build with position independent code" OFF)
option(CLANG_TIDY "Build with clang-tidy" OFF)

if(CLANG_TIDY)
  find_program(CLANG_TIDY_EXE clang-tidy)
  if(NOT CLANG_TIDY_EXE)
    message(FATAL_ERROR "clang-tidy not found")
  endif()
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};--config-file=${CMAKE_SOURCE_DIR}/.clang-tidy")
endif()

if(DEFINED ENV{TVM_LIBRARY_PATH})
  set(TVM_LIBRARY_PATH "$ENV{TVM_LIBRARY_PATH}")
  message(STATUS "Using TVM_LIBRARY_PATH from environment: ${TVM_LIBRARY_PATH}")
else()
  message(FATAL_ERROR "TVM_LIBRARY_PATH NOT SET, you should build TVM from source.")
endif()

if (DEFINED TVM_LIBRARY_PATH)
  get_filename_component(TVM_LIBRARY_ABS_PATH "${TVM_LIBRARY_PATH}" ABSOLUTE)
  message(STATUS "Using TVM_LIBRARY_PATH: ${TVM_LIBRARY_ABS_PATH}")
  add_library(tvm SHARED IMPORTED)
  set_target_properties(tvm PROPERTIES
    IMPORTED_LOCATION "${TVM_LIBRARY_ABS_PATH}/libtvm.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TVM_LIBRARY_ABS_PATH}/../include"
  )
  add_library(tvm_runtime SHARED IMPORTED)
  set_target_properties(tvm_runtime PROPERTIES
    IMPORTED_LOCATION "${TVM_LIBRARY_ABS_PATH}/libtvm_runtime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TVM_LIBRARY_ABS_PATH}/../include"
  )
else()
  message(FATAL_ERROR "TVM_LIBRARY_PATH NOT SET, you should build TVM from source.")
endif()

if (NOT DEFINED TVM_SOURCE_DIR)
  if(DEFINED ENV{TVM_SOURCE_DIR})
    set(TVM_SOURCE_DIR "$ENV{TVM_SOURCE_DIR}")
    message(STATUS "Using TVM_SOURCE_DIR from environment: ${TVM_SOURCE_DIR}")
  else()
    if (DEFINED TVM_LIBRARY_PATH)
      set(TVM_SOURCE_DIR "${TVM_LIBRARY_PATH}/..")
      message(STATUS "Using TVM_SOURCE_DIR derived from TVM_LIBRARY_PATH: ${TVM_SOURCE_DIR}")
    else()
      message(FATAL_ERROR "TVM_SOURCE_DIR is not defined, and TVM_LIBRARY_PATH is not set.")
    endif()
  endif()
endif()

# Step 0. Add some useful 3rdparty include directories
include_directories(${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include)
include_directories(${TVM_SOURCE_DIR}/3rdparty/dlpack/include)

# Step 1. Add your own include directory
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/tests/cpp/include)  # Add test include directory

# Step 2. Add your own source files
file(GLOB SOURCES
  ${CMAKE_SOURCE_DIR}/src/*.cc
)

# Step 3. Create a shared library
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Step 4. Link libraries for the shared library
target_link_libraries(${PROJECT_NAME} PRIVATE tvm)

# Step 5. Add your own test source files
file(GLOB TEST_SOURCES
  ${CMAKE_SOURCE_DIR}/tests/cpp/*.cc
  ${CMAKE_SOURCE_DIR}/tests/cpp/src/*.cc
)

# Step 6. Create an executable for the test
add_executable(${PROJECT_NAME}_test ${TEST_SOURCES})

# Step 7. Link libraries for the test executable
target_link_libraries(${PROJECT_NAME}_test PRIVATE ${PROJECT_NAME} tvm)
