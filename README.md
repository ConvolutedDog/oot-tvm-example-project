# Out-of-tree TVM Example Project

This project demonstrates how to build an **out-of-tree TVM example project** using CMake.
It provides a minimal example of how to link against the TVM library and use
its headers in an external project.

## Prerequisites

Before building this project, ensure you have the following installed:

- **TVM**: Build TVM from source and set the environment variables `TVM_LIBRARY_PATH`.
- **CMake**: Version 3.10 or higher.
- **C++ Compiler**: Supports C++17.

## Environment Variables

This project relies on the following environment variables:

- `TVM_LIBRARY_PATH`: Path to the directory containing `libtvm.so` and `libtvm_runtime.so`.

  ```bash
  export TVM_LIBRARY_PATH=/path/to/tvm/build
  ```

## Building the Project and Run the Example

- **Clone the repository**:

  ```bash
  git clone https://github.com/ConvolutedDog/oot-tvm-example.git
  cd oot-tvm-example
  ```

- **Create a build directory and run CMake**:

  ```bash
  mkdir build && cd build && cmake ..
  ```

- **Build the project**:

  ```bash
  make -j
  ```

- **Run the example**:

  ```bash
  ./oottvm_test
  ```

## Project Structure

- `CMakeLists.txt:` The main CMake configuration file. You can follow the
example steps in this file to add your own source files and dependencies.
- `src/`: Example source file demonstrating how to use TVM in an out-of-tree project.
- `include/`: Directory for your project's header files (if any).
- `tests/`: Example test files.

## Customizing the Project

- **Step 1. Add some useful 3rdparty include directories**
  ```cmake
  include_directories(${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include)
  include_directories(${TVM_SOURCE_DIR}/3rdparty/dlpack/include)
  ```

- **Step 2. Add your own include directory**
  ```cmake
  include_directories(${CMAKE_SOURCE_DIR}/include)
  # Add test include directory
  include_directories(${CMAKE_SOURCE_DIR}/tests/cpp/include)
  ```

- **Step 3. Add your own source files**
  ```cmake
  file(GLOB SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.cc
  )
  ```

- **Step 4. Create a shared library**
  ```cmake
  add_library(${PROJECT_NAME} SHARED ${SOURCES})
  ```

- **Step 5. Link libraries for the shared library**
  ```cmake
  target_link_libraries(${PROJECT_NAME} PRIVATE tvm)
  ```

- **Step 6. Add your own test source files**
  ```cmake
  file(GLOB_RECURSE TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/tests/cpp/*.cc
  )
  ```

- **Step 7. Create an executable for the test**
  ```cmake
  add_executable(${PROJECT_NAME}_test ${TEST_SOURCES})
  ```

- **Step 8. Link libraries for the test executable**
  ```cmake
  target_link_libraries(${PROJECT_NAME}_test PRIVATE ${PROJECT_NAME} tvm)
  ```

## Troubleshooting

**Library not found**: Ensure `TVM_LIBRARY_PATH` points to the directory
containing `libtvm.so` and `libtvm_runtime.so`.
