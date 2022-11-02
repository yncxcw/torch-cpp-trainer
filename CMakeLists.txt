cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(torch_cpp_trainer VERSION 1.0.0 LANGUAGES CXX)

set(PRO_EXECUTABLE_NAME ${PROJECT_NAME}_main)
set(PRO_LIB_NAME ${PROJECT_NAME}_lib)

# Find torch on you system
# set(CMAKE_PREFIX_PATH "/home/wei/.local/lib/python3.8/site-packages/torch/share/cmake")
set(CMAKE_PREFIX_PATH "/usr/local/libtorch/share/cmake")
find_package(Torch REQUIRED)

# Add sources files
set(
    sources
    src/residual_block.cpp
    src/resnet.cpp
    src/trainer.cpp
    src/cifar10_dataset.cpp
)

set(
    sources_exec
    src/main.cpp
    ${sources}
)

# Add executable
add_executable(${PRO_EXECUTABLE_NAME} ${sources_exec})
target_include_directories(${PRO_EXECUTABLE_NAME} PUBLIC include)
target_link_libraries(${PRO_EXECUTABLE_NAME} ${TORCH_LIBRARIES})
set_target_properties(${PRO_EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# Add lib
add_library(${PRO_LIB_NAME} SHARED ${sources})
target_include_directories(${PRO_LIB_NAME} PUBLIC include)
target_link_libraries(${PRO_LIB_NAME} ${TORCH_LIBRARIES})
set_target_properties(${PRO_LIB_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# Add tests
enable_testing()
add_subdirectory(tests)