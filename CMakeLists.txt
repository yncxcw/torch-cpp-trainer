cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(torch-cpp-trainer VERSION 1.0.0 LANGUAGES CXX)

set(EXECUTABLE_NAME train)

# Find torch on you system
set(CMAKE_PREFIX_PATH "/home/wei/.local/lib/python3.8/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add source files
file(GLOB SRC_FILES "src/*.cpp")
# Add executable
add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE ${SRC_FILES})

target_include_directories(${EXECUTABLE_NAME} PRIVATE include)
target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# Add tests
enable_testing()
add_subdirectory(tests)