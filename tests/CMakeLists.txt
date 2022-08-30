cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(${CMAKE_PROJECT_NAME}Tests VERSION 1.0.0 LANGUAGES CXX)


# Add test sources files
file(GLOB TEST_SOURCES "src/*.cpp")
# Load GTest
find_package(GTest REQUIRED)

foreach(file ${TEST_SOURCES})
    # Define each unitest executable
    string(REGEX REPLACE "(.*/)([a-zA-Z0-9_ ]+)(\.cpp)" test_name ${file}) 
    add_executable(${test_name}_Tests ${file})
    # Add include
    include(GoogleTest)
    # include(GoogleTest)
    # Add linking library
    target_link_libraries(
        ${test_name}_Tests
        GTest::gest_main
        pthread
        ${TORCH_LIBRARIES}
    )

    get_target_property(OUT ${test_name}_Tests LINK_LIBRARIES)
    message(STATUS ${OUT})
    add_test(
        NAME
            ${test_name}
        COMMAND
            ${test_name}_Tests        
    )


endforeach()
message("Finished add unit test for ${CMAKE_PROJECT_NAME}.")