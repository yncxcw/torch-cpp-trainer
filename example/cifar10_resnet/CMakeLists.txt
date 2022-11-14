# Copyright ynjassionchen@gmail.com

# Is this necessary ??? 
add_executable(main train.cpp)
target_link_libraries(main ${PRO_LIB_NAME})
message(${CMAKE_SOURCE_DIR})
target_include_directories(main PUBLIC ${CMAKE_SOURCE_DIR}/include)