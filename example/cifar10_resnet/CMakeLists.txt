# Copyright ynjassionchen@gmail.com

add_executable(main train.cpp)
target_link_libraries(main ${PRO_LIB_NAME})
target_include_directories(main PUBLIC ${CMAKE_SOURCE_DIR}/include)