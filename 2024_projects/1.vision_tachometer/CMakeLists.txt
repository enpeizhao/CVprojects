cmake_minimum_required(VERSION 3.10)

project(HelloWorld)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV found")
    message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

add_executable(HelloWorld main.cpp)
target_link_libraries(HelloWorld ${OpenCV_LIBS})
target_include_directories(HelloWorld PRIVATE ${OpenCV_INCLUDE_DIRS})
