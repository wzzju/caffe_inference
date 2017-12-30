cmake_minimum_required(VERSION 3.8)
project(caffe_inference)

set(CMAKE_CXX_STANDARD 11)

set(SOURCES_DIR ${CMAKE_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_library(z-lib
        NAMES z)

find_package(OpenCV REQUIRED)

add_executable(caffe_inference
        ${SOURCES_DIR}/layers/conv_layer.cpp
        ${SOURCES_DIR}/layers/fc_layer.cpp
        ${SOURCES_DIR}/layers/pooling_layer.cpp
        ${SOURCES_DIR}/layers/relu_layer.cpp
        ${SOURCES_DIR}/layers/softmax_layer.cpp
        ${SOURCES_DIR}/cnpy.cpp
        ${SOURCES_DIR}/utilities.cpp
        ${SOURCES_DIR}/net.cpp
        ${SOURCES_DIR}/main.cpp)

target_link_libraries(caffe_inference ${z-lib} ${OpenCV_LIBS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)