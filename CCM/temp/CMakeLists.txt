cmake_minimum_required(VERSION 3.9.6)
project(testPySpark)

set(CMAKE_CXX_STANDARD 11)

# add_executable(testPySpark ccm.h ccm.cpp main.cpp)
add_executable(testPySpark ccm.h ccm.cpp testPySpark.cpp)