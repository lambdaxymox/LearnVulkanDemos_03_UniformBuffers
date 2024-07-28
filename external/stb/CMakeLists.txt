cmake_minimum_required(VERSION 3.28)
project(stb LANGUAGES CXX)

add_library(stb SHARED)
target_sources(stb PRIVATE stb/stb_image.cpp)
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
