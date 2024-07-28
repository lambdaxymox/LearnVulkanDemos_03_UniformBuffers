cmake_minimum_required(VERSION 3.28)
project(tiny_obj_loader LANGUAGES CXX)

add_library(tiny_obj_loader SHARED)
target_sources(tiny_obj_loader PRIVATE tiny_obj_loader/tiny_obj_loader.cpp)
target_include_directories(tiny_obj_loader INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
