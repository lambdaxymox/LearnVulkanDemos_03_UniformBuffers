cmake_minimum_required(VERSION 3.28)
project(compile_hlsl_shaders LANGUAGES CXX)

set(HLSL_SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../shaders")
set(HLSL_SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(HLSL_EMBEDDING_FILE_NAME "shaders_hlsl.h")
set(HLSL_HEADER_DESTINATION "${HLSL_SHADER_BINARY_DIR}/${HLSL_EMBEDDING_FILE_NAME}.in")

file(MAKE_DIRECTORY "${HLSL_SHADER_BINARY_DIR}")


#[[
The `always_rebuild` command produces no output. This target ensures
that any command that depends on it is always considered out of date.
In particular, we define a target that always triggers a rebuild.

Since `always_rebuild` does not generate any output files, CMake treats it 
as always out of date, since CMake has no output from the target to check.
]]
add_custom_command(
    OUTPUT always_rebuild
    COMMAND cmake -E echo
    COMMENT "Force shaders to rebuild every build step."
)

add_custom_command(
    OUTPUT "${HLSL_HEADER_DESTINATION}"
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CompileHLSL_Run.cmake" 
            "${HLSL_SHADER_SOURCE_DIR}" 
            "${HLSL_SHADER_BINARY_DIR}" 
            "${HLSL_EMBEDDING_FILE_NAME}"
    COMMENT "Compiling HLSL shaders"
    DEPENDS  always_rebuild
    VERBATIM
)

add_custom_target(HLSL_Shaders ALL
    DEPENDS
        "${HLSL_HEADER_DESTINATION}"
)

add_library(compile_hlsl_shaders SHARED)
target_sources(compile_hlsl_shaders
    PRIVATE
        compile_hlsl_shaders/shaders_hlsl.cpp
)
target_include_directories(compile_hlsl_shaders 
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE
        "${HLSL_SHADER_BINARY_DIR}"
)
add_dependencies(compile_hlsl_shaders HLSL_Shaders)
