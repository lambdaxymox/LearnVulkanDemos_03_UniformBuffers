cmake_minimum_required(VERSION 3.28)
project(compile_glsl_shaders LANGUAGES CXX)

set(GLSL_SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../shaders")
set(GLSL_SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(GLSL_EMBEDDING_FILE_NAME "shaders_glsl.h")
set(GLSL_HEADER_DESTINATION "${GLSL_SHADER_BINARY_DIR}/${GLSL_EMBEDDING_FILE_NAME}.in")

file(MAKE_DIRECTORY "${GLSL_SHADER_BINARY_DIR}")


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
    OUTPUT "${GLSL_HEADER_DESTINATION}"
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CompileGLSL_Run.cmake" 
            "${GLSL_SHADER_SOURCE_DIR}" 
            "${GLSL_SHADER_BINARY_DIR}" 
            "${GLSL_EMBEDDING_FILE_NAME}"
    COMMENT "Compiling GLSL shaders"
    DEPENDS  always_rebuild
    VERBATIM
)

add_custom_target(GLSL_Shaders ALL
    DEPENDS
        "${GLSL_HEADER_DESTINATION}"
)

add_library(compile_glsl_shaders SHARED)
target_sources(compile_glsl_shaders
    PRIVATE
        compile_glsl_shaders/shaders_glsl.cpp
)
target_include_directories(compile_glsl_shaders 
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE
        "${GLSL_SHADER_BINARY_DIR}"
)
add_dependencies(compile_glsl_shaders GLSL_Shaders)
