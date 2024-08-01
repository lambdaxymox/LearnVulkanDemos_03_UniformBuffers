cmake_minimum_required(VERSION 3.28)

project("LearnVulkanDemos_03_UniformBuffers" VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(NoInSourceBuilds)
CheckNoInSourceBuilds()

find_package(Vulkan REQUIRED)

add_subdirectory(external/glfw-3.4)
add_subdirectory(external/glm-1.0.1)
add_subdirectory(external/fmt-10.2.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(LearnVulkanDemos_03_UniformBuffers)
target_sources(LearnVulkanDemos_03_UniformBuffers PRIVATE
    src/main.cpp
    src/engine.cpp
    src/engine_impl_fmt.cpp
)
target_link_libraries(LearnVulkanDemos_03_UniformBuffers PRIVATE Vulkan::Vulkan)
target_link_libraries(LearnVulkanDemos_03_UniformBuffers PRIVATE glfw)
target_link_libraries(LearnVulkanDemos_03_UniformBuffers PRIVATE glm)
target_link_libraries(LearnVulkanDemos_03_UniformBuffers PRIVATE fmt)
add_dependencies(LearnVulkanDemos_03_UniformBuffers GLSL_Shaders)
add_dependencies(LearnVulkanDemos_03_UniformBuffers HLSL_Shaders)

include(CompileGLSL)
include(CompileHLSL)

CompileGLSL_FindCompilerCommands(GLSL_COMPILER_COMMAND)
CompileGLSL_FindSourceFiles(
    "${PROJECT_SOURCE_DIR}/shaders"
    GLSL_SOURCE_FILES
)
CompileGLSL_CompileShaders(
    "${GLSL_COMPILER_COMMAND}"
    "${GLSL_SOURCE_FILES}"
    "${PROJECT_SOURCE_DIR}/shaders"
    GLSL_SPIRV_BINARY_FILES
)

CompileHLSL_FindCompilerCommands(HLSL_COMPILER_COMMAND)
CompileHLSL_FindSourceFiles(
    "${PROJECT_SOURCE_DIR}/shaders"
    HLSL_SOURCE_FILES
)
CompileHLSL_CompileShaders(
    "${HLSL_COMPILER_COMMAND}"
    "${HLSL_SOURCE_FILES}"
    "${PROJECT_SOURCE_DIR}/shaders"
    HLSL_SPIRV_BINARY_FILES
)

add_custom_target(
    GLSL_Shaders 
    DEPENDS ${GLSL_SPIRV_BINARY_FILES}
)
add_custom_target(
    HLSL_Shaders
    DEPENDS ${HLSL_SPIRV_BINARY_FILES}
)

add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E env $<TARGET_FILE:LearnVulkanDemos_03_UniformBuffers>
    DEPENDS "LearnVulkanDemos_03_UniformBuffers"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    VERBATIM
)
