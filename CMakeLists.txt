function(FindGLSLSourceFiles shaderFiles)
    file(GLOB_RECURSE GLSL_SOURCE_FILES
        "${PROJECT_SOURCE_DIR}/shaders/*.frag.glsl"
        "${PROJECT_SOURCE_DIR}/shaders/*.vert.glsl"
        "${PROJECT_SOURCE_DIR}/shaders/*.comp.glsl"
    )
    set(${shaderFiles} ${GLSL_SOURCE_FILES} PARENT_SCOPE)
endfunction()

function(BuildGLSLShaders shaderSourceFiles spirvBinaryFiles)
    foreach(GLSL_FILE IN LISTS shaderSourceFiles)
        message(STATUS "COMPILING SHADER")
        get_filename_component(FILE_NAME ${GLSL_FILE} NAME)
        set(SPIRV_OUTPUT_FILE "${PROJECT_SOURCE_DIR}/shaders/${FILE_NAME}.spv")
        message(STATUS "Shader Input: ${GLSL_FILE}")
        message(STATUS "Shader Output: ${SPIRV_OUTPUT_FILE}")
        add_custom_command(
            OUTPUT ${SPIRV_OUTPUT_FILE}
            COMMAND ${GLSL_VALIDATOR} -V ${GLSL_FILE} -o ${SPIRV_OUTPUT_FILE}
            DEPENDS ${GLSL_FILE}
        )
        list(APPEND SPIRV_OUTPUT_FILES ${SPIRV_OUTPUT_FILE})
    endforeach(GLSL_FILE)
    set(${spirvBinaryFiles} ${SPIRV_OUTPUT_FILES} PARENT_SCOPE)
endfunction()

function(FindHLSLSourceFiles shaderFiles)
    file(GLOB_RECURSE HLSL_SOURCE_FILES
        "${PROJECT_SOURCE_DIR}/shaders/*.frag.hlsl"
        "${PROJECT_SOURCE_DIR}/shaders/*.vert.hlsl"
        "${PROJECT_SOURCE_DIR}/shaders/*.comp.hlsl"
    )

    set(${shaderFiles} ${HLSL_SOURCE_FILES} PARENT_SCOPE)
endfunction()

function(BuildHLSLShaders shaderSourceFiles spirvBinaryFiles)
    foreach(HLSL_FILE IN LISTS shaderSourceFiles)
        message(STATUS "COMPILING SHADER")
        get_filename_component(FILE_NAME ${HLSL_FILE} NAME)
        set(SPIRV_OUTPUT_FILE "${PROJECT_SOURCE_DIR}/shaders/${FILE_NAME}.spv")
        message(STATUS "Shader Input: ${HLSL_FILE}")
        message(STATUS "Shader Output: ${SPIRV_OUTPUT_FILE}")

        ## Determine the shader type based on the file extension.
        get_filename_component(SHADER_TYPE ${HLSL_FILE} EXT)
        string(TOLOWER ${SHADER_TYPE} SHADER_TYPE)
        set(COMPILE_OPTIONS "")
        if (SHADER_TYPE STREQUAL ".vert.hlsl")
            set(COMPILE_OPTIONS "-T vs_6_1")
        elseif (SHADER_TYPE STREQUAL ".frag.hlsl")
            set(COMPILE_OPTIONS "-T ps_6_1")
        elseif (SHADER_TYPE STREQUAL ".comp.hlsl")
            set(COMPILE_OPTIONS "-T cs_6_1")
        endif()

        add_custom_command(
            OUTPUT ${SPIRV_OUTPUT_FILE}
            COMMAND ${DX_SHADER_COMPILER} -spirv ${COMPILE_OPTIONS} -E main -Fo ${SPIRV_OUTPUT_FILE} ${HLSL_FILE}
            DEPENDS ${HLSL_FILE}
        )
        list(APPEND SPIRV_OUTPUT_FILES ${SPIRV_OUTPUT_FILE})
    endforeach(HLSL_FILE)
    set(${spirvBinaryFiles} ${SPIRV_OUTPUT_FILES} PARENT_SCOPE)
endfunction()

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
add_subdirectory(external/stb)
add_subdirectory(external/tiny_obj_loader)

# add_subdirectory(internal/vulkan_engine)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(LearnVulkanDemos_03_UniformBuffers)
target_sources(LearnVulkanDemos_03_UniformBuffers PRIVATE
    src/main.cpp
    src/engine.cpp
)
target_link_libraries(LearnVulkanDemos_03_UniformBuffers PRIVATE Vulkan::Vulkan)
# target_link_libraries(LearnVulkanDemos_03_UniformBuffers PRIVATE vulkan_engine)
target_link_libraries(LearnVulkanDemos_03_UniformBuffers PRIVATE glfw)
target_link_libraries(LearnVulkanDemos_03_UniformBuffers PRIVATE glm)
target_link_libraries(LearnVulkanDemos_03_UniformBuffers PRIVATE fmt)
target_link_libraries(LearnVulkanDemos_03_UniformBuffers PRIVATE stb)
target_link_libraries(LearnVulkanDemos_03_UniformBuffers PRIVATE tiny_obj_loader)
add_dependencies(LearnVulkanDemos_03_UniformBuffers GLSL_Shaders)
add_dependencies(LearnVulkanDemos_03_UniformBuffers HLSL_Shaders)

find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)
find_program(DX_SHADER_COMPILER dxc HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

FindGLSLSourceFiles(GLSL_SOURCE_FILES)
BuildGLSLShaders("${GLSL_SOURCE_FILES}" GLSL_SPIRV_BINARY_FILES)
FindHLSLSourceFiles(HLSL_SOURCE_FILES)
BuildHLSLShaders("${HLSL_SOURCE_FILES}" HLSL_SPIRV_BINARY_FILES)

add_custom_target(
    GLSL_Shaders 
    DEPENDS ${GLSL_SPIRV_BINARY_FILES}
)
add_custom_target(
    HLSL_Shaders
    DEPENDS ${HLSL_SPIRV_BINARY_FILES}
)
