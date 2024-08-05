include(${CMAKE_CURRENT_LIST_DIR}/CompileGLSL_Lib.cmake)


function(CompileGLSL_Run inShaderInputPath inShaderOutputPath inShaderEmbeddingOutputName)
    CompileGLSL_FindCompilerCommands(GLSL_COMPILER_COMMAND)
    CompileGLSL_SetupBuildProcess("${inShaderOutputPath}")
    CompileGLSL_FindSourceFiles(
        "${inShaderInputPath}"
        GLSL_SOURCE_FILES
    )
    CompileGLSL_BuildShaders(
        "${GLSL_COMPILER_COMMAND}"
        "${GLSL_SOURCE_FILES}"
        "${inShaderOutputPath}"
        "${inShaderEmbeddingOutputName}"
        GLSL_SPIRV_BINARY_EMBEDDING_FILE
    )
endfunction()


if(${CMAKE_ARGC} LESS 6)
    message(FATAL_ERROR "Usage: cmake -P CompileGLSL_Run.cmake <inShaderInputPath> <inShaderOutputPath> <inShaderEmbeddingOutputName>")
endif()

set(inShaderInputPath "${CMAKE_ARGV3}")
set(inShaderOutputPath "${CMAKE_ARGV4}")
set(inShaderEmbeddingOutputName "${CMAKE_ARGV5}")

message(STATUS "inShaderInputPath = `${inShaderInputPath}`")
message(STATUS "inShaderOutputPath = `${inShaderOutputPath}`")
message(STATUS "inShaderEmbeddingOutputName = `${inShaderEmbeddingOutputName}`")

CompileGLSL_Run(${inShaderInputPath} ${inShaderOutputPath} ${inShaderEmbeddingOutputName})
