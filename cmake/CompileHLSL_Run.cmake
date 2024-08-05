include(${CMAKE_CURRENT_LIST_DIR}/CompileHLSL_Lib.cmake)


function(CompileHLSL_Run inShaderInputPath inShaderOutputPath inShaderEmbeddingOutputName)
    CompileHLSL_FindCompilerCommands(GLSL_COMPILER_COMMAND)
    CompileHLSL_SetupBuildProcess("${inShaderOutputPath}")
    CompileHLSL_FindSourceFiles(
        "${inShaderInputPath}"
        GLSL_SOURCE_FILES
    )
    CompileHLSL_BuildShaders(
        "${GLSL_COMPILER_COMMAND}"
        "${GLSL_SOURCE_FILES}"
        "${inShaderOutputPath}"
        "${inShaderEmbeddingOutputName}"
        GLSL_SPIRV_BINARY_EMBEDDING_FILE
    )
endfunction()


if(${CMAKE_ARGC} LESS 6)
    message(FATAL_ERROR "Usage: cmake -P CompileHLSL_Run.cmake <inShaderInputPath> <inShaderOutputPath> <inShaderEmbeddingOutputName>")
endif()

set(inShaderInputPath "${CMAKE_ARGV3}")
set(inShaderOutputPath "${CMAKE_ARGV4}")
set(inShaderEmbeddingOutputName "${CMAKE_ARGV5}")

message(STATUS "inShaderInputPath = `${inShaderInputPath}`")
message(STATUS "inShaderOutputPath = `${inShaderOutputPath}`")
message(STATUS "inShaderEmbeddingOutputName = `${inShaderEmbeddingOutputName}`")

CompileHLSL_Run(${inShaderInputPath} ${inShaderOutputPath} ${inShaderEmbeddingOutputName})
