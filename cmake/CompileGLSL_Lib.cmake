function(CompileGLSL_GetShaderTypeExtension inShaderFileName outShaderType_extension)
    get_filename_component(fileExtension "${inShaderFileName}" EXT)
    string(REGEX REPLACE "^\\." "" fileExtension "${fileExtension}")
    string(REPLACE "." ";" shaderType_extension "${fileExtension}")
    list(FILTER shaderType_extension EXCLUDE REGEX "^$")

    set(${outShaderType_extension} "${shaderType_extension}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_GetShaderStage inShaderFile outShaderStage)
    get_filename_component(shaderFileName "${inShaderFile}" NAME)
    
    set(shaderType_extension)
    CompileGLSL_GetShaderTypeExtension("${shaderFileName}" shaderType_extension)
    
    list(LENGTH shaderType_extension partCount)
    if(NOT partCount EQUAL 2)
        message(FATAL_ERROR
            "Expected shader file name of the form `<shaderName>.<vert|frag|comp>.glsl`."
            "Got `${inShaderFileName}` from input `${inShaderFile}`."
        )
    endif()

    list(GET shaderType_extension 0 shaderType)
    set(shaderStage)
    if(${shaderType} STREQUAL "vert")
        set(shaderStage "vertex")
    elseif(${shaderType} STREQUAL "frag")
        set(shaderStage "fragment")
    elseif(${shaderType} STREQUAL "comp")
        set(shaderStage "compute")
    else()
        message(FATAL_ERROR "Expected a shader with shader type `vert`, `frag`, or `comp`. Got `${shaderType}`.")
    endif()

    set(${outShaderStage} "${shaderStage}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_GetCompileOptions GLSL_COMPILER_COMMAND inShaderSourceFile inSpirvBinaryFile inShaderStage outCompileOptions)
    set(glslCompilerName)
    get_filename_component(glslCompilerName "${GLSL_COMPILER_COMMAND}" NAME)

    set(compileOptions)
    if (${glslCompilerName} STREQUAL "glslc")
        list(APPEND compileOptions -fshader-stage=${inShaderStage} -o "${inSpirvBinaryFile}" "${inShaderSourceFile}")
    elseif (${glslCompilerName} STREQUAL "glslangValidator")
        list(APPEND compileOptions -V "${inShaderSourceFile}" -o "${inSpirvBinaryFile}")
    else()
        message(FATAL_ERROR
            "Unsupported compiler. The supported compilers are `glslc` and `glslangValidator`. "
            "Got `${GLSL_COMPILER_COMMAND}`."
        )
    endif()

    set(${outCompileOptions} "${compileOptions}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_CompileShader GLSL_COMPILER_COMMAND inShaderSourceFile inSpirvBinaryOutputPath outSpirvBinaryOutputFile)
    get_filename_component(fileName "${inShaderSourceFile}" NAME)
    set(spirvOutputFile "${inSpirvBinaryOutputPath}/${fileName}.spv")

    set(shaderStage)
    CompileGLSL_GetShaderStage("${inShaderSourceFile}" shaderStage)

    set(compileOptions)
    CompileGLSL_GetCompileOptions("${GLSL_COMPILER_COMMAND}" "${inShaderSourceFile}" "${spirvOutputFile}" "${shaderStage}" compileOptions)

    list(APPEND fullCommand "${GLSL_COMPILER_COMMAND}" "${compileOptions}")

    message(STATUS "Compiling shader")
    message(STATUS "Input: `${fileName}`")
    message(STATUS "Output: `${spirvOutputFile}`")
    message(STATUS "Full compiler command: `${fullCommand}`")
    execute_process(COMMAND ${fullCommand} RESULT_VARIABLE result)
    if(result EQUAL 0)
        message(STATUS "Compiling shader complete")
    else()
        message(FATAL_ERROR "Shader compilation failed with result `${result}`")
    endif()

    set(${outSpirvBinaryOutputFile} "${spirvOutputFile}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_CompileShaders GLSL_COMPILER_COMMAND inShaderSourceFiles inSpirvBinaryOutputPath outSpirvBinaryFiles)
    list(APPEND spirvBinaryFiles)
    foreach(shaderSourceFile IN LISTS inShaderSourceFiles)
        get_filename_component(fileName "${shaderSourceFile}" NAME)
        set(spirvOutputFile "${inSpirvBinaryOutputPath}/${fileName}.spv")

        set(spirvOutputFile)
        CompileGLSL_CompileShader("${GLSL_COMPILER_COMMAND}" "${shaderSourceFile}" "${inSpirvBinaryOutputPath}" spirvOutputFile)

        list(APPEND spirvBinaryFiles "${spirvOutputFile}")
    endforeach()

    set(${outSpirvBinaryFiles} "${spirvBinaryFiles}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_ReadSpirvBinaries inSpirvBinaryFiles outSpirvBinaries)
    list(APPEND spirvBinaries)
    foreach(spirvBinaryFile IN LISTS inSpirvBinaryFiles)
        get_filename_component(fileName "${spirvBinaryFile}" NAME)
        
        file(READ "${spirvBinaryFile}" spirvBinary HEX)
        
        list(APPEND spirvBinaries "${spirvBinary}")
    endforeach()

    set(${outSpirvBinaries} "${spirvBinaries}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_EmbedShader_GenerateEmbeddingName inSpirvBinaryFileName outSpirvBinaryEmbeddingName)
    string(APPEND spirvBinaryEmbeddingName "")
    string(REPLACE "." "_" spirvBinaryEmbeddingName "${inSpirvBinaryFileName}")

    set(${outSpirvBinaryEmbeddingName} "${spirvBinaryEmbeddingName}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_FormatSpirvBinaryToOctets inInputFile inOctetsPerLine outFormattedOctets outOctetsWritten)
    set(formattedData "")
    set(currentLine "")
    set(count 0)
    set(octetsWritten 0)

    # Process each byte
    string(LENGTH "${inInputFile}" binaryLength)
    math(EXPR lastIndex "${binaryLength} - 1")
    foreach(index RANGE 0 ${lastIndex} 2)
        string(SUBSTRING "${inInputFile}" ${index} 2 byte)
        string(APPEND currentLine "0x${byte}")
        math(EXPR count "${count} + 1")
        math(EXPR octetsWritten "${octetsWritten} + 1")

        # Add a comma if this is not the last line of the file.
        if(NOT index EQUAL ${lastIndex})
            string(APPEND currentLine ",")
        endif()

        # Start a new line if we've reached inOctetsPerLine.
        if(count EQUAL inOctetsPerLine)
            string(APPEND formattedData "${currentLine}\n")
            set(currentLine "")
            set(count 0)
        else()
            string(APPEND currentLine " ")
        endif()
    endforeach()

    if(NOT currentLine STREQUAL "")
        string(APPEND formattedData "${currentLine}\n")
    endif()
    
    set(${outFormattedOctets} "${formattedData}" PARENT_SCOPE)
    set(${outOctetsWritten} "${octetsWritten}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_EmbedShader inShaderFileName inSpirvBinaryFileName inSpirvBinaryFileData outSpirvBinaryEmbeddingData)
    string(APPEND shaderName "")
    CompileGLSL_EmbedShader_GenerateEmbeddingName("${inShaderFileName}" shaderName)

    string(APPEND arrayName "")
    CompileGLSL_EmbedShader_GenerateEmbeddingName("${inSpirvBinaryFileName}" arrayName)

    set(formattedOctets "")
    set(octetsWritten 0)
    CompileGLSL_FormatSpirvBinaryToOctets("${inSpirvBinaryFileData}" 12 formattedOctets octetsWritten)
    
    string(REGEX REPLACE "^0x" "    0x" formattedOctets "${formattedOctets}")
    string(REGEX REPLACE ",\n0x" ",\n    0x" formattedOctets "${formattedOctets}")
    
    math(EXPR arraySize "${octetsWritten}")

    string(APPEND combinedContent "// Shader: `${inShaderFileName}`\n")
    string(APPEND combinedContent "// SPIR-V Binary: `${inSpirvBinaryFileName}`\n")
    string(APPEND combinedContent "const std::string ${shaderName} = std::string { \"${inShaderFileName}\" };\n")
    string(APPEND combinedContent "const std::array<uint8_t, ${arraySize}> ${arrayName} = std::array<uint8_t, ${arraySize}> {")
    string(APPEND combinedContent "\n")
    string(APPEND combinedContent "${formattedOctets}")
    string(APPEND combinedContent "};")

    set(${outSpirvBinaryEmbeddingData} "${combinedContent}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_EmbedShaders inShaderSourceFiles inSpirvBinaryFiles inSpirvBinaries outSpirvBinaryEmbedding)
    set(spirvBinaryEmbedding)
    set(listLength 0)
    list(LENGTH inSpirvBinaries listLength)
    math(EXPR lastIndex "${listLength} - 1")
    foreach(index RANGE 0 ${lastIndex})
        set(shaderSourceFile)
        list(GET inShaderSourceFiles "${index}" shaderSourceFile)
        get_filename_component(shaderSourceFileName "${shaderSourceFile}" NAME)

        set(spirvBinaryFile)
        list(GET inSpirvBinaryFiles "${index}" spirvBinaryFile)
        get_filename_component(spirvBinaryFileName "${spirvBinaryFile}" NAME)

        string(APPEND spirvBinary "")
        list(GET inSpirvBinaries "${index}" spirvBinary)

        set(spirvBinaryEmbeddingFragment)
        CompileGLSL_EmbedShader("${shaderSourceFileName}" "${spirvBinaryFileName}" "${spirvBinary}" spirvBinaryEmbeddingFragment)

        string(APPEND spirvBinaryEmbedding "\n")
        string(APPEND spirvBinaryEmbedding "${spirvBinaryEmbeddingFragment}")
        string(APPEND spirvBinaryEmbedding "\n")
    endforeach()

    set(${outSpirvBinaryEmbedding} "${spirvBinaryEmbedding}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_GetIncludeGuard inFileName outIncludeGuardName)
    string(TOUPPER "${inFileName}" includeGuardName)
    string(REGEX REPLACE "[.-]" "_" includeGuardName "${includeGuardName}")
    string(PREPEND includeGuardName "_")

    set(${outIncludeGuardName} "${includeGuardName}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_CreateEmbeddingFile inSpirvEmbedding inFinalShaderEmbeddingOutputPath inFinalShaderEmbeddingFileName outFinalShaderEmbeddingFile)
    set(finalOutputFileName "${inFinalShaderEmbeddingFileName}.in")
    set(finalOutputFile "${inFinalShaderEmbeddingOutputPath}/${finalOutputFileName}")
    
    set(finalOutputIncludeGuardName)
    CompileGLSL_GetIncludeGuard("${finalOutputFileName}" finalOutputIncludeGuardName)

    string(APPEND combinedContent "")
    string(APPEND combinedContent "#ifndef ${finalOutputIncludeGuardName}\n")
    string(APPEND combinedContent "#define ${finalOutputIncludeGuardName}\n")
    string(APPEND combinedContent "\n")
    string(APPEND combinedContent "#include <array>\n")
    string(APPEND combinedContent "#include <string>\n")
    string(APPEND combinedContent "\n")
    string(APPEND combinedContent "${inSpirvEmbedding}")
    string(APPEND combinedContent "\n")
    string(APPEND combinedContent "#endif // ${finalOutputIncludeGuardName}")
    string(APPEND combinedContent "\n")

    file(WRITE "${finalOutputFile}" "${combinedContent}")

    set(${outFinalShaderEmbeddingFile} "${finalOutputFile}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_BuildShaders GLSL_COMPILER_COMMAND inShaderSourceFiles inFinalShaderEmbeddingOutputPath inFinalShaderEmbeddingFileName outFinalShaderEmbeddingFile)
    CompileGLSL_CompileShaders(
        "${GLSL_COMPILER_COMMAND}"
        "${inShaderSourceFiles}"
        "${inFinalShaderEmbeddingOutputPath}"
        spirvBinaryFiles
    )

    list(APPEND spirvBinaries)
    CompileGLSL_ReadSpirvBinaries("${spirvBinaryFiles}" spirvBinaries)

    string(APPEND spirvBinaryEmbedding "")
    CompileGLSL_EmbedShaders("${inShaderSourceFiles}" "${spirvBinaryFiles}" "${spirvBinaries}" spirvBinaryEmbedding)

    CompileGLSL_CreateEmbeddingFile(
        "${spirvBinaryEmbedding}"
        "${inFinalShaderEmbeddingOutputPath}"
        "${inFinalShaderEmbeddingFileName}"
        spirvBinaryEmbeddingFile
    )

    set(${outFinalShaderEmbeddingFile} "${spirvBinaryEmbeddingFile}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_GetSupportedCompilers outSupportedCompilers)
    list(APPEND supportedCompilers "glslc" "glslangValidator")
    set(${outSupportedCompilers} "${supportedCompilers}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_SetupBuildProcess inShaderOutputDir)
    file(MAKE_DIRECTORY "${inShaderOutputDir}")
endfunction()

function(CompileGLSL_FindCompilerCommands GLSL_COMPILER_COMMAND)
    CompileGLSL_GetSupportedCompilers(candidateGLSLCompilerCommands)
    find_program(_possibleGLSLCompilerCommand
        NAMES ${candidateGLSLCompilerCommands}
        PATHS /usr/bin
              /usr/local/bin
              $ENV{VULKAN_SDK}/Bin/
              $ENV{VULKAN_SDK}/Bin32/
    )

    if(${_possibleGLSLCompilerCommand} STREQUAL "_possibleGLSLCompilerCommand-NOTFOUND")
        message(FATAL_ERROR
            "One of the programs in `${candidateGLSLCompilerCommands}` could not be found. "
            "Make sure a GLSL compiler such as `glslc` or `glslangValidator` is installed on your system. This "
            "would typically come alongside either an OpenGL, Direct3D or Vulkan SDK. Otherwise, "
            "install it by another means. "
        )
    endif()

    set(${GLSL_COMPILER_COMMAND} "${_possibleGLSLCompilerCommand}" PARENT_SCOPE)
endfunction()

function(CompileGLSL_FindSourceFiles inShaderPath outFoundShaderFiles)
    file(GLOB_RECURSE GLSL_SOURCE_FILES
        "${inShaderPath}/*.frag.glsl"
        "${inShaderPath}/*.vert.glsl"
        "${inShaderPath}/*.comp.glsl"
    )

    set(${outFoundShaderFiles} "${GLSL_SOURCE_FILES}" PARENT_SCOPE)
endfunction()
