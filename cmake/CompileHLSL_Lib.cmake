function(CompileHLSL_GetShaderTypeExtension inShaderFileName outShaderType_extension)
    get_filename_component(fileExtension "${inShaderFileName}" EXT)
    string(REGEX REPLACE "^\\." "" fileExtension "${fileExtension}")
    string(REPLACE "." ";" shaderType_extension "${fileExtension}")
    list(FILTER shaderType_extension EXCLUDE REGEX "^$")

    set(${outShaderType_extension} "${shaderType_extension}" PARENT_SCOPE)
endfunction()

function(CompileHLSL_GetShaderStage inShaderFile outShaderStage)
    get_filename_component(shaderFileName "${inShaderFile}" NAME)
    
    set(shaderType_extension)
    CompileHLSL_GetShaderTypeExtension("${shaderFileName}" shaderType_extension)
    
    list(LENGTH shaderType_extension partCount)
    if(NOT partCount EQUAL 2)
        message(FATAL_ERROR
            "Expected shader file name of the form `<shaderName>.<vert|frag|comp>.hlsl`."
            "Got `${inShaderFileName}` from input `${inShaderFile}`."
        )
    endif()

    list(GET shaderType_extension 0 shaderType)
    set(shaderStage)
    if(shaderType STREQUAL "vert")
        set(shaderStage "vs_6_1")
    elseif(shaderType STREQUAL "frag")
        set(shaderStage "ps_6_1")
    elseif(shaderType STREQUAL "comp")
        set(shaderStage "cs_6_1")
    else()
        message(FATAL_ERROR "Expected `inShaderType` to be one of `vert`, `frag`, or `comp`. Got `${shaderType}`.")
    endif()

    set(${outShaderStage} "${shaderStage}" PARENT_SCOPE)
endfunction()

function(CompileHLSL_GetCompileOptions HLSL_COMPILER_COMMAND inShaderSourceFile inSpirvBinaryFile inShaderStage outCompileOptions)
    set(hlslCompilerName)
    get_filename_component(hlslCompilerName "${HLSL_COMPILER_COMMAND}" NAME)

    set(compileOptions)
    if (${hlslCompilerName} STREQUAL "dxc")
        list(APPEND compileOptions -spirv -T ${shaderStage} -E main -Fo ${inSpirvBinaryFile} ${inShaderSourceFile})
    else()
        message(FATAL_ERROR
            "Unsupported compiler. The supported compilers are `dxc`. "
            "Got `${HLSL_COMPILER_COMMAND}`."
        )
    endif()

    set(${outCompileOptions} "${compileOptions}" PARENT_SCOPE)
endfunction()

function(CompileHLSL_CompileShader HLSL_COMPILER_COMMAND inShaderSourceFile inSpirvBinaryOutputPath outSpirvBinaryOutputFile)
    get_filename_component(fileName "${inShaderSourceFile}" NAME)
    set(spirvOutputFile "${inSpirvBinaryOutputPath}/${fileName}.spv")

    set(shaderStage)
    CompileHLSL_GetShaderStage("${inShaderSourceFile}" shaderStage)

    set(compileOptions)
    CompileHLSL_GetCompileOptions("${HLSL_COMPILER_COMMAND}" "${inShaderSourceFile}" "${spirvOutputFile}" "${shaderStage}" compileOptions)

    list(APPEND fullCommand "${HLSL_COMPILER_COMMAND}" "${compileOptions}")

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

function(CompileHLSL_CompileShaders HLSL_COMPILER_COMMAND inShaderSourceFiles inSpirvBinaryOutputPath outSpirvBinaryFiles)
    list(APPEND spirvBinaryFiles)
    foreach(shaderSourceFile IN LISTS inShaderSourceFiles)
        get_filename_component(fileName "${shaderSourceFile}" NAME)
        set(spirvOutputFile "${inSpirvBinaryOutputPath}/${fileName}.spv")

        set(spirvOutputFile)
        CompileHLSL_CompileShader("${HLSL_COMPILER_COMMAND}" "${shaderSourceFile}" "${inSpirvBinaryOutputPath}" spirvOutputFile)

        list(APPEND spirvBinaryFiles "${spirvOutputFile}")
    endforeach()

    set(${outSpirvBinaryFiles} "${spirvBinaryFiles}" PARENT_SCOPE)
endfunction()

function(CompileHLSL_ReadSpirvBinaries inSpirvBinaryFiles outSpirvBinaries)
    list(APPEND spirvBinaries)
    foreach(spirvBinaryFile IN LISTS inSpirvBinaryFiles)
        get_filename_component(fileName "${spirvBinaryFile}" NAME)
        
        file(READ "${spirvBinaryFile}" spirvBinary HEX)
        
        list(APPEND spirvBinaries "${spirvBinary}")
    endforeach()

    set(${outSpirvBinaries} "${spirvBinaries}" PARENT_SCOPE)
endfunction()

function(CompileHLSL_EmbedShader_GenerateEmbeddingName inSpirvBinaryFileName outSpirvBinaryEmbeddingName)
    string(APPEND spirvBinaryEmbeddingName "")
    string(REPLACE "." "_" spirvBinaryEmbeddingName "${inSpirvBinaryFileName}")

    set(${outSpirvBinaryEmbeddingName} "${spirvBinaryEmbeddingName}" PARENT_SCOPE)
endfunction()

function(CompileHLSL_FormatSpirvBinaryToOctets inInputFile inOctetsPerLine outFormattedOctets outOctetsWritten)
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

function(CompileHLSL_EmbedShader inShaderFileName inSpirvBinaryFileName inSpirvBinaryFileData outSpirvBinaryEmbeddingData)
    string(APPEND shaderName "")
    CompileHLSL_EmbedShader_GenerateEmbeddingName("${inShaderFileName}" shaderName)

    string(APPEND arrayName "")
    CompileHLSL_EmbedShader_GenerateEmbeddingName("${inSpirvBinaryFileName}" arrayName)

    set(formattedOctets "")
    set(octetsWritten 0)
    CompileHLSL_FormatSpirvBinaryToOctets("${inSpirvBinaryFileData}" 12 formattedOctets octetsWritten)
    
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

function(CompileHLSL_EmbedShaders inShaderSourceFiles inSpirvBinaryFiles inSpirvBinaries outSpirvBinaryEmbedding)
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
        CompileHLSL_EmbedShader("${shaderSourceFileName}" "${spirvBinaryFileName}" "${spirvBinary}" spirvBinaryEmbeddingFragment)

        string(APPEND spirvBinaryEmbedding "\n")
        string(APPEND spirvBinaryEmbedding "${spirvBinaryEmbeddingFragment}")
        string(APPEND spirvBinaryEmbedding "\n")
    endforeach()

    set(${outSpirvBinaryEmbedding} "${spirvBinaryEmbedding}" PARENT_SCOPE)
endfunction()

function(CompileHLSL_GetIncludeGuard inFileName outIncludeGuardName)
    string(TOUPPER "${inFileName}" includeGuardName)
    string(REGEX REPLACE "[.-]" "_" includeGuardName "${includeGuardName}")
    string(PREPEND includeGuardName "_")

    set(${outIncludeGuardName} "${includeGuardName}" PARENT_SCOPE)
endfunction()

function(CompileHLSL_CreateEmbeddingFile inSpirvEmbedding inFinalShaderEmbeddingOutputPath inFinalShaderEmbeddingFileName outFinalShaderEmbeddingFile)
    set(finalOutputFileName "${inFinalShaderEmbeddingFileName}.in")
    set(finalOutputFile "${inFinalShaderEmbeddingOutputPath}/${finalOutputFileName}")
    
    set(finalOutputIncludeGuardName)
    CompileHLSL_GetIncludeGuard("${finalOutputFileName}" finalOutputIncludeGuardName)

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

function(CompileHLSL_BuildShaders HLSL_COMPILER_COMMAND inShaderSourceFiles inFinalShaderEmbeddingOutputPath inFinalShaderEmbeddingFileName outFinalShaderEmbeddingFile)
    CompileHLSL_CompileShaders(
        "${HLSL_COMPILER_COMMAND}"
        "${inShaderSourceFiles}"
        "${inFinalShaderEmbeddingOutputPath}"
        spirvBinaryFiles
    )

    list(APPEND spirvBinaries)
    CompileHLSL_ReadSpirvBinaries("${spirvBinaryFiles}" spirvBinaries)

    string(APPEND spirvBinaryEmbedding "")
    CompileHLSL_EmbedShaders("${inShaderSourceFiles}" "${spirvBinaryFiles}" "${spirvBinaries}" spirvBinaryEmbedding)

    CompileHLSL_CreateEmbeddingFile(
        "${spirvBinaryEmbedding}"
        "${inFinalShaderEmbeddingOutputPath}"
        "${inFinalShaderEmbeddingFileName}"
        spirvBinaryEmbeddingFile
    )

    set(${outFinalShaderEmbeddingFile} "${spirvBinaryEmbeddingFile}" PARENT_SCOPE)
endfunction()

function(CompileHLSL_SetupBuildProcess inShaderOutputDir)
    file(MAKE_DIRECTORY "${inShaderOutputDir}")
endfunction()

function(CompileHLSL_GetSupportedCompilers outSupportedCompilers)
    list(APPEND supportedCompilers "dxc")
    set(${outSupportedCompilers} ${supportedCompilers} PARENT_SCOPE)
endfunction()

function(CompileHLSL_FindCompilerCommands HLSL_COMPILER_COMMAND)
    CompileHLSL_GetSupportedCompilers(candidateHLSLCompilerCommands)
    find_program(_possibleHLSLCompilerCommand
        NAMES ${candidateHLSLCompilerCommands}
        PATHS /usr/bin
              /usr/local/bin
              $ENV{VULKAN_SDK}/Bin/
              $ENV{VULKAN_SDK}/Bin32/
    )

    if(${_possibleHLSLCompilerCommand} STREQUAL "_possbileHLSLCompilerCommand-NOTFOUND")
        message(FATAL_ERROR
            "One of the programs in `${candidateHLSLCompilerCommands}` could not be found. "
            "Make sure a HLSL compiler such as `dxc` is installed on your system. This "
            "would typically come alongside either Direct3D or Vulkan SDK. Otherwise, "
            "install it by another means. "
        )
    endif()

    set(${HLSL_COMPILER_COMMAND} ${_possibleHLSLCompilerCommand} PARENT_SCOPE)
endfunction()

function(CompileHLSL_FindSourceFiles inShaderPath outFoundShaderFiles)
    file(GLOB_RECURSE HLSL_SOURCE_FILES
        "${inShaderPath}/*.frag.hlsl"
        "${inShaderPath}/*.vert.hlsl"
        "${inShaderPath}/*.comp.hlsl"
    )

    set(${outFoundShaderFiles} ${HLSL_SOURCE_FILES} PARENT_SCOPE)
endfunction()
