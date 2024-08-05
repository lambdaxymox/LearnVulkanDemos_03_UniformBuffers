#include "shaders_glsl.h"

#include "shaders_glsl.h.in"


std::unordered_map<std::string, std::vector<uint8_t>> shaders_glsl::createGlslShaders() {
    const auto shaders = std::unordered_map<std::string, std::vector<uint8_t>> {
        { shader_frag_glsl, std::vector<uint8_t> { shader_frag_glsl_spv.begin(), shader_frag_glsl_spv.end() } },
        { shader_vert_glsl, std::vector<uint8_t> { shader_vert_glsl_spv.begin(), shader_vert_glsl_spv.end() } },
    };

    return shaders;
}

