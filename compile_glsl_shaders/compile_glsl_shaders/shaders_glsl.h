#ifndef _SHADERS_GLSL_H
#define _SHADERS_GLSL_H

#include <unordered_map>
#include <string>
#include <vector>

namespace shaders_glsl {

std::unordered_map<std::string, std::vector<uint8_t>> createGlslShaders();

}

#endif // _SHADERS_GLSL_H
