#ifndef _SHADERS_HLSL_H
#define _SHADERS_HLSL_H

#include <unordered_map>
#include <string>
#include <vector>

namespace shaders_hlsl {

std::unordered_map<std::string, std::vector<uint8_t>> createHlslShaders();

}

#endif // _SHADERS_HLSL_H
