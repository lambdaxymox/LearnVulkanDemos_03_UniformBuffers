#ifndef _ENGINE_IMPL_FMT_H
#define _ENGINE_IMPL_FMT_H

#include "engine.h"

#include <fmt/core.h>


template <> struct fmt::formatter<VulkanEngine::PhysicalDeviceProperties>: fmt::formatter<string_view> {
    auto format(const VulkanEngine::PhysicalDeviceProperties& deviceProperties, format_context& ctx) const -> format_context::iterator;
};

template <> struct fmt::formatter<VulkanEngine::VulkanInstanceProperties>: fmt::formatter<string_view> {
    auto format(const VulkanEngine::VulkanInstanceProperties& platformInfo, format_context& ctx) const -> format_context::iterator;
};

#endif // _ENGINE_IMPL_FMT_H
