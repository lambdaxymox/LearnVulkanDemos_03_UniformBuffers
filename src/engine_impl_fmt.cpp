#include "engine_impl_fmt.h"


using PhysicalDeviceProperties = VulkanEngine::PhysicalDeviceProperties;
using VulkanInstanceProperties = VulkanEngine::VulkanInstanceProperties;


template <> struct fmt::formatter<std::vector<std::string>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    };

    auto format(const std::vector<std::string>& vec, format_context& ctx) const -> format_context::iterator {
        auto appender = fmt::format_to(ctx.out(), "{}", "[");
        if (!vec.empty()) {
            auto lastItem = vec.end() - 1;
            for (auto item = vec.begin(); item != lastItem; item++) {
                fmt::format_to(appender, "\"{}\", ", *item);
            }
            fmt::format_to(appender, "\"{}\"", *lastItem);
        }
        fmt::format_to(appender, "{}", "]");

        return appender;
    }
};

template <> struct fmt::formatter<VkExtensionProperties> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    };

    auto format(const VkExtensionProperties& extensionProperties, format_context& ctx) const -> format_context::iterator {
        auto specVariant = VK_API_VERSION_VARIANT(extensionProperties.specVersion);
        auto specMajor = VK_API_VERSION_MAJOR(extensionProperties.specVersion);
        auto specMinor = VK_API_VERSION_MINOR(extensionProperties.specVersion);
        auto specPatch = VK_API_VERSION_PATCH(extensionProperties.specVersion);
        std::string_view extensionName { extensionProperties.extensionName, sizeof(extensionProperties.extensionName) };

        auto appender = fmt::format_to(ctx.out(), "{}", "VkExtensionProperties {{ ");
        fmt::format_to(appender, "extensionName: \"{}\", ", extensionName);
        fmt::format_to(appender, "specVersion: {}.{}.{}.{}", specVariant, specMajor, specMinor, specPatch);
        fmt::format_to(ctx.out(), "{}", " }}");
        
        return appender;
    }
};

template <> struct fmt::formatter<VkLayerProperties> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    };
    
    auto format(const VkLayerProperties& layerProperties, format_context& ctx) const -> format_context::iterator {
        auto specVariant = VK_API_VERSION_VARIANT(layerProperties.specVersion);
        auto specMajor = VK_API_VERSION_MAJOR(layerProperties.specVersion);
        auto specMinor = VK_API_VERSION_MINOR(layerProperties.specVersion);
        auto specPatch = VK_API_VERSION_PATCH(layerProperties.specVersion);
        auto implVariant = VK_API_VERSION_VARIANT(layerProperties.implementationVersion);
        auto implMajor = VK_API_VERSION_MAJOR(layerProperties.implementationVersion);
        auto implMinor = VK_API_VERSION_MINOR(layerProperties.implementationVersion);
        auto implPatch = VK_API_VERSION_PATCH(layerProperties.implementationVersion);
        std::string_view layerName { layerProperties.layerName, sizeof(layerProperties.layerName) };
        std::string_view description { layerProperties.description, sizeof(layerProperties.description) };

        auto appender = fmt::format_to(ctx.out(), "{}", "VkLayerProperties {{ ");
        fmt::format_to(appender, "layerName: \"{}\", ", layerName);
        fmt::format_to(appender, "specVersion: {}.{}.{}.{}, ", specVariant, specMajor, specMinor, specPatch);
        fmt::format_to(appender, "implementationVersion: {}.{}.{}.{}, ", implVariant, implMajor, implMinor, implPatch);
        fmt::format_to(appender, "description: \"{}\"", description);
        fmt::format_to(ctx.out(), "{}", " }}");

        return appender;
    }
};

template <> struct fmt::formatter<std::vector<VkExtensionProperties>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    };

    auto format(const std::vector<VkExtensionProperties>& vec, format_context& ctx) const -> format_context::iterator {
        auto appender = fmt::format_to(ctx.out(), "{}", "[");
        if (!vec.empty()) {
            auto lastItem = vec.end() - 1;
            for (auto item = vec.begin(); item != lastItem; item++) {
                fmt::format_to(appender, "\"{}\", ", *item);
            }
            fmt::format_to(appender, "\"{}\"", *lastItem);
        }
        fmt::format_to(appender, "{}", "]");

        return appender;
    }
};

template <> struct fmt::formatter<std::vector<VkLayerProperties>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    };

    auto format(const std::vector<VkLayerProperties>& vec, format_context& ctx) const -> format_context::iterator {
        auto appender = fmt::format_to(ctx.out(), "{}", "[");
        if (!vec.empty()) {
            auto lastItem = vec.end() - 1;
            for (auto item = vec.begin(); item != lastItem; item++) {
                fmt::format_to(appender, "\"{}\", ", *item);
            }
            fmt::format_to(appender, "\"{}\"", *lastItem);
        }
        fmt::format_to(appender, "{}", "]");

        return appender;
    }
};


auto fmt::formatter<PhysicalDeviceProperties>::format(const PhysicalDeviceProperties& deviceProperties, format_context& ctx) const -> format_context::iterator {
    return fmt::format_to(
        ctx.out(),
        "PhysicalDeviceProperties {{ deviceExtensions: {} }}",
        deviceProperties.getExtensions()
    );
}

auto fmt::formatter<VulkanInstanceProperties>::format(const VulkanInstanceProperties& platformInfo, format_context& ctx) const -> format_context::iterator {
    return fmt::format_to(
        ctx.out(),
        "VulkanInstanceProperties {{ availableLayers: {}, availableExtensions: {} }}",
        platformInfo.getAvailableLayers(), 
        platformInfo.getAvailableExtensions()
    );
}
