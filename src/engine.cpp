#include "engine.h"


using VulkanInstanceProperties = VulkanEngine::VulkanInstanceProperties;

VulkanInstanceProperties::VulkanInstanceProperties(std::vector<VkLayerProperties> availableLayers, std::vector<VkExtensionProperties> availableExtensions) 
    : m_availableLayers { availableLayers }
    , m_availableExtensions { availableExtensions }
{
    for (const auto& layerProperties : availableLayers) {
        if (strcmp(layerProperties.layerName, Constants::VK_LAYER_KHRONOS_validation.data()) == 0) {
            m_validationLayersAvailable = true;
        }
    }

    for (const auto& extensionProperties : availableExtensions) {
        if (strcmp(extensionProperties.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
            m_debugUtilsAvailable = true;
        }
    }
}

bool VulkanInstanceProperties::isExtensionAvailable(const char* layerName) const {
    for (const auto& layerProperties : m_availableLayers) {
        if (strcmp(layerProperties.layerName, layerName) == 0) {
            return true;
        }
    }

    return false;
}

bool VulkanInstanceProperties::isLayerAvailable(const char* extensionName) const {
    for (const auto& extensionProperties : m_availableExtensions) {
        if (strcmp(extensionProperties.extensionName, extensionName) == 0) {
            return true;
        }
    }

    return false;
}

const std::vector<VkLayerProperties>& VulkanInstanceProperties::getAvailableLayers() const {
    return m_availableLayers;
}

const std::vector<VkExtensionProperties>& VulkanInstanceProperties::getAvailableExtensions() const {
    return m_availableExtensions;
}

bool VulkanInstanceProperties::areValidationLayersAvailable() const {
    return m_validationLayersAvailable;
}
    
bool VulkanInstanceProperties::areDebugUtilsAvailable() const {
    return m_debugUtilsAvailable;
}


using PhysicalDeviceProperties = VulkanEngine::PhysicalDeviceProperties;

PhysicalDeviceProperties::PhysicalDeviceProperties(std::vector<VkExtensionProperties> deviceExtensions)
    : m_deviceExtensions { deviceExtensions } 
{
}

const std::vector<VkExtensionProperties>& PhysicalDeviceProperties::getExtensions() const {
    return m_deviceExtensions;
}


#ifndef GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif // GLFW_INCLUDE_VULKAN

using VulkanInstanceProperties = VulkanEngine::VulkanInstanceProperties;
using PhysicalDeviceProperties = VulkanEngine::PhysicalDeviceProperties;


VulkanInstanceProperties VulkanEngine::PlatformInfoProvider::getVulkanInstanceInfo() const {
    auto availableLayers = this->getAvailableVulkanInstanceLayers();
    auto availableExtensions = this->getAvailableVulkanInstanceExtensions();    
    auto instanceInfo = VulkanInstanceProperties { availableLayers, availableExtensions };

    return instanceInfo;
}

std::vector<std::string> VulkanEngine::PlatformInfoProvider::getWindowSystemInstanceExtensions() const {
    uint32_t requiredExtensionCount = 0;
    const char** requiredExtensionNames = glfwGetRequiredInstanceExtensions(&requiredExtensionCount);
    auto requiredExtensions = std::vector<std::string> {};
    for (int i = 0; i < requiredExtensionCount; i++) {
        requiredExtensions.emplace_back(std::string(requiredExtensionNames[i]));
    }

    return requiredExtensions;
}

PhysicalDeviceProperties VulkanEngine::PlatformInfoProvider::getAvailableVulkanDeviceExtensions(
    VkPhysicalDevice physicalDevice
) const {
    auto deviceExtensionProperties =  std::vector<VkExtensionProperties> {};
    uint32_t numInstanceExtensions = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &numInstanceExtensions, nullptr);
    if (numInstanceExtensions > 0) {
        deviceExtensionProperties.resize(numInstanceExtensions);
        vkEnumerateDeviceExtensionProperties(
            physicalDevice,
            nullptr, 
            &numInstanceExtensions, 
            deviceExtensionProperties.data()
        );
    }

    return PhysicalDeviceProperties(deviceExtensionProperties);
}

std::vector<VkLayerProperties> VulkanEngine::PlatformInfoProvider::getAvailableVulkanInstanceLayers() const {
    auto instanceLayerProperties = std::vector<VkLayerProperties> {};
    uint32_t numInstanceExtensions = 0;
    vkEnumerateInstanceLayerProperties(&numInstanceExtensions, nullptr);
    if (numInstanceExtensions > 0) {
        instanceLayerProperties.resize(numInstanceExtensions);
        vkEnumerateInstanceLayerProperties(
            &numInstanceExtensions, 
            instanceLayerProperties.data()
        );
    }

    return instanceLayerProperties;
}

std::vector<VkExtensionProperties> VulkanEngine::PlatformInfoProvider::getAvailableVulkanInstanceExtensions() const {
    auto instanceExtensionProperties = std::vector<VkExtensionProperties> {};
    uint32_t numInstanceExtensions = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &numInstanceExtensions, nullptr);
    if (numInstanceExtensions > 0) {
        instanceExtensionProperties.resize(numInstanceExtensions);
        vkEnumerateInstanceExtensionProperties(
            nullptr, 
            &numInstanceExtensions, 
            instanceExtensionProperties.data()
        );
    }

    return instanceExtensionProperties;
}

std::vector<std::string> VulkanEngine::PlatformInfoProvider::detectMissingInstanceExtensions(
    const VulkanInstanceProperties& instanceInfo,
    const std::vector<std::string>& instanceExtensions
) const {
    auto missingInstanceExtensions = std::vector<std::string> {};
    auto availableInstanceExtensions = instanceInfo.getAvailableExtensions();
    for (const auto& extensionName : instanceExtensions) {
        auto found = std::find_if(
            std::begin(availableInstanceExtensions),
            std::end(availableInstanceExtensions),
            [extensionName](const auto& extension) {
                return strcmp(extensionName.data(), extension.extensionName) == 0;
            }
        );
        auto extensionNotFound = (found == std::end(availableInstanceExtensions));
        if (extensionNotFound) {
            missingInstanceExtensions.emplace_back(extensionName);
        }
    }

    return missingInstanceExtensions;
}

std::vector<std::string> VulkanEngine::PlatformInfoProvider::detectMissingInstanceLayers(
    const VulkanInstanceProperties& instanceInfo,
    const std::vector<std::string>& instanceLayers
) const {
    auto missingInstanceLayers = std::vector<std::string> {};
    auto availableInstanceLayers = instanceInfo.getAvailableLayers();
    for (const auto& layerName : instanceLayers) {
        auto found = std::find_if(
            std::begin(availableInstanceLayers),
            std::end(availableInstanceLayers),
            [layerName](const auto& layer) {
                return strcmp(layerName.data(), layer.layerName) == 0;
            }
        );
        auto layerNotFound = (found == std::end(availableInstanceLayers));
        if (layerNotFound) {
            missingInstanceLayers.emplace_back(layerName);
        }
    }

    return missingInstanceLayers;
}

std::vector<std::string> VulkanEngine::PlatformInfoProvider::detectMissingRequiredDeviceExtensions(
    const PhysicalDeviceProperties& physicalDeviceProperties,
    const std::vector<std::string>& requiredExtensions
) const {
    auto missingExtensions = std::vector<std::string> {};
    for (const auto& requiredExtension : requiredExtensions) {
        auto found = std::find_if(
            std::begin(requiredExtensions),
            std::end(requiredExtensions),
            [requiredExtension](const auto& extension) {
            return strcmp(requiredExtension.data(), extension.data()) == 0;
            }
        );
        auto extensionNotFound = (found == std::end(requiredExtensions));
        if (extensionNotFound) {
            missingExtensions.emplace_back(requiredExtension);
        }
    }

    return missingExtensions;
}

bool VulkanEngine::PlatformInfoProvider::areValidationLayersSupported() const {
    auto instanceInfo = this->getVulkanInstanceInfo();

    return instanceInfo.areValidationLayersAvailable();
}


