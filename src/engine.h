#ifndef _ENGINE_H
#define _ENGINE_H

#include <vulkan/vulkan.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <optional>
#include <set>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <random>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <unordered_set>

#ifndef GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif // GLFW_INCLUDE_VULKAN

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <stb/stb_image.h>
#include <tiny_obj_loader/tiny_obj_loader.h>


#ifdef NDEBUG
const bool ENABLE_VALIDATION_LAYERS = false;
const bool ENABLE_DEBUGGING_EXTENSIONS = false;
#else
const bool ENABLE_VALIDATION_LAYERS = true;
const bool ENABLE_DEBUGGING_EXTENSIONS = true;
#endif


namespace VulkanEngine {

namespace Constants {

    const std::string VK_LAYER_KHRONOS_validation = std::string { "VK_LAYER_KHRONOS_validation" };
    const std::string VK_KHR_portability_subset = std::string { "VK_KHR_portability_subset" };
    const std::vector<std::string> VALIDATION_LAYERS = std::vector<std::string> { 
        VK_LAYER_KHRONOS_validation
    };
} // namespace Contants

class VulkanInstanceProperties final {
    public:
        explicit VulkanInstanceProperties(std::vector<VkLayerProperties> availableLayers, std::vector<VkExtensionProperties> availableExtensions);
    
        bool isExtensionAvailable(const char* layerName) const;

        bool isLayerAvailable(const char* extensionName) const;

        const std::vector<VkLayerProperties>& getAvailableLayers() const;

        const std::vector<VkExtensionProperties>& getAvailableExtensions() const;

        bool areValidationLayersAvailable() const;
    
        bool areDebugUtilsAvailable() const;
    private:
        std::vector<VkLayerProperties> m_availableLayers;
        std::vector<VkExtensionProperties> m_availableExtensions;
        bool m_validationLayersAvailable = false;
	    bool m_debugUtilsAvailable = false;
};

class PhysicalDeviceProperties final {
    public:
        explicit PhysicalDeviceProperties(std::vector<VkExtensionProperties> deviceExtensions);

        const std::vector<VkExtensionProperties>& getExtensions() const;
    private:
        std::vector<VkExtensionProperties> m_deviceExtensions;
};

class PlatformInfoProvider {
    public:
        enum class Platform {
            Apple,
            Linux,
            Windows,
            Unknown
    }   ;

        explicit PlatformInfoProvider() = default;
        ~PlatformInfoProvider() = default;

        VulkanInstanceProperties getVulkanInstanceInfo() const;

        std::vector<std::string> getWindowSystemInstanceExtensions() const;

        std::vector<std::string> detectMissingRequiredDeviceExtensions(
            const PhysicalDeviceProperties& physicalDeviceProperties,
            const std::vector<std::string>& requiredExtensions
        ) const;

        Platform detectOperatingSystem() const {
            #if defined(__APPLE__) || defined(__MACH__)
            return Platform::Apple;
            #elif defined(__LINUX__)
            return Platform::Linux;
            #elif defined(_WIN32)
            return Platform::Windows;
            #else
            return Platform::Unknown;
            #endif
        }

        std::vector<VkExtensionProperties> getAvailableVulkanInstanceExtensions() const;

        std::vector<VkLayerProperties> getAvailableVulkanInstanceLayers() const;

        std::vector<std::string> detectMissingInstanceExtensions(
            const VulkanInstanceProperties& instanceInfo,
            const std::vector<std::string>& instanceExtensions
        ) const;

        std::vector<std::string> detectMissingInstanceLayers(
            const VulkanInstanceProperties& instanceInfo,
            const std::vector<std::string>& instanceExtensions
        ) const;

        PhysicalDeviceProperties getAvailableVulkanDeviceExtensions(VkPhysicalDevice physicalDevice) const;

        bool areValidationLayersSupported() const;
};

struct QueueFamilyIndices final {
    std::optional<uint32_t> graphicsAndComputeFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const {
        return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails final {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class VulkanInstanceSpec final {
    public:
        explicit VulkanInstanceSpec() = default;
        explicit VulkanInstanceSpec(
            std::vector<std::string> instanceExtensions,
            std::vector<std::string> instanceLayers,
            VkInstanceCreateFlags instanceCreateFlags,
            std::string applicationName,
            std::string engineName
        ) 
            : m_instanceExtensions { instanceExtensions }
            , m_instanceLayers { instanceLayers }
            , m_instanceCreateFlags { instanceCreateFlags }
            , m_applicationName { applicationName }
            , m_engineName { engineName }
        {
        }
        ~VulkanInstanceSpec() = default;

        const std::vector<std::string>& instanceExtensions() const {
            return m_instanceExtensions;
        }

        const std::vector<std::string>& instanceLayers() const {
            return m_instanceLayers;
        }

        VkInstanceCreateFlags instanceCreateFlags() const {
            return m_instanceCreateFlags;
        }

        const std::string& applicationName() const {
            return m_applicationName;
        }

        const std::string& engineName() const {
            return m_engineName;
        }

        bool areValidationLayersEnabled() const {
            auto found = std::find(
                m_instanceLayers.begin(), 
                m_instanceLayers.end(), 
                VulkanEngine::Constants::VK_LAYER_KHRONOS_validation
            );
        
            return found != m_instanceLayers.end();
        }
    private:
        std::vector<std::string> m_instanceExtensions;
        std::vector<std::string> m_instanceLayers;
        VkInstanceCreateFlags m_instanceCreateFlags;
        std::string m_applicationName;
        std::string m_engineName;
};

class InstanceSpecProvider final {
    public:
        explicit InstanceSpecProvider() = default;
        explicit InstanceSpecProvider(bool enableValidationLayers, bool enableDebuggingExtensions)
            : m_enableValidationLayers { enableValidationLayers }
            , m_enableDebuggingExtensions { enableDebuggingExtensions }
        {
        }

        ~InstanceSpecProvider() {
            m_enableValidationLayers = false;
            m_enableDebuggingExtensions = false;
        }

        VulkanInstanceSpec createInstanceSpec() const {
            const auto instanceExtensions = this->getInstanceExtensions();
            const auto instanceLayers = this->getInstanceLayers();
            const auto instanceCreateFlags = this->minInstanceCreateFlags();
            const auto applicationName = std::string { "" };
            const auto engineName = std::string { "" };

            return VulkanInstanceSpec {
                instanceExtensions,
                instanceLayers,
                instanceCreateFlags,
                applicationName,
                engineName
            };
        }
    private:
        bool m_enableValidationLayers;
        bool m_enableDebuggingExtensions;

        enum class Platform {
            Apple,
            Linux,
            Windows,
            Unknown
        };

        Platform getOperatingSystem() const {
            #if defined(__APPLE__) || defined(__MACH__)
            return Platform::Apple;
            #elif defined(__LINUX__)
            return Platform::Linux;
            #elif defined(_WIN32)
            return Platform::Windows;
            #else
            return Platform::Unknown;
            #endif
        }

        VkInstanceCreateFlags minInstanceCreateFlags() const {
            auto instanceCreateFlags = 0;
            if (this->getOperatingSystem() == Platform::Apple) {
                instanceCreateFlags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            }

            return instanceCreateFlags;
        }

        std::vector<std::string> getWindowSystemInstanceRequirements() const {        
            uint32_t requiredExtensionCount = 0;
            const char** requiredExtensionNames = glfwGetRequiredInstanceExtensions(&requiredExtensionCount);
            auto requiredExtensions = std::vector<std::string> {};
            for (int i = 0; i < requiredExtensionCount; i++) {
                requiredExtensions.emplace_back(std::string(requiredExtensionNames[i]));
            }

            return requiredExtensions;
        }

        std::vector<std::string> getInstanceExtensions() const {
            auto instanceExtensions = this->getWindowSystemInstanceRequirements();
            instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            instanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
            if (m_enableDebuggingExtensions) {
                instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }

            return instanceExtensions;
        }

        std::vector<std::string> getInstanceLayers() const {
            auto instanceLayers = std::vector<std::string> {};
            if (m_enableValidationLayers) {
                instanceLayers.push_back(VulkanEngine::Constants::VK_LAYER_KHRONOS_validation);
            }

            return instanceLayers;
        }
};

class SystemFactory final {
    public:
        explicit SystemFactory() = default;

        VkInstance create(const VulkanInstanceSpec& instanceSpec) {
            if (instanceSpec.areValidationLayersEnabled() && !m_infoProvider->areValidationLayersSupported()) {
                throw std::runtime_error("validation layers requested, but not available!");
            }

            const auto instanceInfo = m_infoProvider->getVulkanInstanceInfo();
            const auto instanceExtensions = instanceSpec.instanceExtensions();
            const auto instanceLayers = instanceSpec.instanceLayers();
            const auto missingExtensions = m_infoProvider->detectMissingInstanceExtensions(
                instanceInfo,
                instanceExtensions
            );
            const auto missingLayers = m_infoProvider->detectMissingInstanceLayers(
                instanceInfo,
                instanceLayers
            );
            if (missingExtensions.empty() && !missingLayers.empty()) {
                auto errorMessage = std::string { "Vulkan does not have the required extensions on this system:\n" };
                for (const auto& extensionName : missingExtensions) {
                    errorMessage.append(extensionName);
                    errorMessage.append("\n");
                }

                throw std::runtime_error { errorMessage };
            }

            if (!missingExtensions.empty() && missingLayers.empty()) {
                auto errorMessage = std::string { "Vulkan does not have the required layers on this system:\n" };
                for (const auto& layerName : missingLayers) {
                    errorMessage.append(layerName);
                    errorMessage.append("\n");
                }

                throw std::runtime_error { errorMessage };
            }

            if (!missingExtensions.empty() && !missingLayers.empty()) {
                auto errorMessage = std::string { "Vulkan does not have the required extensions on this system:\n" };
                for (const auto& extensionName : missingExtensions) {
                    errorMessage.append(extensionName);
                    errorMessage.append("\n");
                }

                errorMessage.append(std::string { "Vulkan does not have the required layers on this system:\n" });
                for (const auto& layerName : missingLayers) {
                    errorMessage.append(layerName);
                    errorMessage.append("\n");
                }

                throw std::runtime_error { errorMessage };
            }

            const auto instanceCreateFlags = instanceSpec.instanceCreateFlags();
            const auto enabledLayerNames = SystemFactory::convertToCStrings(instanceLayers);
            const auto enabledExtensionNames = SystemFactory::convertToCStrings(instanceExtensions);

            const auto appInfo = VkApplicationInfo {
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pApplicationName = instanceSpec.applicationName().data(),
                .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                .pEngineName = instanceSpec.engineName().data(),
                .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                .apiVersion = VK_API_VERSION_1_3
            };
            const auto flags = (VkInstanceCreateFlags {}) | instanceCreateFlags;
            const auto createInfo = VkInstanceCreateInfo {
                .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                .pApplicationInfo = &appInfo,
                .flags = flags,
                .enabledLayerCount = static_cast<uint32_t>(enabledLayerNames.size()),
                .ppEnabledLayerNames = enabledLayerNames.data(),
                .enabledExtensionCount = static_cast<uint32_t>(enabledExtensionNames.size()),
                .ppEnabledExtensionNames = enabledExtensionNames.data(),
            };

            auto instance = VkInstance {};
            const auto result = vkCreateInstance(&createInfo, nullptr, &instance);
            if (result != VK_SUCCESS) {
                throw std::runtime_error(fmt::format("Failed to create Vulkan instance."));
            }

            return instance;
        }
    private:
        PlatformInfoProvider* m_infoProvider;

        static std::vector<const char*> convertToCStrings(const std::vector<std::string>& strings) {
            auto cStrings = std::vector<const char*> {};
            cStrings.reserve(strings.size());
            std::transform(
                strings.begin(), 
                strings.end(), 
                std::back_inserter(cStrings),
                [](const std::string& string) { return string.c_str(); }
            );
    
            return cStrings;
        }
};

class PhysicalDeviceSpec final {
    public:
        explicit PhysicalDeviceSpec() = default;
        explicit PhysicalDeviceSpec(const std::vector<std::string>& requiredExtensions, bool hasGraphicsFamily, bool hasPresentFamily)
            : m_requiredExtensions { requiredExtensions }
            , m_hasGraphicsFamily { hasGraphicsFamily }
            , m_hasPresentFamily { hasPresentFamily }
        {
        }
        ~PhysicalDeviceSpec() = default;

        const std::vector<std::string>& requiredExtensions() const {
            return m_requiredExtensions;
        }

        bool hasGraphicsFamily() const {
            return m_hasGraphicsFamily;
        }

        bool hasPresentFamily() const {
            return m_hasPresentFamily;
        }
    private:
        std::vector<std::string> m_requiredExtensions;
        bool m_hasGraphicsFamily;
        bool m_hasPresentFamily;
};

class PhysicalDeviceSpecProvider final {
    public:
        explicit PhysicalDeviceSpecProvider() = default;

        PhysicalDeviceSpec createPhysicalDeviceSpec() const {
            const auto requiredExtensions = this->getPhysicalDeviceRequirements();

            return PhysicalDeviceSpec { requiredExtensions, true, true };
        }
    private:
        enum class Platform {
            Apple,
            Linux,
            Windows,
            Unknown
        };

        Platform getOperatingSystem() const {
            #if defined(__APPLE__) || defined(__MACH__)
            return Platform::Apple;
            #elif defined(__LINUX__)
            return Platform::Linux;
            #elif defined(_WIN32)
            return Platform::Windows;
            #else
            return Platform::Unknown;
            #endif
        }

        std::vector<std::string> getPhysicalDeviceRequirements() const {
            auto physicalDeviceExtensions = std::vector<std::string> {};
            if (this->getOperatingSystem() == Platform::Apple) {
                physicalDeviceExtensions.push_back(VulkanEngine::Constants::VK_KHR_portability_subset);
            }

            physicalDeviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

            return physicalDeviceExtensions;
        }
};

class PhysicalDeviceSelector final {
    public:
        explicit PhysicalDeviceSelector(VkInstance instance, PlatformInfoProvider* infoProvider)
            : m_instance { instance }
            , m_infoProvider { infoProvider }
        {
        }

        ~PhysicalDeviceSelector() {
            m_instance = VK_NULL_HANDLE;
            m_infoProvider = nullptr;
        }

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

            auto queueFamilies = std::vector<VkQueueFamilyProperties> { queueFamilyCount };
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

            int i = 0;
            auto indices = QueueFamilyIndices {};
            for (const auto& queueFamily : queueFamilies) {
                if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                    indices.graphicsAndComputeFamily = i;
                }

                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);

                if (presentSupport) {
                    indices.presentFamily = i;
                }

                if (indices.isComplete()) {
                    break;
                }

                i++;
            }

            return indices;
        }

        bool checkDeviceExtensionSupport(VkPhysicalDevice physicalDevice, const std::vector<std::string>& requiredExtensions) const {
            uint32_t extensionCount = 0;
            vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);

            auto availableExtensions = std::vector<VkExtensionProperties> { extensionCount };
            vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

            auto remainingExtensions = std::set<std::string> { requiredExtensions.begin(), requiredExtensions.end() };
            for (const auto& extension : availableExtensions) {
                for (const auto& requiredExtension : requiredExtensions) {
                    remainingExtensions.erase(extension.extensionName);
                }
            }

            return remainingExtensions.empty();
        }

        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
            auto details = SwapChainSupportDetails {};
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &details.capabilities);

            uint32_t formatCount = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);

            if (formatCount != 0) {
                details.formats.resize(formatCount);
                vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, details.formats.data());
            }

            uint32_t presentModeCount = 0;
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);

            if (presentModeCount != 0) {
                details.presentModes.resize(presentModeCount);
                vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, details.presentModes.data());
            }

            return details;
        }

        bool isPhysicalDeviceCompatible(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, const PhysicalDeviceSpec& physicalDeviceSpec) const {
            const auto indices = this->findQueueFamilies(physicalDevice, surface);
            const bool areRequiredExtensionsSupported = this->checkDeviceExtensionSupport(
                physicalDevice,
                physicalDeviceSpec.requiredExtensions()
            );

            bool swapChainCompatible = false;
            if (areRequiredExtensionsSupported) {
                SwapChainSupportDetails swapChainSupport = this->querySwapChainSupport(physicalDevice, surface);
                swapChainCompatible = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
            }

            auto supportedFeatures = VkPhysicalDeviceFeatures {};
            vkGetPhysicalDeviceFeatures(physicalDevice, &supportedFeatures);

            return indices.isComplete() && areRequiredExtensionsSupported && swapChainCompatible && supportedFeatures.samplerAnisotropy;
        }

        std::vector<VkPhysicalDevice> findAllPhysicalDevices() const {
            uint32_t physicalDeviceCount = 0;
            vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, nullptr);

            auto physicalDevices = std::vector<VkPhysicalDevice> { physicalDeviceCount };
            vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, physicalDevices.data());

            return physicalDevices;
        }

        std::vector<VkPhysicalDevice> findCompatiblePhysicalDevices(VkSurfaceKHR surface, const PhysicalDeviceSpec& physicalDeviceSpec) const {
            const auto physicalDevices = this->findAllPhysicalDevices();
            if (physicalDevices.empty()) {
                throw std::runtime_error("failed to find GPUs with Vulkan support!");
            }

            auto compatiblePhysicalDevices = std::vector<VkPhysicalDevice> {};
            for (const auto& physicalDevice : physicalDevices) {
                if (this->isPhysicalDeviceCompatible(physicalDevice, surface, physicalDeviceSpec)) {
                    compatiblePhysicalDevices.emplace_back(physicalDevice);
                }
            }

            return compatiblePhysicalDevices;
        }

        VkPhysicalDevice selectPhysicalDeviceForSurface(VkSurfaceKHR surface, const PhysicalDeviceSpec& physicalDeviceSpec) const {
            const auto physicalDevices = this->findCompatiblePhysicalDevices(surface, physicalDeviceSpec);
            if (physicalDevices.empty()) {
                throw std::runtime_error("failed to find a suitable GPU!");
            }

            VkPhysicalDevice selectedPhysicalDevice = physicalDevices[0];

            return selectedPhysicalDevice;
        }
    private:
        VkInstance m_instance;
        PlatformInfoProvider* m_infoProvider;
};

class LogicalDeviceSpec final {
    public:
        explicit LogicalDeviceSpec() = default;
        explicit LogicalDeviceSpec(const std::vector<std::string>& requiredExtensions, bool requireSamplerAnisotropy)
            : m_requiredExtensions { requiredExtensions }
            , m_requireSamplerAnisotropy { requireSamplerAnisotropy }
        {
        }
        ~LogicalDeviceSpec() = default;

        const std::vector<std::string>& requiredExtensions() const {
            return m_requiredExtensions;
        }

        bool requireSamplerAnisotropy() const {
            return m_requireSamplerAnisotropy;
        }
    private:
        std::vector<std::string> m_requiredExtensions;
        bool m_requireSamplerAnisotropy;
};

class LogicalDeviceSpecProvider final {
    public:
        explicit LogicalDeviceSpecProvider() = default;
        explicit LogicalDeviceSpecProvider(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) 
            : m_physicalDevice { physicalDevice }
            , m_surface { surface }
        {
        }

        ~LogicalDeviceSpecProvider() {
            m_physicalDevice = VK_NULL_HANDLE;
            m_surface = VK_NULL_HANDLE;
        }

        LogicalDeviceSpec createLogicalDeviceSpec() const {
            auto requiredExtensions = this->getLogicalDeviceRequirements();
            bool requireSamplerAnisotropy = true;

            return LogicalDeviceSpec { requiredExtensions, requireSamplerAnisotropy };
        }
    private:
        VkPhysicalDevice m_physicalDevice;
        VkSurfaceKHR m_surface;

        enum class Platform {
            Apple,
            Linux,
            Windows,
            Unknown
        };

        Platform getOperatingSystem() const {
            #if defined(__APPLE__) || defined(__MACH__)
            return Platform::Apple;
            #elif defined(__LINUX__)
            return Platform::Linux;
            #elif defined(_WIN32)
            return Platform::Windows;
            #else
            return Platform::Unknown;
            #endif
        }

        std::vector<std::string> getLogicalDeviceRequirements() const {
            auto logicalDeviceExtensions = std::vector<std::string> {};
            if (this->getOperatingSystem() == Platform::Apple) {
                logicalDeviceExtensions.push_back(VulkanEngine::Constants::VK_KHR_portability_subset);
            }

            logicalDeviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

            return logicalDeviceExtensions;
        }
};

class LogicalDeviceFactory final {
    public:
        explicit LogicalDeviceFactory(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, PlatformInfoProvider* infoProvider)
            : m_physicalDevice { physicalDevice }
            , m_surface { surface }
            , m_infoProvider { infoProvider }
        {
        }

        ~LogicalDeviceFactory() {
            m_physicalDevice = VK_NULL_HANDLE;
            m_surface = VK_NULL_HANDLE;
            m_infoProvider = nullptr;
        }

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
            auto indices = QueueFamilyIndices {};

            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

            auto queueFamilies = std::vector<VkQueueFamilyProperties> { queueFamilyCount };
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

            int i = 0;
            for (const auto& queueFamily : queueFamilies) {
                if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                    indices.graphicsAndComputeFamily = i;
                }

                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);

                if (presentSupport) {
                    indices.presentFamily = i;
                }

                if (indices.isComplete()) {
                    break;
                }

                i++;
            }

            return indices;
        }

        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
            auto details = SwapChainSupportDetails {};
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &details.capabilities);

            uint32_t formatCount = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);

            if (formatCount != 0) {
                details.formats.resize(formatCount);
                vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, details.formats.data());
            }

            uint32_t presentModeCount = 0;
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);

            if (presentModeCount != 0) {
                details.presentModes.resize(presentModeCount);
                vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, details.presentModes.data());
            }

            return details;
        }

        std::tuple<VkDevice, VkQueue, VkQueue, VkQueue> createLogicalDevice(const LogicalDeviceSpec& logicalDeviceSpec) {
            const auto indices = this->findQueueFamilies(m_physicalDevice, m_surface);
            const auto uniqueQueueFamilies = std::set<uint32_t> {
                indices.graphicsAndComputeFamily.value(), 
                indices.presentFamily.value()
            };
            const float queuePriority = 1.0f;
            auto queueCreateInfos = std::vector<VkDeviceQueueCreateInfo> {};
            for (uint32_t queueFamily : uniqueQueueFamilies) {
                const auto queueCreateInfo = VkDeviceQueueCreateInfo {
                    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    .queueFamilyIndex = queueFamily,
                    .queueCount = 1,
                    .pQueuePriorities = &queuePriority,
                };

                queueCreateInfos.push_back(queueCreateInfo);
            }

            const auto requireSamplerAnisotropy = [&logicalDeviceSpec]() -> VkBool32 { 
                if (logicalDeviceSpec.requireSamplerAnisotropy()) {    
                    return VK_TRUE;
                } else {
                    return VK_FALSE;
                }
            }();
            const auto deviceExtensionProperties = m_infoProvider->getAvailableVulkanDeviceExtensions(m_physicalDevice);
            const auto missingExtensions = m_infoProvider->detectMissingRequiredDeviceExtensions(
                deviceExtensionProperties, 
                logicalDeviceSpec.requiredExtensions()
            );
            if (!missingExtensions.empty()) {
                auto errorMessage = std::string { "Vulkan does not have the required extensions on this system: " };
                for (const auto& extension : missingExtensions) {
                    errorMessage.append(extension);
                    errorMessage.append("\n");
                }

                throw std::runtime_error(errorMessage);
            }
            const auto enabledExtensions = LogicalDeviceFactory::convertToCStrings(logicalDeviceSpec.requiredExtensions());
            const auto validationLayersCStrings = []() {
                if (ENABLE_VALIDATION_LAYERS) {
                    return LogicalDeviceFactory::convertToCStrings(Constants::VALIDATION_LAYERS);
                } else {
                    return std::vector<const char*> {};
                }
            }();

            const auto deviceFeatures = VkPhysicalDeviceFeatures {
                .samplerAnisotropy = requireSamplerAnisotropy,
            };

            const auto createInfo = VkDeviceCreateInfo {
                .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
                .pQueueCreateInfos = queueCreateInfos.data(),
                .pEnabledFeatures = &deviceFeatures,
                .enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size()),
                .ppEnabledExtensionNames = enabledExtensions.data(),
                .enabledLayerCount = static_cast<uint32_t>(validationLayersCStrings.size()),
                .ppEnabledLayerNames = validationLayersCStrings.data(),
            };

            auto device = VkDevice {};
            const auto result = vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &device);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create logical device!");
            }

            auto graphicsQueue = VkQueue {};
            vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0, &graphicsQueue);

            auto computeQueue = VkQueue {};
            vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0, &computeQueue);
        
            auto presentQueue = VkQueue {};
            vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

            return std::make_tuple(device, graphicsQueue, computeQueue, presentQueue);
        }
    private:
        VkPhysicalDevice m_physicalDevice;
        VkSurfaceKHR m_surface;
        PlatformInfoProvider* m_infoProvider;

        static std::vector<const char*> convertToCStrings(const std::vector<std::string>& strings) {
            auto cStrings = std::vector<const char*> {};
            cStrings.reserve(strings.size());
            std::transform(
                strings.begin(), 
                strings.end(), 
                std::back_inserter(cStrings),
                [](const std::string& string) { return string.c_str(); }
            );
    
            return cStrings;
        }
};

class VulkanDebugMessenger final {
    public:
        explicit VulkanDebugMessenger()
            : m_instance { VK_NULL_HANDLE }
            , m_debugMessenger { VK_NULL_HANDLE }
        {
        }

        ~VulkanDebugMessenger() {
            this->cleanup();
        }

        static VulkanDebugMessenger* create(VkInstance instance) {
            if (instance == VK_NULL_HANDLE) {
                throw std::invalid_argument { "Got an empty `VkInstance` handle" };
            }

            uint32_t physicalDeviceCount = 0;
            const auto resultEnumeratePhysicalDevices = vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
            if (resultEnumeratePhysicalDevices != VK_SUCCESS) {
                throw std::invalid_argument { "Got an invalid `VkInstance` handle" };
            }

            const auto createInfo = VkDebugUtilsMessengerCreateInfoEXT {
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                .messageSeverity = 
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                .messageType = 
                    VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                .pfnUserCallback = debugCallback,
            };

            auto debugMessenger = static_cast<VkDebugUtilsMessengerEXT>(nullptr);
            const auto result = VulkanDebugMessenger::CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger);
            if (result != VK_SUCCESS) {
                throw std::runtime_error { "failed to set up debug messenger!" };
            }

            auto vulkanDebugMessenger = new VulkanDebugMessenger {};
            vulkanDebugMessenger->m_instance = instance;
            vulkanDebugMessenger->m_debugMessenger = debugMessenger;

            return vulkanDebugMessenger;
        }

        static VkResult CreateDebugUtilsMessengerEXT(
            VkInstance instance, 
            const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, 
            const VkAllocationCallbacks* pAllocator, 
            VkDebugUtilsMessengerEXT* pDebugMessenger
        ) {
            const auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT")
            );

            if (func != nullptr) {
                return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
            } else {
                return VK_ERROR_EXTENSION_NOT_PRESENT;
            }
        }

        static void DestroyDebugUtilsMessengerEXT(
            VkInstance instance, 
            VkDebugUtilsMessengerEXT debugMessenger, 
            const VkAllocationCallbacks* pAllocator
        ) {
            const auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT")
            );

            if (func != nullptr) {
                func(instance, debugMessenger, pAllocator);
            }
        }

        static const std::string& messageSeverityToString(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity) {
            static const std::string MESSAGE_SEVERITY_INFO  = std::string { "INFO " };
            static const std::string MESSAGE_SEVERITY_WARN  = std::string { "WARN " };
            static const std::string MESSAGE_SEVERITY_ERROR = std::string { "ERROR" };

            if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
                return MESSAGE_SEVERITY_ERROR;
            } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
                return MESSAGE_SEVERITY_WARN;
            } else {
                return MESSAGE_SEVERITY_INFO;
            }
        }

        static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData
        ) {
            const auto messageSeverityString = VulkanDebugMessenger::messageSeverityToString(messageSeverity);
            fmt::println(std::cerr, "[{}] {}", messageSeverityString, pCallbackData->pMessage);

            return VK_FALSE;
        }

        void cleanup() {
            if (m_instance == VK_NULL_HANDLE) {
                return;
            }

            if (m_debugMessenger != VK_NULL_HANDLE) {
                VulkanDebugMessenger::DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
            }

            m_debugMessenger = VK_NULL_HANDLE;
            m_instance = VK_NULL_HANDLE;
        }
    private:
        VkInstance m_instance;
        VkDebugUtilsMessengerEXT m_debugMessenger;
};

class SurfaceProvider final {
    public:
        explicit SurfaceProvider() = delete;
        explicit SurfaceProvider(VkInstance instance, GLFWwindow* window)
            : m_instance { instance }
            , m_window { window }
        {
        }

        ~SurfaceProvider() {
            m_instance = VK_NULL_HANDLE;
            m_window = nullptr;
        }

        VkSurfaceKHR createSurface() {
            auto surface = VkSurfaceKHR {};
            const auto result = glfwCreateWindowSurface(m_instance, m_window, nullptr, &surface);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create window surface!");
            }

            return surface;
        }
    private:
        VkInstance m_instance;
        GLFWwindow* m_window;
};

class WindowSystem final {
    public:
        explicit WindowSystem() = default;
        explicit WindowSystem(VkInstance instance) : m_instance { instance } {}
        ~WindowSystem() {
            m_framebufferResized = false;
            m_windowExtent = VkExtent2D { 0, 0 };
            m_surface = VK_NULL_HANDLE;

            glfwDestroyWindow(m_window);

            m_instance = VK_NULL_HANDLE;
        }

        static WindowSystem* create(VkInstance instance) {
            return new WindowSystem { instance };
        }

        void createWindow(uint32_t width, uint32_t height, const std::string& title) {
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

            auto window = glfwCreateWindow(width, height, title.data(), nullptr, nullptr);
            glfwSetWindowUserPointer(window, this);
            glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

            m_window = window;
            m_windowExtent = VkExtent2D { width, height };
        }

        GLFWwindow* getWindow() const {
            return m_window;
        }

        SurfaceProvider createSurfaceProvider() {
            return SurfaceProvider { m_instance, m_window};
        }

        bool hasFramebufferResized() const {
            return m_framebufferResized;
        }

        void setFramebufferResized(bool framebufferResized) {
            m_framebufferResized = framebufferResized;
        }

        void setWindowTitle(const std::string& title) {
            glfwSetWindowTitle(m_window, title.data());
        }
    private:
        VkInstance m_instance;
        GLFWwindow* m_window;
        VkSurfaceKHR m_surface;
        VkExtent2D m_windowExtent;
        bool m_framebufferResized;

        static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
            auto windowSystem = reinterpret_cast<WindowSystem*>(glfwGetWindowUserPointer(window));
            windowSystem->m_framebufferResized = true;
            windowSystem->m_windowExtent = VkExtent2D { 
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };
        }
};

class GpuDevice final {
    public:
        explicit GpuDevice() = delete;
        explicit GpuDevice(
            VkInstance instance,
            VkPhysicalDevice physicalDevice, 
            VkDevice device, 
            VkQueue graphicsQueue,
            VkQueue computeQueue,
            VkQueue presentQueue,
            VkCommandPool commandPool
        )   : m_instance { instance }
            , m_physicalDevice { physicalDevice }
            , m_device { device }
            , m_graphicsQueue { graphicsQueue }
            , m_computeQueue { computeQueue }
            , m_presentQueue { presentQueue }
            , m_commandPool { commandPool }
            , m_shaderModules { std::unordered_set<VkShaderModule> {} }
        {
            m_msaaSamples = GpuDevice::getMaxUsableSampleCount(physicalDevice);
        }

        ~GpuDevice() {
            for (const auto& shaderModule : m_shaderModules) {
                vkDestroyShaderModule(m_device, shaderModule, nullptr);
            }

            vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
            vkDestroyCommandPool(m_device, m_commandPool, nullptr);
            vkDestroyDevice(m_device, nullptr);

            m_surface = VK_NULL_HANDLE;
            m_commandPool = VK_NULL_HANDLE;
            m_presentQueue = VK_NULL_HANDLE;
            m_graphicsQueue = VK_NULL_HANDLE;
            m_device = VK_NULL_HANDLE;
            m_physicalDevice = VK_NULL_HANDLE;
            m_instance = VK_NULL_HANDLE;
        }

        VkPhysicalDevice getPhysicalDevice() const {
            return m_physicalDevice;
        }

        VkDevice getLogicalDevice() const {
            return m_device;
        }

        VkQueue getGraphicsQueue() const {
            return m_graphicsQueue;
        }

        VkQueue getComputeQueue() const {
            return m_computeQueue;
        }

        VkQueue getPresentQueue() const {
            return m_presentQueue;
        }

        VkCommandPool getCommandPool() const {
            return m_commandPool;
        }

        VkSampleCountFlagBits getMsaaSamples() const {
            return m_msaaSamples;
        }

        static VkSampleCountFlagBits getMaxUsableSampleCount(VkPhysicalDevice physicalDevice) {
            auto physicalDeviceProperties = VkPhysicalDeviceProperties {};
            vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

            const auto sampleCounts = VkSampleCountFlags { 
                physicalDeviceProperties.limits.framebufferColorSampleCounts & 
                physicalDeviceProperties.limits.framebufferDepthSampleCounts
            };
            if (sampleCounts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
            if (sampleCounts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
            if (sampleCounts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
            if (sampleCounts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
            if (sampleCounts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
            if (sampleCounts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

            return VK_SAMPLE_COUNT_1_BIT;
        }

        VkSurfaceKHR createRenderSurface(SurfaceProvider& surfaceProvider) {
            const auto surface = surfaceProvider.createSurface();

            m_surface = surface;

            return surface;
        }

        VkShaderModule createShaderModuleFromFile(const std::string& fileName) {
            const auto shaderCode = this->loadShaderFromFile(fileName);

            return this->createShaderModule(shaderCode);
        }

        VkShaderModule createShaderModule(std::istream& stream) {
            const auto shaderCode = this->loadShader(stream);

            return this->createShaderModule(shaderCode);
        }

        VkShaderModule createShaderModule(const std::vector<char>& code) {
            const auto createInfo = VkShaderModuleCreateInfo {
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = code.size(),
                .pCode = reinterpret_cast<const uint32_t*>(code.data()),
                .pNext = nullptr,
                .flags = 0,
            };

            auto shaderModule = VkShaderModule {};
            const auto result = vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create shader module!");
            }

            m_shaderModules.insert(shaderModule);

            return shaderModule;
        }
    private:
        VkInstance m_instance;
        VkPhysicalDevice m_physicalDevice;
        VkDevice m_device;
        VkQueue m_graphicsQueue;
        VkQueue m_computeQueue;
        VkQueue m_presentQueue;
        VkCommandPool m_commandPool;
        VkSurfaceKHR m_surface;
        VkSampleCountFlagBits m_msaaSamples = VK_SAMPLE_COUNT_1_BIT;

        std::unordered_set<VkShaderModule> m_shaderModules;

        std::vector<char> loadShader(std::istream& stream) {
            const size_t shaderSize = static_cast<size_t>(stream.tellg());
            auto buffer = std::vector<char>(shaderSize);

            stream.seekg(0);
            stream.read(buffer.data(), shaderSize);

            return buffer;
        }

        std::ifstream openShaderFile(const std::string& fileName) {
            auto file = std::ifstream { fileName, std::ios::ate | std::ios::binary };

            if (!file.is_open()) {
                throw std::runtime_error("failed to open file!");
            }

            return file;
        }

        std::vector<char> loadShaderFromFile(const std::string& fileName) {
            auto stream = this->openShaderFile(fileName);
            auto shader = this->loadShader(stream);
            stream.close();

            return shader;
        }

        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
            auto memProperties = VkPhysicalDeviceMemoryProperties {};
            vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

            for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
                if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                    return i;
                }
            }

            throw std::runtime_error("failed to find suitable memory type!");
        }
};

class GpuDeviceInitializer final {
    public:
        explicit GpuDeviceInitializer(VkInstance instance)
            : m_instance { instance }
        {
            m_infoProvider = new PlatformInfoProvider {};
        }

        ~GpuDeviceInitializer() {
            vkDestroySurfaceKHR(m_instance, m_dummySurface, nullptr);

            m_instance = VK_NULL_HANDLE;
        }

        GpuDevice* createGpuDevice() {
            this->createDummySurface();
            this->selectPhysicalDevice();
            this->createLogicalDevice();
            this->createCommandPool();

            const auto gpuDevice = new GpuDevice {
                m_instance,
                m_physicalDevice,
                m_device,
                m_graphicsQueue,
                m_computeQueue,
                m_presentQueue,
                m_commandPool
            };

            return gpuDevice;
        }
    private:
        VkInstance m_instance;
        PlatformInfoProvider* m_infoProvider;
        VkSurfaceKHR m_dummySurface;
        VkPhysicalDevice m_physicalDevice;
        VkDevice m_device;
        VkQueue m_graphicsQueue;
        VkQueue m_computeQueue;
        VkQueue m_presentQueue;
        VkCommandPool m_commandPool;

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
            auto indices = QueueFamilyIndices {};

            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

            auto queueFamilies = std::vector<VkQueueFamilyProperties> { queueFamilyCount };
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

            int i = 0;
            for (const auto& queueFamily : queueFamilies) {
                if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                    indices.graphicsAndComputeFamily = i;
                }

                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);

                if (presentSupport) {
                    indices.presentFamily = i;
                }

                if (indices.isComplete()) {
                    break;
                }

                i++;
            }

            return indices;
        }

        void createDummySurface() {
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

            auto dummyWindow = glfwCreateWindow(1, 1, "DUMMY WINDOW", nullptr, nullptr);
            auto dummySurface = VkSurfaceKHR {};
            const auto result = glfwCreateWindowSurface(m_instance, dummyWindow, nullptr, &dummySurface);
            if (result != VK_SUCCESS) {
                glfwDestroyWindow(dummyWindow);
                dummyWindow = nullptr;

                throw std::runtime_error("failed to create window surface!");
            }

            glfwDestroyWindow(dummyWindow);
            dummyWindow = nullptr;

            m_dummySurface = dummySurface;
        }

        void selectPhysicalDevice() {
            const auto physicalDeviceSpecProvider = PhysicalDeviceSpecProvider {};
            const auto physicalDeviceSpec = physicalDeviceSpecProvider.createPhysicalDeviceSpec();
            const auto physicalDeviceSelector = PhysicalDeviceSelector { m_instance, m_infoProvider };
            const auto selectedPhysicalDevice = physicalDeviceSelector.selectPhysicalDeviceForSurface(
                m_dummySurface,
                physicalDeviceSpec
            );

            m_physicalDevice = selectedPhysicalDevice;
        }

        void createLogicalDevice() {
            const auto logicalDeviceSpecProvider = LogicalDeviceSpecProvider { m_physicalDevice, m_dummySurface };
            const auto logicalDeviceSpec = logicalDeviceSpecProvider.createLogicalDeviceSpec();
            auto factory = LogicalDeviceFactory { m_physicalDevice, m_dummySurface, m_infoProvider };
            const auto [device, graphicsQueue, computeQueue, presentQueue] = factory.createLogicalDevice(logicalDeviceSpec);

            m_device = device;
            m_graphicsQueue = graphicsQueue;
            m_computeQueue = computeQueue;
            m_presentQueue = presentQueue;
        }

        void createCommandPool() {
            const auto queueFamilyIndices = this->findQueueFamilies(m_physicalDevice, m_dummySurface);
            const auto poolInfo = VkCommandPoolCreateInfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value(),
            };

            auto commandPool = VkCommandPool {};
            const auto result = vkCreateCommandPool(m_device, &poolInfo, nullptr, &commandPool);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create command pool!");
            }

            m_commandPool = commandPool;
        }
};

class Engine final {
    public:
        explicit Engine() = default;
        ~Engine() {
            delete m_windowSystem;
            delete m_gpuDevice;
            delete m_debugMessenger;

            vkDestroyInstance(m_instance, nullptr);

            delete m_systemFactory;
            delete m_infoProvider;

            glfwTerminate();
        }

        static std::unique_ptr<Engine> createDebugMode() {
            return Engine::create(true);
        }

        static std::unique_ptr<Engine> createReleaseMode() {
            return Engine::create(false);
        }

        VkInstance getInstance() const {
            return m_instance;
        }

        VkPhysicalDevice getPhysicalDevice() const {
            return m_gpuDevice->getPhysicalDevice();
        }

        VkDevice getLogicalDevice() const {
            return m_gpuDevice->getLogicalDevice();
        }

        VkQueue getGraphicsQueue() const {
            return m_gpuDevice->getGraphicsQueue();
        }

        VkQueue getComputeQueue() const {
            return m_gpuDevice->getComputeQueue();
        }

        VkQueue getPresentQueue() const {
            return m_gpuDevice->getPresentQueue();
        }

        VkCommandPool getCommandPool() const {
            return m_gpuDevice->getCommandPool();
        }

        VkSurfaceKHR getSurface() const {
            return m_surface;
        }

        VkSampleCountFlagBits getMsaaSamples() const {
            return m_gpuDevice->getMsaaSamples();
        }

        GLFWwindow* getWindow() const {
            return m_windowSystem->getWindow();
        }

        bool hasFramebufferResized() const {
            return m_windowSystem->hasFramebufferResized();
        }

        void setFramebufferResized(bool framebufferResized) {
            m_windowSystem->setFramebufferResized(framebufferResized);
        }

        bool isInitialized() const {
            return m_instance != VK_NULL_HANDLE;
        }

        void createGLFWLibrary() {
            const auto result = glfwInit();
            if (!result) {
                glfwTerminate();

                auto errorMessage = std::string { "Failed to initialize GLFW" };

                throw std::runtime_error { errorMessage };
            }
        }

        void createInfoProvider() {
            const auto infoProvider = new PlatformInfoProvider {};

            m_infoProvider = infoProvider;
        }

        void createSystemFactory() {
            const auto systemFactory = new SystemFactory {};

            m_systemFactory = systemFactory;
        }

        void createInstance() {
            const auto instanceSpecProvider = InstanceSpecProvider { m_enableValidationLayers, m_enableDebuggingExtensions };
            const auto instanceSpec = instanceSpecProvider.createInstanceSpec();
            const auto instance = m_systemFactory->create(instanceSpec);
        
            m_instance = instance;
        }

        void createWindowSystem() {
            const auto windowSystem = WindowSystem::create(m_instance);

            m_windowSystem = windowSystem;
        }

        void createDebugMessenger() {
            if (!m_enableValidationLayers) {
                return;
            }

            const auto debugMessenger = VulkanDebugMessenger::create(m_instance);

            m_debugMessenger = debugMessenger;
        }

        void createWindow(uint32_t width, uint32_t height, const std::string& title) {
            m_windowSystem->createWindow(width, height, title);
       
            this->createRenderSurface();
        }

        void createGpuDevice() {
            auto gpuDeviceInitializer = GpuDeviceInitializer { m_instance };
            const auto gpuDevice = gpuDeviceInitializer.createGpuDevice();

            m_gpuDevice = gpuDevice;
        }

        void createRenderSurface() {
            auto surfaceProvider = m_windowSystem->createSurfaceProvider();
            const auto surface = m_gpuDevice->createRenderSurface(surfaceProvider);

            m_surface = surface;
        }

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
            auto indices = QueueFamilyIndices {};

            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

            auto queueFamilies = std::vector<VkQueueFamilyProperties> { queueFamilyCount };
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

            int i = 0;
            for (const auto& queueFamily : queueFamilies) {
                if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                    indices.graphicsAndComputeFamily = i;
                }

                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);

                if (presentSupport) {
                    indices.presentFamily = i;
                }

                if (indices.isComplete()) {
                    break;
                }

                i++;
            }

            return indices;
        }

        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
            auto details = SwapChainSupportDetails {};
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &details.capabilities);

            uint32_t formatCount = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);

            if (formatCount != 0) {
                details.formats.resize(formatCount);
                vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, details.formats.data());
            }

            uint32_t presentModeCount = 0;
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);

            if (presentModeCount != 0) {
                details.presentModes.resize(presentModeCount);
                vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, details.presentModes.data());
            }

            return details;
        }

        VkShaderModule createShaderModuleFromFile(const std::string& fileName) {
            return m_gpuDevice->createShaderModuleFromFile(fileName);
        }

        VkShaderModule createShaderModule(std::istream& stream) {
            return m_gpuDevice->createShaderModule(stream);
        }

        VkShaderModule createShaderModule(std::vector<char>& code) {
            return m_gpuDevice->createShaderModule(code);
        }
    private:
        PlatformInfoProvider* m_infoProvider;
        SystemFactory* m_systemFactory;
        VkInstance m_instance;
        VulkanDebugMessenger* m_debugMessenger;
        WindowSystem* m_windowSystem;
        VkSurfaceKHR m_surface;

        GpuDevice* m_gpuDevice;

        bool m_enableValidationLayers; 
        bool m_enableDebuggingExtensions;

        static std::unique_ptr<Engine> create(bool enableDebugging) {
            auto newEngine = std::make_unique<Engine>();

            if (enableDebugging) {
                newEngine->m_enableValidationLayers = true;
                newEngine->m_enableDebuggingExtensions = true;
            } else {
                newEngine->m_enableValidationLayers = false;
                newEngine->m_enableDebuggingExtensions = false;
            }

            newEngine->createGLFWLibrary();
            newEngine->createInfoProvider();
            newEngine->createSystemFactory();
            newEngine->createInstance();
            newEngine->createDebugMessenger();
            newEngine->createGpuDevice();
            newEngine->createWindowSystem();

            return newEngine;
        }
};

}

#endif // _ENGINE_H
