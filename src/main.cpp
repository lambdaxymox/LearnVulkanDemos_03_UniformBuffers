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

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <glm/glm.hpp>

#include <vulkan_engine/vulkan_platform.h>
#include <vulkan_engine/vulkan_platform_impl_fmt.h>

#include <unordered_set>

#ifndef GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif // GLFW_INCLUDE_VULKAN


#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<std::string> validationLayers = std::vector<std::string> { 
    VulkanEngine::Constants::VK_LAYER_KHRONOS_validation
};

const std::vector<const char*> deviceExtensions = std::vector<const char*> {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
const bool enableDebuggingExtensions = false;
#else
const bool enableValidationLayers = true;
const bool enableDebuggingExtensions = true;
#endif

const int MAX_FRAMES_IN_FLIGHT = 2;




struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = std::vector<Vertex> {
    Vertex { { -0.5f, -0.5f}, { 1.0f, 0.0f, 0.0f } },
    Vertex { {  0.5f, -0.5f}, { 0.0f, 1.0f, 0.0f } },
    Vertex { {  0.5f,  0.5f}, { 0.0f, 0.0f, 1.0f } },
    Vertex { { -0.5f,  0.5f}, { 1.0f, 1.0f, 1.0f } }
};

const std::vector<uint16_t> indices = std::vector<uint16_t> {
    0, 1, 2, 2, 3, 0
};

/// @brief The uniform buffer object for distpaching camera data to the GPU.
///
/// @note Vulkan expects data to be aligned in a specific way. For example,
/// let `T` be a data type.
///
/// @li If `T` is a scalar, `align(T) == sizeof(T)`
/// @li If `T` is a scalar, `align(vec2<T>) == 2 * sizeof(T)`
/// @li If `T` is a scalar, `align(vec3<T>) == 4 * sizeof(T)`
/// @li If `T` is a scalar, `align(vec4<T>) == 4 * sizeof(T)`
/// @li If `T` is a scalar, `align(mat4<T>) == 4 * sizeof(T)`
/// @li If `T` is a structure type, `align(T) == max(align(members(T)))`
///
/// In particular, each data type is a nice multiple of the alignment of the largest
/// scalar type constituting that data type. See the specification
/// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap15.html#interfaces-resources-layout
/// for more details.
struct UniformBufferObject {
    glm::mat4x4 model;
    glm::mat4x4 view;
    glm::mat4x4 proj;
};


using VulkanInstanceProperties = VulkanEngine::VulkanPlatform::VulkanInstanceProperties;
using PhysicalDeviceProperties = VulkanEngine::VulkanPlatform::PhysicalDeviceProperties;
using Platform = VulkanEngine::VulkanPlatform::PlatformInfoProvider::Platform;
using PlatformInfoProvider = VulkanEngine::VulkanPlatform::PlatformInfoProvider;


struct QueueFamilyIndices final {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
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
        auto instanceExtensions = this->getInstanceExtensions();
        auto instanceLayers = this->getInstanceLayers();
        auto instanceCreateFlags = this->minInstanceCreateFlags();
        auto applicationName = std::string { "" };
        auto engineName = std::string { "" };

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

        auto instanceInfo = m_infoProvider->getVulkanInstanceInfo();
        auto instanceExtensions = instanceSpec.instanceExtensions();
        auto instanceLayers = instanceSpec.instanceLayers();
        auto missingExtensions = m_infoProvider->detectMissingInstanceExtensions(
            instanceInfo,
            instanceExtensions
        );
        auto missingLayers = m_infoProvider->detectMissingInstanceLayers(
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

        auto instanceCreateFlags = instanceSpec.instanceCreateFlags();
        auto enabledLayerNames = SystemFactory::convertToCStrings(instanceLayers);
        auto enabledExtensionNames = SystemFactory::convertToCStrings(instanceExtensions);

        auto appInfo = VkApplicationInfo {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = instanceSpec.applicationName().data();
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = instanceSpec.engineName().data();
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        auto createInfo = VkInstanceCreateInfo {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.flags |= instanceCreateFlags;
        createInfo.enabledLayerCount = enabledLayerNames.size();
        createInfo.ppEnabledLayerNames = enabledLayerNames.data();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensionNames.size());
        createInfo.ppEnabledExtensionNames = enabledExtensionNames.data();

        VkInstance instance = VK_NULL_HANDLE;
        auto result = vkCreateInstance(&createInfo, nullptr, &instance);
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
        auto requiredExtensions = this->getPhysicalDeviceRequirements();

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
        auto indices = QueueFamilyIndices {};

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

        auto queueFamilies = std::vector<VkQueueFamilyProperties> { queueFamilyCount };
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
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
        uint32_t extensionCount;
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
        QueueFamilyIndices indices = this->findQueueFamilies(physicalDevice, surface);

        bool areRequiredExtensionsSupported = this->checkDeviceExtensionSupport(
            physicalDevice,
            physicalDeviceSpec.requiredExtensions()
        );

        bool swapChainCompatible = false;
        if (areRequiredExtensionsSupported) {
            SwapChainSupportDetails swapChainSupport = this->querySwapChainSupport(physicalDevice, surface);
            swapChainCompatible = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && areRequiredExtensionsSupported && swapChainCompatible;
    }

    std::vector<VkPhysicalDevice> findAllPhysicalDevices() const {
        uint32_t physicalDeviceCount = 0;
        vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, nullptr);

        auto physicalDevices = std::vector<VkPhysicalDevice> { physicalDeviceCount };
        vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, physicalDevices.data());

        return physicalDevices;
    }

    std::vector<VkPhysicalDevice> findCompatiblePhysicalDevices(VkSurfaceKHR surface, const PhysicalDeviceSpec& physicalDeviceSpec) const {
        auto physicalDevices = this->findAllPhysicalDevices();
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
        auto physicalDevices = this->findCompatiblePhysicalDevices(surface, physicalDeviceSpec);
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
    explicit LogicalDeviceSpec(const std::vector<std::string>& requiredExtensions)
        : m_requiredExtensions { requiredExtensions }
    {
    }
    ~LogicalDeviceSpec() = default;

    const std::vector<std::string>& requiredExtensions() const {
        return m_requiredExtensions;
    }
private:
    std::vector<std::string> m_requiredExtensions;
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

        return LogicalDeviceSpec { requiredExtensions };
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
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
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

    std::tuple<VkDevice, VkQueue, VkQueue> createLogicalDevice(const LogicalDeviceSpec& logicalDeviceSpec) {
        QueueFamilyIndices indices = this->findQueueFamilies(m_physicalDevice, m_surface);

        auto uniqueQueueFamilies = std::set<uint32_t> {
            indices.graphicsFamily.value(), 
            indices.presentFamily.value()
        };
        auto queueCreateInfos = std::vector<VkDeviceQueueCreateInfo> {};
        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            auto queueCreateInfo = VkDeviceQueueCreateInfo {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        auto deviceFeatures = VkPhysicalDeviceFeatures {};

        auto createInfo = VkDeviceCreateInfo {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        
        createInfo.pEnabledFeatures = &deviceFeatures;

        auto deviceExtensionProperties = m_infoProvider->getAvailableVulkanDeviceExtensions(m_physicalDevice);
        auto missingExtensions = m_infoProvider->detectMissingRequiredDeviceExtensions(
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
        auto enabledExtensions = LogicalDeviceFactory::convertToCStrings(logicalDeviceSpec.requiredExtensions());
        createInfo.enabledExtensionCount = enabledExtensions.size();
        createInfo.ppEnabledExtensionNames = enabledExtensions.data();
        
        auto validationLayersCStrings = LogicalDeviceFactory::convertToCStrings(validationLayers);

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayersCStrings.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        VkDevice device = VK_NULL_HANDLE;
        auto result = vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &device);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }


        VkQueue graphicsQueue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        
        VkQueue presentQueue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

        return std::make_tuple(device, graphicsQueue, presentQueue);
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
        auto result = vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
        if (result != VK_SUCCESS) {
            throw std::invalid_argument { "Got an invalid `VkInstance` handle" };
        }

        auto createInfo = VkDebugUtilsMessengerCreateInfoEXT {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = 
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = 
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;

        auto debugMessenger = static_cast<VkDebugUtilsMessengerEXT>(nullptr);
        result = VulkanDebugMessenger::CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger);
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
        auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
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
        auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
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
        auto messageSeverityString = VulkanDebugMessenger::messageSeverityToString(messageSeverity);
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
    SurfaceProvider() = delete;
    SurfaceProvider(VkInstance instance, GLFWwindow* window)
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
        auto result = glfwCreateWindowSurface(m_instance, m_window, nullptr, &surface);
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
        VkQueue presentQueue,
        VkCommandPool commandPool
    )   : m_instance { instance }
        , m_physicalDevice { physicalDevice }
        , m_device { device }
        , m_graphicsQueue { graphicsQueue }
        , m_presentQueue { presentQueue }
        , m_commandPool { commandPool }
        , m_shaderModules { std::unordered_set<VkShaderModule> {} }
    {
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

    VkQueue getPresentQueue() const {
        return m_presentQueue;
    }

    VkCommandPool getCommandPool() const {
        return m_commandPool;
    }

    VkSurfaceKHR createRenderSurface(SurfaceProvider& surfaceProvider) {
        auto surface = surfaceProvider.createSurface();

        m_surface = surface;

        return surface;
    }

    VkShaderModule createShaderModuleFromFile(const std::string& fileName) {
        auto shaderCode = this->loadShaderFromFile(fileName);

        return this->createShaderModule(shaderCode);
    }

    VkShaderModule createShaderModule(std::istream& stream) {
        auto shaderCode = this->loadShader(stream);

        return this->createShaderModule(shaderCode);
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        auto createInfo = VkShaderModuleCreateInfo {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        createInfo.pNext = nullptr;
        createInfo.flags = 0;

        auto shaderModule = VkShaderModule {};
        auto result = vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule);
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
    VkQueue m_presentQueue;
    VkCommandPool m_commandPool;
    VkSurfaceKHR m_surface;

    std::unordered_set<VkShaderModule> m_shaderModules;

    std::vector<char> loadShader(std::istream& stream) {
        size_t shaderSize = static_cast<size_t>(stream.tellg());
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

        auto gpuDevice = new GpuDevice { m_instance, m_physicalDevice, m_device, m_graphicsQueue, m_presentQueue, m_commandPool };

        return gpuDevice;
    }
private:
    VkInstance m_instance;
    PlatformInfoProvider* m_infoProvider;
    VkSurfaceKHR m_dummySurface;
    VkPhysicalDevice m_physicalDevice;
    VkDevice m_device;
    VkQueue m_graphicsQueue;
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
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
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
        auto result = glfwCreateWindowSurface(m_instance, dummyWindow, nullptr, &dummySurface);
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
        auto physicalDeviceSpecProvider = PhysicalDeviceSpecProvider {};
        auto physicalDeviceSpec = physicalDeviceSpecProvider.createPhysicalDeviceSpec();
        auto physicalDeviceSelector = PhysicalDeviceSelector { m_instance, m_infoProvider };
        auto selectedPhysicalDevice = physicalDeviceSelector.selectPhysicalDeviceForSurface(
            m_dummySurface,
            physicalDeviceSpec
        );

        m_physicalDevice = selectedPhysicalDevice;
    }

    void createLogicalDevice() {
        auto logicalDeviceSpecProvider = LogicalDeviceSpecProvider { m_physicalDevice, m_dummySurface };
        auto logicalDeviceSpec = logicalDeviceSpecProvider.createLogicalDeviceSpec();
        auto factory = LogicalDeviceFactory { m_physicalDevice, m_dummySurface, m_infoProvider };
        auto [device, graphicsQueue, presentQueue] = factory.createLogicalDevice(logicalDeviceSpec);

        m_device = device;
        m_graphicsQueue = graphicsQueue;
        m_presentQueue = presentQueue;
    }

    void createCommandPool() {
        auto queueFamilyIndices = this->findQueueFamilies(m_physicalDevice, m_dummySurface);

        auto poolInfo = VkCommandPoolCreateInfo {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        VkCommandPool commandPool = VK_NULL_HANDLE;
        auto result = vkCreateCommandPool(m_device, &poolInfo, nullptr, &commandPool);
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

    static Engine* createDebugMode() {
        return Engine::create(true);
    }

    static Engine* createReleaseMode() {
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

    VkQueue getPresentQueue() const {
        return m_gpuDevice->getPresentQueue();
    }

    VkCommandPool getCommandPool() const {
        return m_gpuDevice->getCommandPool();
    }

    VkSurfaceKHR getSurface() const {
        return m_surface;
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
        auto result = glfwInit();
        if (!result) {
            glfwTerminate();

            auto errorMessage = std::string { "Failed to initialize GLFW" };

            throw std::runtime_error { errorMessage };
        }
    }

    void createInfoProvider() {
        auto infoProvider = new PlatformInfoProvider {};

        m_infoProvider = infoProvider;
    }

    void createSystemFactory() {
        auto systemFactory = new SystemFactory {};

        m_systemFactory = systemFactory;
    }

    void createInstance() {
        auto instanceSpecProvider = InstanceSpecProvider { m_enableValidationLayers, m_enableDebuggingExtensions };
        auto instanceSpec = instanceSpecProvider.createInstanceSpec();
        auto instance = m_systemFactory->create(instanceSpec);
        
        m_instance = instance;
    }

    void createWindowSystem() {
        auto windowSystem = WindowSystem::create(m_instance);

        m_windowSystem = windowSystem;
    }

    void createDebugMessenger() {
        if (!m_enableValidationLayers) {
            return;
        }

        auto debugMessenger = VulkanDebugMessenger::create(m_instance);

        m_debugMessenger = debugMessenger;
    }

    void createWindow(uint32_t width, uint32_t height, const std::string& title) {
        m_windowSystem->createWindow(width, height, title);
       
        this->createRenderSurface();
    }

    void createGpuDevice() {
        auto gpuDeviceInitializer = GpuDeviceInitializer { m_instance };
        auto gpuDevice = gpuDeviceInitializer.createGpuDevice();

        m_gpuDevice = gpuDevice;
    }

    void createRenderSurface() {
        auto surfaceProvider = m_windowSystem->createSurfaceProvider();
        auto surface = m_gpuDevice->createRenderSurface(surfaceProvider);

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
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
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

    static Engine* create(bool enableDebugging) {
        auto newEngine = new Engine {};

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

class App {
public:
    void run() {
        this->initEngine();
        this->mainLoop();
        this->cleanup();
    }
private:
    Engine* m_engine;

    VkBuffer m_vertexBuffer;
    VkDeviceMemory m_vertexBufferMemory;
    VkBuffer m_indexBuffer;
    VkDeviceMemory m_indexBufferMemory;

    std::vector<VkBuffer> m_uniformBuffers;
    std::vector<VkDeviceMemory> m_uniformBuffersMemory;
    std::vector<void*> m_uniformBuffersMapped;

    VkDescriptorPool m_descriptorPool;
    std::vector<VkDescriptorSet> m_descriptorSets;
    VkDescriptorSetLayout m_descriptorSetLayout;

    std::vector<VkCommandBuffer> m_commandBuffers;

    VkRenderPass m_renderPass;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_graphicsPipeline;

    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence> m_inFlightFences;

    VkSwapchainKHR m_swapChain;
    std::vector<VkImage> m_swapChainImages;
    VkFormat m_swapChainImageFormat;
    VkExtent2D m_swapChainExtent;
    std::vector<VkImageView> m_swapChainImageViews;
    std::vector<VkFramebuffer> m_swapChainFramebuffers;
    
    uint32_t m_currentFrame = 0;

    bool m_enableValidationLayers { false };
    bool m_enableDebuggingExtensions { false };

    void cleanup() {
        if (m_engine->isInitialized()) {
            this->cleanupSwapChain();

            vkDestroyPipeline(m_engine->getLogicalDevice(), m_graphicsPipeline, nullptr);
            vkDestroyPipelineLayout(m_engine->getLogicalDevice(), m_pipelineLayout, nullptr);
            vkDestroyRenderPass(m_engine->getLogicalDevice(), m_renderPass, nullptr);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vkDestroySemaphore(m_engine->getLogicalDevice(), m_renderFinishedSemaphores[i], nullptr);
                vkDestroySemaphore(m_engine->getLogicalDevice(), m_imageAvailableSemaphores[i], nullptr);
                vkDestroyFence(m_engine->getLogicalDevice(), m_inFlightFences[i], nullptr);
            }

            vkDestroyDescriptorPool(m_engine->getLogicalDevice(), m_descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(m_engine->getLogicalDevice(), m_descriptorSetLayout, nullptr);

            vkDestroyBuffer(m_engine->getLogicalDevice(), m_indexBuffer, nullptr);
            vkFreeMemory(m_engine->getLogicalDevice(), m_indexBufferMemory, nullptr);

            vkDestroyBuffer(m_engine->getLogicalDevice(), m_vertexBuffer, nullptr);
            vkFreeMemory(m_engine->getLogicalDevice(), m_vertexBufferMemory, nullptr);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vkDestroyBuffer(m_engine->getLogicalDevice(), m_uniformBuffers[i], nullptr);
                vkFreeMemory(m_engine->getLogicalDevice(), m_uniformBuffersMemory[i], nullptr);
            }

            delete m_engine;
        }
    }

    void createEngine() {
        auto engine = Engine::createDebugMode();
        engine->createWindow(WIDTH, HEIGHT, "Uniform Buffers");

        m_engine = engine;
    }

    void initEngine() {
        this->createEngine();

        this->createVertexBuffer();
        this->createIndexBuffer();
        this->createDescriptorSetLayout();
        this->createUniformBuffers();
        this->createDescriptorPool();
        this->createDescriptorSets();
        this->createCommandBuffers();
        this->createSwapChain();
        this->createImageViews();
        this->createRenderPass();
        this->createGraphicsPipeline();
        this->createFramebuffers();
        this->createSyncObjects();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(m_engine->getWindow())) {
            glfwPollEvents();
            this->draw();
        }

        vkDeviceWaitIdle(m_engine->getLogicalDevice());
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(m_engine->getPhysicalDevice(), &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        auto bufferInfo = VkBufferCreateInfo {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        auto result = vkCreateBuffer(m_engine->getLogicalDevice(), &bufferInfo, nullptr, &buffer);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        auto memRequirements = VkMemoryRequirements {};
        vkGetBufferMemoryRequirements(m_engine->getLogicalDevice(), buffer, &memRequirements);

        auto allocInfo = VkMemoryAllocateInfo {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        result = vkAllocateMemory(m_engine->getLogicalDevice(), &allocInfo, nullptr, &bufferMemory);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(m_engine->getLogicalDevice(), buffer, bufferMemory, 0);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        auto allocInfo = VkCommandBufferAllocateInfo {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = m_engine->getCommandPool();
        allocInfo.commandBufferCount = 1;

        auto commandBuffer = VkCommandBuffer {};
        vkAllocateCommandBuffers(m_engine->getLogicalDevice(), &allocInfo, &commandBuffer);

        auto beginInfo = VkCommandBufferBeginInfo {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        auto copyRegion = VkBufferCopy {};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        auto submitInfo = VkSubmitInfo {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(m_engine->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_engine->getGraphicsQueue());

        vkFreeCommandBuffers(m_engine->getLogicalDevice(), m_engine->getCommandPool(), 1, &commandBuffer);
    }

    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        auto stagingBuffer = VkBuffer {};
        auto stagingBufferMemory = VkDeviceMemory {};
        VkBufferUsageFlags stagingBufferUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VkMemoryPropertyFlags stagingBufferPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        this->createBuffer(bufferSize, stagingBufferUsageFlags, stagingBufferPropertyFlags, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(m_engine->getLogicalDevice(), stagingBufferMemory, 0, bufferSize, 0, &data);
        
        memcpy(data, vertices.data(), (size_t) bufferSize);
        
        vkUnmapMemory(m_engine->getLogicalDevice(), stagingBufferMemory);

        VkBufferUsageFlags vertexBufferUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        VkMemoryPropertyFlags vertexBufferPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        this->createBuffer(bufferSize, vertexBufferUsageFlags, vertexBufferPropertyFlags, m_vertexBuffer, m_vertexBufferMemory);

        this->copyBuffer(stagingBuffer, m_vertexBuffer, bufferSize);

        vkDestroyBuffer(m_engine->getLogicalDevice(), stagingBuffer, nullptr);
        vkFreeMemory(m_engine->getLogicalDevice(), stagingBufferMemory, nullptr);
    }

    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        auto stagingBuffer = VkBuffer {};
        auto stagingBufferMemory = VkDeviceMemory {};
        VkBufferUsageFlags stagingBufferUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VkMemoryPropertyFlags stagingBufferPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        this->createBuffer(bufferSize, stagingBufferUsageFlags, stagingBufferPropertyFlags, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(m_engine->getLogicalDevice(), stagingBufferMemory, 0, bufferSize, 0, &data);
        
        memcpy(data, indices.data(), (size_t) bufferSize);
        
        vkUnmapMemory(m_engine->getLogicalDevice(), stagingBufferMemory);

        VkBufferUsageFlags vertexBufferUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        VkMemoryPropertyFlags vertexBufferPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        this->createBuffer(bufferSize, vertexBufferUsageFlags, vertexBufferPropertyFlags, m_indexBuffer, m_indexBufferMemory);

        this->copyBuffer(stagingBuffer, m_indexBuffer, bufferSize);

        vkDestroyBuffer(m_engine->getLogicalDevice(), stagingBuffer, nullptr);
        vkFreeMemory(m_engine->getLogicalDevice(), stagingBufferMemory, nullptr);
    }

    void createDescriptorSetLayout() {
        auto uboLayoutBinding = VkDescriptorSetLayoutBinding {};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.pImmutableSamplers = nullptr;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        auto layoutInfo = VkDescriptorSetLayoutCreateInfo {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &uboLayoutBinding;

        auto descriptorSetLayout = VkDescriptorSetLayout {};
        auto result = vkCreateDescriptorSetLayout(m_engine->getLogicalDevice(), &layoutInfo, nullptr, &descriptorSetLayout);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }

        m_descriptorSetLayout = descriptorSetLayout;
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        m_uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        m_uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        m_uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            VkMemoryPropertyFlags propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            this->createBuffer(bufferSize, usageFlags, propertyFlags, m_uniformBuffers[i], m_uniformBuffersMemory[i]);
            
            vkMapMemory(m_engine->getLogicalDevice(), m_uniformBuffersMemory[i], 0, bufferSize, 0, &m_uniformBuffersMapped[i]);
        }
    }

    void createDescriptorPool() {
        auto poolSize = VkDescriptorPoolSize {};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        auto poolInfo = VkDescriptorPoolCreateInfo {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        auto descriptorPool = VkDescriptorPool {};
        auto result = vkCreateDescriptorPool(m_engine->getLogicalDevice(), &poolInfo, nullptr, &descriptorPool);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }

        m_descriptorPool = descriptorPool;
    }

    void createDescriptorSets() {
        auto layouts = std::vector<VkDescriptorSetLayout> { MAX_FRAMES_IN_FLIGHT, m_descriptorSetLayout };
        
        auto allocInfo = VkDescriptorSetAllocateInfo {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        m_descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        auto result = vkAllocateDescriptorSets(m_engine->getLogicalDevice(), &allocInfo, m_descriptorSets.data());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            auto bufferInfo = VkDescriptorBufferInfo {};
            bufferInfo.buffer = m_uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            auto descriptorWrite = VkWriteDescriptorSet {};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = m_descriptorSets[i];
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pBufferInfo = &bufferInfo;
            descriptorWrite.pImageInfo = nullptr;       // Optional
            descriptorWrite.pTexelBufferView = nullptr; // Optional

            vkUpdateDescriptorSets(m_engine->getLogicalDevice(), 1, &descriptorWrite, 0, nullptr);
        }
    }

    void createCommandBuffers() {
        m_commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        auto allocInfo = VkCommandBufferAllocateInfo {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_engine->getCommandPool();
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());

        auto result = vkAllocateCommandBuffers(m_engine->getLogicalDevice(), &allocInfo, m_commandBuffers.data());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    VkSurfaceFormatKHR selectSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR selectSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        // We would probably want to use `VK_PRESENT_MODE_FIFO_KHR` on mobile devices.
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D selectSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetWindowSize(m_engine->getWindow(), &width, &height);

            auto actualExtent = VkExtent2D {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(
                actualExtent.width, 
                capabilities.minImageExtent.width, 
                capabilities.maxImageExtent.width
            );
            actualExtent.height = std::clamp(
                actualExtent.height, 
                capabilities.minImageExtent.height, 
                capabilities.maxImageExtent.height
            );

            return actualExtent;
        }
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = m_engine->querySwapChainSupport(m_engine->getPhysicalDevice(), m_engine->getSurface());
        VkSurfaceFormatKHR surfaceFormat = this->selectSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = this->selectSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = this->selectSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        auto createInfo = VkSwapchainCreateInfoKHR {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = m_engine->getSurface();

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = m_engine->findQueueFamilies(m_engine->getPhysicalDevice(), m_engine->getSurface());
        auto queueFamilyIndices = std::array<uint32_t, 2> { 
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(m_engine->getLogicalDevice(), &createInfo, nullptr, &m_swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(m_engine->getLogicalDevice(), m_swapChain, &imageCount, nullptr);
        m_swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(m_engine->getLogicalDevice(), m_swapChain, &imageCount, m_swapChainImages.data());

        m_swapChainImageFormat = surfaceFormat.format;
        m_swapChainExtent = extent;
    }

    void createImageViews() {
        m_swapChainImageViews.resize(m_swapChainImages.size());

        for (size_t i = 0; i < m_swapChainImages.size(); i++) {
            auto createInfo = VkImageViewCreateInfo {};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = m_swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = m_swapChainImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            auto result = vkCreateImageView(m_engine->getLogicalDevice(), &createInfo, nullptr, &m_swapChainImageViews[i]);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Failed to create image views!");
            }
        }
    }

    void createRenderPass() {
        auto colorAttachment = VkAttachmentDescription {};
        colorAttachment.format = m_swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        auto colorAttachmentRef = VkAttachmentReference {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        auto subpass = VkSubpassDescription {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        auto renderPassInfo = VkRenderPassCreateInfo {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        auto result = vkCreateRenderPass(m_engine->getLogicalDevice(), &renderPassInfo, nullptr, &m_renderPass);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createGraphicsPipeline() {
        auto vertexShaderModule = m_engine->createShaderModuleFromFile("shaders/shader.vert.hlsl.spv");
        auto fragmentShaderModule = m_engine->createShaderModuleFromFile("shaders/shader.frag.hlsl.spv");

        auto vertexShaderStageInfo = VkPipelineShaderStageCreateInfo {};
        vertexShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertexShaderStageInfo.module = vertexShaderModule;
        vertexShaderStageInfo.pName = "main";

        auto fragmentShaderStageInfo = VkPipelineShaderStageCreateInfo {};
        fragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragmentShaderStageInfo.module = fragmentShaderModule;
        fragmentShaderStageInfo.pName = "main";

        auto shaderStages = std::array<VkPipelineShaderStageCreateInfo, 2> {
            vertexShaderStageInfo,
            fragmentShaderStageInfo
        };

        auto vertexInputInfo = VkPipelineVertexInputStateCreateInfo {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        auto inputAssembly = VkPipelineInputAssemblyStateCreateInfo {};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // Without dynamic state, the viewport and scissor rectangle need to be set 
        // in the pipeline using the `VkPipelineViewportStateCreateInfo` struct. This
        // makes the viewport and scissor rectangle for this pipeline immutable.
        // Any changes to these values would require a new pipeline to be created with
        // the new values.
        // VkPipelineViewportStateCreateInfo viewportState{};
        // viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        // viewportState.viewportCount = 1;
        // viewportState.pViewports = &viewport;
        // viewportState.scissorCount = 1;
        // viewportState.pScissors = &scissor;
        auto viewportState = VkPipelineViewportStateCreateInfo {};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        auto rasterizer = VkPipelineRasterizationStateCreateInfo {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        /*
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        */
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        auto multisampling = VkPipelineMultisampleStateCreateInfo {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.pSampleMask = nullptr;            // Optional.
        multisampling.alphaToCoverageEnable = VK_FALSE; // Optional.
        multisampling.alphaToOneEnable = VK_FALSE;      // Optional.

        auto colorBlendAttachment = VkPipelineColorBlendAttachmentState {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;             // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;             // Optional
        // // Alpha blending:
        // // finalColor.rgb = newAlpha * newColor + (1 - newAlpha) * oldColor;
        // // finalColor.a = newAlpha.a;
        // colorBlendAttachment.blendEnable = VK_TRUE;
        // colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        // colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        // colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        // colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        // colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        // colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        auto colorBlending = VkPipelineColorBlendStateCreateInfo {};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        auto dynamicStates = std::vector<VkDynamicState> {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        auto dynamicState = VkPipelineDynamicStateCreateInfo {};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        auto pipelineLayoutInfo = VkPipelineLayoutCreateInfo {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;    // Optional
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

        auto result = vkCreatePipelineLayout(m_engine->getLogicalDevice(), &pipelineLayoutInfo, nullptr, &m_pipelineLayout);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        auto pipelineInfo = VkGraphicsPipelineCreateInfo {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;       // Optional
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = m_pipelineLayout;
        pipelineInfo.renderPass = m_renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
        pipelineInfo.basePipelineIndex = -1;              // Optional

        result = vkCreateGraphicsPipelines(
            m_engine->getLogicalDevice(), 
            VK_NULL_HANDLE, 
            1, 
            &pipelineInfo, 
            nullptr, 
            &m_graphicsPipeline
        );
        
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        /*
        vkDestroyShaderModule(m_engine->getLogicalDevice(), fragmentShaderModule, nullptr);
        vkDestroyShaderModule(m_engine->getLogicalDevice(), vertexShaderModule, nullptr);
        */
    }

    void createFramebuffers() {
        m_swapChainFramebuffers.resize(m_swapChainImageViews.size());

        for (size_t i = 0; i < m_swapChainImageViews.size(); i++) {
            auto attachments = std::array<VkImageView, 1> {
                m_swapChainImageViews[i]
            };

            auto framebufferInfo = VkFramebufferCreateInfo {};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = m_renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = m_swapChainExtent.width;
            framebufferInfo.height = m_swapChainExtent.height;
            framebufferInfo.layers = 1;

            auto result = vkCreateFramebuffer(
                m_engine->getLogicalDevice(),
                &framebufferInfo,
                nullptr,
                &m_swapChainFramebuffers[i]
            );

            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        auto beginInfo = VkCommandBufferBeginInfo {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;                  // Optional.
        beginInfo.pInheritanceInfo = nullptr; // Optional.

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        auto renderPassInfo = VkRenderPassBeginInfo {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = m_renderPass;
        renderPassInfo.framebuffer = m_swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = VkOffset2D { 0, 0 };
        renderPassInfo.renderArea.extent = m_swapChainExtent;

        auto clearValue = VkClearValue {};
        clearValue.color = VkClearColorValue { { 0.0f, 0.0f, 0.0f, 1.0f } };

        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearValue;

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

        auto viewport = VkViewport {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(m_swapChainExtent.width);
        viewport.height = static_cast<float>(m_swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        auto scissor = VkRect2D {};
        scissor.offset = VkOffset2D { 0, 0 };
        scissor.extent = m_swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        auto vertexBuffers = std::array<VkBuffer, 1> { m_vertexBuffer };
        auto offsets = std::array<VkDeviceSize, 1> { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers.data(), offsets.data());

        vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            1,
            &m_descriptorSets[m_currentFrame],
            0,
            nullptr
        );
        
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createSyncObjects() {
        m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        auto semaphoreInfo = VkSemaphoreCreateInfo {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        auto fenceInfo = VkFenceCreateInfo {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            auto result = vkCreateSemaphore(m_engine->getLogicalDevice(), &semaphoreInfo, nullptr, &m_imageAvailableSemaphores[i]);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create in-flight semaphore synchronization object");
            }

            result = vkCreateSemaphore(m_engine->getLogicalDevice(), &semaphoreInfo, nullptr, &m_renderFinishedSemaphores[i]);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create render finished synchronization object");
            }

            result = vkCreateFence(m_engine->getLogicalDevice(), &fenceInfo, nullptr, &m_inFlightFences[i]);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create in-flight fence synchronization object");
            }
        }
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        auto ubo = UniformBufferObject {};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(
            glm::vec3(2.0f, 2.0f, 2.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f)
        );
        ubo.proj = glm::perspective(
            glm::radians(45.0f),
            m_swapChainExtent.width / (float) m_swapChainExtent.height,
            0.1f,
            10.0f
        );
        ubo.proj[1][1] *= -1;

        memcpy(m_uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void draw() {
        vkWaitForFences(m_engine->getLogicalDevice(), 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        auto result = vkAcquireNextImageKHR(
            m_engine->getLogicalDevice(), 
            m_swapChain, 
            UINT64_MAX, 
            m_imageAvailableSemaphores[m_currentFrame], 
            VK_NULL_HANDLE, 
            &imageIndex
        );

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            this->recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        this->updateUniformBuffer(m_currentFrame);

        vkResetFences(m_engine->getLogicalDevice(), 1, &m_inFlightFences[m_currentFrame]);

        vkResetCommandBuffer(m_commandBuffers[m_currentFrame], /* VkCommandBufferResetFlagBits */ 0);
        this->recordCommandBuffer(m_commandBuffers[m_currentFrame], imageIndex);

        auto submitInfo = VkSubmitInfo {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        auto waitSemaphores = std::array<VkSemaphore, 1> { m_imageAvailableSemaphores[m_currentFrame] };
        auto waitStages = std::array<VkPipelineStageFlags, 1> { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffers[m_currentFrame];

        auto signalSemaphores = std::array<VkSemaphore, 1> { m_renderFinishedSemaphores[m_currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores.data();

        result = vkQueueSubmit(m_engine->getGraphicsQueue(), 1, &submitInfo, m_inFlightFences[m_currentFrame]);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        auto presentInfo = VkPresentInfoKHR {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores.data();

        auto swapChains = std::array<VkSwapchainKHR, 1> { m_swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains.data();
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(m_engine->getPresentQueue(), &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_engine->hasFramebufferResized()) {
            m_engine->setFramebufferResized(false);
            this->recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void cleanupSwapChain() {
        for (size_t i = 0; i < m_swapChainFramebuffers.size(); i++) {
            vkDestroyFramebuffer(m_engine->getLogicalDevice(), m_swapChainFramebuffers[i], nullptr);
        }

        for (size_t i = 0; i < m_swapChainImageViews.size(); i++) {
            vkDestroyImageView(m_engine->getLogicalDevice(), m_swapChainImageViews[i], nullptr);
        }

        vkDestroySwapchainKHR(m_engine->getLogicalDevice(), m_swapChain, nullptr);
    }

    void recreateSwapChain() {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(m_engine->getWindow(), &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(m_engine->getWindow(), &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(m_engine->getLogicalDevice());

        this->cleanupSwapChain();
        this->createSwapChain();
        this->createImageViews();
        this->createFramebuffers();
    }
};

int main() {
    App app;

    try {
        app.run();
    } catch (const std::exception& exception) {
        fmt::println(std::cerr, "{}", exception.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
