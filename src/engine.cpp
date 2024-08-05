#include "engine.h"


using QueueFamilyIndices = VulkanEngine::QueueFamilyIndices;
using SwapChainSupportDetails = VulkanEngine::SwapChainSupportDetails;


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


using VulkanInstanceSpec = VulkanEngine::VulkanInstanceSpec;

VulkanEngine::VulkanInstanceSpec::VulkanInstanceSpec(
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

const std::vector<std::string>& VulkanInstanceSpec::instanceExtensions() const {
    return m_instanceExtensions;
}

const std::vector<std::string>& VulkanInstanceSpec::instanceLayers() const {
    return m_instanceLayers;
}

VkInstanceCreateFlags VulkanInstanceSpec::instanceCreateFlags() const {
    return m_instanceCreateFlags;
}

const std::string& VulkanInstanceSpec::applicationName() const {
    return m_applicationName;
}

const std::string& VulkanInstanceSpec::engineName() const {
    return m_engineName;
}

bool VulkanInstanceSpec::areValidationLayersEnabled() const {
    auto found = std::find(
        m_instanceLayers.begin(), 
        m_instanceLayers.end(), 
        VulkanEngine::Constants::VK_LAYER_KHRONOS_validation
    );
        
    return found != m_instanceLayers.end();
}


using InstanceSpecProvider = VulkanEngine::InstanceSpecProvider;

InstanceSpecProvider::InstanceSpecProvider(bool enableValidationLayers, bool enableDebuggingExtensions)
    : m_enableValidationLayers { enableValidationLayers }
    , m_enableDebuggingExtensions { enableDebuggingExtensions }
{
}

InstanceSpecProvider::~InstanceSpecProvider() {
    m_enableValidationLayers = false;
    m_enableDebuggingExtensions = false;
}

VulkanInstanceSpec InstanceSpecProvider::createInstanceSpec() const {
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

VkInstanceCreateFlags InstanceSpecProvider::minInstanceCreateFlags() const {
    auto instanceCreateFlags = 0;
    if (this->getOperatingSystem() == Platform::Apple) {
        instanceCreateFlags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    return instanceCreateFlags;
}

std::vector<std::string> InstanceSpecProvider::getWindowSystemInstanceRequirements() const {        
    uint32_t requiredExtensionCount = 0;
    const char** requiredExtensionNames = glfwGetRequiredInstanceExtensions(&requiredExtensionCount);
    auto requiredExtensions = std::vector<std::string> {};
    for (int i = 0; i < requiredExtensionCount; i++) {
        requiredExtensions.emplace_back(std::string(requiredExtensionNames[i]));
    }

    return requiredExtensions;
}

std::vector<std::string> InstanceSpecProvider::getInstanceExtensions() const {
    auto instanceExtensions = this->getWindowSystemInstanceRequirements();
    instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    if (m_enableDebuggingExtensions) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

   return instanceExtensions;
}

std::vector<std::string> InstanceSpecProvider::getInstanceLayers() const {
    auto instanceLayers = std::vector<std::string> {};
    if (m_enableValidationLayers) {
        instanceLayers.push_back(VulkanEngine::Constants::VK_LAYER_KHRONOS_validation);
    }

    return instanceLayers;
}


using SystemFactory = VulkanEngine::SystemFactory;


VkInstance SystemFactory::create(const VulkanInstanceSpec& instanceSpec) {
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

std::vector<const char*>  SystemFactory::convertToCStrings(const std::vector<std::string>& strings) {
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


using PhysicalDeviceSpec = VulkanEngine::PhysicalDeviceSpec;

PhysicalDeviceSpec::PhysicalDeviceSpec(const std::vector<std::string>& requiredExtensions, bool hasGraphicsFamily, bool hasPresentFamily)
    : m_requiredExtensions { requiredExtensions }
    , m_hasGraphicsFamily { hasGraphicsFamily }
    , m_hasPresentFamily { hasPresentFamily }
{
}

const std::vector<std::string>& PhysicalDeviceSpec::requiredExtensions() const {
    return m_requiredExtensions;
}

bool PhysicalDeviceSpec::hasGraphicsFamily() const {
    return m_hasGraphicsFamily;
}

bool PhysicalDeviceSpec::hasPresentFamily() const {
    return m_hasPresentFamily;
}


using PhysicalDeviceSpecProvider = VulkanEngine::PhysicalDeviceSpecProvider;

PhysicalDeviceSpec PhysicalDeviceSpecProvider::createPhysicalDeviceSpec() const {
    const auto requiredExtensions = this->getPhysicalDeviceRequirements();

    return PhysicalDeviceSpec { requiredExtensions, true, true };
}

std::vector<std::string> PhysicalDeviceSpecProvider::getPhysicalDeviceRequirements() const {
    auto physicalDeviceExtensions = std::vector<std::string> {};
    if (this->getOperatingSystem() == Platform::Apple) {
        physicalDeviceExtensions.push_back(VulkanEngine::Constants::VK_KHR_portability_subset);
    }

    physicalDeviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    return physicalDeviceExtensions;
}


using PhysicalDeviceSelector = VulkanEngine::PhysicalDeviceSelector;

PhysicalDeviceSelector::PhysicalDeviceSelector(VkInstance instance, std::unique_ptr<PlatformInfoProvider> infoProvider)
    : m_instance { instance }
    , m_infoProvider { std::move(infoProvider) }
{
}

PhysicalDeviceSelector::~PhysicalDeviceSelector() {
    m_instance = VK_NULL_HANDLE;
    m_infoProvider = nullptr;
}

QueueFamilyIndices PhysicalDeviceSelector::findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
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

bool PhysicalDeviceSelector::checkDeviceExtensionSupport(VkPhysicalDevice physicalDevice, const std::vector<std::string>& requiredExtensions) const {
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

SwapChainSupportDetails PhysicalDeviceSelector::querySwapChainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
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

bool PhysicalDeviceSelector::isPhysicalDeviceCompatible(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, const PhysicalDeviceSpec& physicalDeviceSpec) const {
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

std::vector<VkPhysicalDevice> PhysicalDeviceSelector::findAllPhysicalDevices() const {
    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, nullptr);

    auto physicalDevices = std::vector<VkPhysicalDevice> { physicalDeviceCount };
    vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, physicalDevices.data());

    return physicalDevices;
}

std::vector<VkPhysicalDevice> PhysicalDeviceSelector::findCompatiblePhysicalDevices(VkSurfaceKHR surface, const PhysicalDeviceSpec& physicalDeviceSpec) const {
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

VkPhysicalDevice PhysicalDeviceSelector::selectPhysicalDeviceForSurface(VkSurfaceKHR surface, const PhysicalDeviceSpec& physicalDeviceSpec) const {
    const auto physicalDevices = this->findCompatiblePhysicalDevices(surface, physicalDeviceSpec);
    if (physicalDevices.empty()) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    VkPhysicalDevice selectedPhysicalDevice = physicalDevices[0];

    return selectedPhysicalDevice;
}


using LogicalDeviceSpec = VulkanEngine::LogicalDeviceSpec;

LogicalDeviceSpec::LogicalDeviceSpec(const std::vector<std::string>& requiredExtensions, bool requireSamplerAnisotropy)
    : m_requiredExtensions { requiredExtensions }
    , m_requireSamplerAnisotropy { requireSamplerAnisotropy }
{
}

const std::vector<std::string>& LogicalDeviceSpec::requiredExtensions() const {
    return m_requiredExtensions;
}

bool LogicalDeviceSpec::requireSamplerAnisotropy() const {
    return m_requireSamplerAnisotropy;
}


using LogicalDeviceSpecProvider = VulkanEngine::LogicalDeviceSpecProvider;

LogicalDeviceSpecProvider::LogicalDeviceSpecProvider(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) 
    : m_physicalDevice { physicalDevice }
    , m_surface { surface }
{
}

LogicalDeviceSpecProvider::~LogicalDeviceSpecProvider() {
    m_physicalDevice = VK_NULL_HANDLE;
    m_surface = VK_NULL_HANDLE;
}

LogicalDeviceSpec LogicalDeviceSpecProvider::createLogicalDeviceSpec() const {
    auto requiredExtensions = this->getLogicalDeviceRequirements();
    bool requireSamplerAnisotropy = true;

    return LogicalDeviceSpec { requiredExtensions, requireSamplerAnisotropy };
}

std::vector<std::string> LogicalDeviceSpecProvider::getLogicalDeviceRequirements() const {
    auto logicalDeviceExtensions = std::vector<std::string> {};
    if (this->getOperatingSystem() == Platform::Apple) {
        logicalDeviceExtensions.push_back(VulkanEngine::Constants::VK_KHR_portability_subset);
    }

    logicalDeviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    return logicalDeviceExtensions;
}


using LogicalDeviceFactory = VulkanEngine::LogicalDeviceFactory;

LogicalDeviceFactory::LogicalDeviceFactory(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, std::unique_ptr<PlatformInfoProvider> infoProvider)
    : m_physicalDevice { physicalDevice }
    , m_surface { surface }
    , m_infoProvider { std::move(infoProvider) }
{
}

LogicalDeviceFactory::~LogicalDeviceFactory() {
    m_physicalDevice = VK_NULL_HANDLE;
    m_surface = VK_NULL_HANDLE;
    m_infoProvider = nullptr;
}

QueueFamilyIndices LogicalDeviceFactory::findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
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

SwapChainSupportDetails LogicalDeviceFactory::querySwapChainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
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

std::tuple<VkDevice, VkQueue, VkQueue, VkQueue> LogicalDeviceFactory::createLogicalDevice(const LogicalDeviceSpec& logicalDeviceSpec) {
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

std::vector<const char*> LogicalDeviceFactory::convertToCStrings(const std::vector<std::string>& strings) {
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


using VulkanDebugMessenger = VulkanEngine::VulkanDebugMessenger;

VulkanDebugMessenger::VulkanDebugMessenger()
    : m_instance { VK_NULL_HANDLE }
    , m_debugMessenger { VK_NULL_HANDLE }
{
}

VulkanDebugMessenger::~VulkanDebugMessenger() {
    this->cleanup();
}

std::unique_ptr<VulkanDebugMessenger> VulkanDebugMessenger::create(VkInstance instance) {
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

    auto vulkanDebugMessenger = std::make_unique<VulkanDebugMessenger>();
    vulkanDebugMessenger->m_instance = instance;
    vulkanDebugMessenger->m_debugMessenger = debugMessenger;

    return vulkanDebugMessenger;
}

VkResult VulkanDebugMessenger::CreateDebugUtilsMessengerEXT(
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

void VulkanDebugMessenger::DestroyDebugUtilsMessengerEXT(
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

const std::string& VulkanDebugMessenger::messageSeverityToString(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity) {
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

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugMessenger::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData
) {
    const auto messageSeverityString = VulkanDebugMessenger::messageSeverityToString(messageSeverity);
    fmt::println(std::cerr, "[{}] {}", messageSeverityString, pCallbackData->pMessage);

    return VK_FALSE;
}

void VulkanDebugMessenger::cleanup() {
    if (m_instance == VK_NULL_HANDLE) {
        return;
    }

    if (m_debugMessenger != VK_NULL_HANDLE) {
        VulkanDebugMessenger::DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
    }

    m_debugMessenger = VK_NULL_HANDLE;
    m_instance = VK_NULL_HANDLE;
}


using SurfaceProvider = VulkanEngine::SurfaceProvider;

SurfaceProvider::SurfaceProvider(VkInstance instance, GLFWwindow* window)
    : m_instance { instance }
    , m_window { window }
{
}

SurfaceProvider::~SurfaceProvider() {
    m_instance = VK_NULL_HANDLE;
    m_window = nullptr;
}

VkSurfaceKHR SurfaceProvider::createSurface() {
    auto surface = VkSurfaceKHR {};
    const auto result = glfwCreateWindowSurface(m_instance, m_window, nullptr, &surface);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    return surface;
}


using WindowSystem = VulkanEngine::WindowSystem;

WindowSystem::WindowSystem(VkInstance instance) : m_instance { instance } {}

WindowSystem::~WindowSystem() {
    m_framebufferResized = false;
    m_windowExtent = VkExtent2D { 0, 0 };
    m_surface = VK_NULL_HANDLE;

    glfwDestroyWindow(m_window);

    m_instance = VK_NULL_HANDLE;
}

std::unique_ptr<WindowSystem> WindowSystem::create(VkInstance instance) {
    return std::make_unique<WindowSystem>(instance);
}

void WindowSystem::createWindow(uint32_t width, uint32_t height, const std::string& title) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

    auto window = glfwCreateWindow(width, height, title.data(), nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    m_window = window;
    m_windowExtent = VkExtent2D { width, height };
}

GLFWwindow* WindowSystem::getWindow() const {
    return m_window;
}

SurfaceProvider WindowSystem::createSurfaceProvider() {
    return SurfaceProvider { m_instance, m_window};
}

bool WindowSystem::hasFramebufferResized() const {
    return m_framebufferResized;
}

void WindowSystem::setFramebufferResized(bool framebufferResized) {
    m_framebufferResized = framebufferResized;
}

void WindowSystem::setWindowTitle(const std::string& title) {
    glfwSetWindowTitle(m_window, title.data());
}

void WindowSystem::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto windowSystem = reinterpret_cast<WindowSystem*>(glfwGetWindowUserPointer(window));
    windowSystem->m_framebufferResized = true;
    windowSystem->m_windowExtent = VkExtent2D { 
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)
    };
}


using GpuDevice = VulkanEngine::GpuDevice;

GpuDevice::GpuDevice(
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

GpuDevice::~GpuDevice() {
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

VkPhysicalDevice GpuDevice::getPhysicalDevice() const {
    return m_physicalDevice;
}

VkDevice GpuDevice::getLogicalDevice() const {
    return m_device;
}

VkQueue GpuDevice::getGraphicsQueue() const {
    return m_graphicsQueue;
}

VkQueue GpuDevice::getComputeQueue() const {
    return m_computeQueue;
}

VkQueue GpuDevice::getPresentQueue() const {
    return m_presentQueue;
}

VkCommandPool GpuDevice::getCommandPool() const {
    return m_commandPool;
}

VkSampleCountFlagBits GpuDevice::getMsaaSamples() const {
    return m_msaaSamples;
}

VkSampleCountFlagBits GpuDevice::getMaxUsableSampleCount(VkPhysicalDevice physicalDevice) {
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

VkSurfaceKHR GpuDevice::createRenderSurface(SurfaceProvider& surfaceProvider) {
    const auto surface = surfaceProvider.createSurface();

    m_surface = surface;

    return surface;
}

VkShaderModule GpuDevice::createShaderModuleFromFile(const std::string& fileName) {
    const auto shaderCode = this->loadShaderFromFile(fileName);

    return this->createShaderModule(shaderCode);
}

VkShaderModule GpuDevice::createShaderModule(std::istream& stream) {
    const auto shaderCode = this->loadShader(stream);

    return this->createShaderModule(shaderCode);
}

VkShaderModule GpuDevice::createShaderModule(const std::vector<char>& code) {
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

VkShaderModule GpuDevice::createShaderModule(const std::vector<unsigned char>& code) {
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

std::vector<char> GpuDevice::loadShader(std::istream& stream) {
    const size_t shaderSize = static_cast<size_t>(stream.tellg());
    auto buffer = std::vector<char>(shaderSize);

    stream.seekg(0);
    stream.read(buffer.data(), shaderSize);

    return buffer;
}

std::ifstream GpuDevice::openShaderFile(const std::string& fileName) {
    auto file = std::ifstream { fileName, std::ios::ate | std::ios::binary };

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    return file;
}

std::vector<char> GpuDevice::loadShaderFromFile(const std::string& fileName) {
    auto stream = this->openShaderFile(fileName);
    auto shader = this->loadShader(stream);
    stream.close();

    return shader;
}

uint32_t GpuDevice::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    auto memProperties = VkPhysicalDeviceMemoryProperties {};
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}


using GpuDeviceInitializer = VulkanEngine::GpuDeviceInitializer;

GpuDeviceInitializer::GpuDeviceInitializer(VkInstance instance)
    : m_instance { instance }
{
}

GpuDeviceInitializer::~GpuDeviceInitializer() {
    vkDestroySurfaceKHR(m_instance, m_dummySurface, nullptr);

    m_instance = VK_NULL_HANDLE;
}

std::unique_ptr<GpuDevice> GpuDeviceInitializer::createGpuDevice() {
    this->createDummySurface();
    this->selectPhysicalDevice();
    this->createLogicalDevice();
    this->createCommandPool();

    auto gpuDevice = std::make_unique<GpuDevice>(
        m_instance,
        m_physicalDevice,
        m_device,
        m_graphicsQueue,
        m_computeQueue,
        m_presentQueue,
        m_commandPool
    );

    return gpuDevice;
}

QueueFamilyIndices GpuDeviceInitializer::findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
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

void GpuDeviceInitializer::createDummySurface() {
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

void GpuDeviceInitializer::selectPhysicalDevice() {
    const auto physicalDeviceSpecProvider = PhysicalDeviceSpecProvider {};
    const auto physicalDeviceSpec = physicalDeviceSpecProvider.createPhysicalDeviceSpec();
    
    auto infoProvider = std::make_unique<PlatformInfoProvider>();
    const auto physicalDeviceSelector = PhysicalDeviceSelector { m_instance, std::move(infoProvider) };
    
    const auto selectedPhysicalDevice = physicalDeviceSelector.selectPhysicalDeviceForSurface(
        m_dummySurface,
        physicalDeviceSpec
    );

    m_physicalDevice = selectedPhysicalDevice;
}

void GpuDeviceInitializer::createLogicalDevice() {
    const auto logicalDeviceSpecProvider = LogicalDeviceSpecProvider { m_physicalDevice, m_dummySurface };
    const auto logicalDeviceSpec = logicalDeviceSpecProvider.createLogicalDeviceSpec();

    auto infoProvider = std::make_unique<PlatformInfoProvider>();
    auto factory = LogicalDeviceFactory { m_physicalDevice, m_dummySurface, std::move(infoProvider) };
    
    const auto [device, graphicsQueue, computeQueue, presentQueue] = factory.createLogicalDevice(logicalDeviceSpec);

    m_device = device;
    m_graphicsQueue = graphicsQueue;
    m_computeQueue = computeQueue;
    m_presentQueue = presentQueue;
}

void GpuDeviceInitializer::createCommandPool() {
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


using Engine = VulkanEngine::Engine;

Engine::~Engine() {
    m_windowSystem.reset();
    m_gpuDevice.reset();
    m_debugMessenger.reset();

    vkDestroyInstance(m_instance, nullptr);

    m_systemFactory.reset();
    m_infoProvider.reset();

    glfwTerminate();
}

std::unique_ptr<Engine> Engine::createDebugMode() {
    return Engine::create(true);
}

std::unique_ptr<Engine> Engine::createReleaseMode() {
    return Engine::create(false);
}

VkInstance Engine::getInstance() const {
    return m_instance;
}

VkPhysicalDevice Engine::getPhysicalDevice() const {
    return m_gpuDevice->getPhysicalDevice();
}

VkDevice Engine::getLogicalDevice() const {
    return m_gpuDevice->getLogicalDevice();
}

VkQueue Engine::getGraphicsQueue() const {
    return m_gpuDevice->getGraphicsQueue();
}

VkQueue Engine::getComputeQueue() const {
    return m_gpuDevice->getComputeQueue();
}

VkQueue Engine::getPresentQueue() const {
    return m_gpuDevice->getPresentQueue();
}

VkCommandPool Engine::getCommandPool() const {
    return m_gpuDevice->getCommandPool();
}

VkSurfaceKHR Engine::getSurface() const {
    return m_surface;
}

VkSampleCountFlagBits Engine::getMsaaSamples() const {
    return m_gpuDevice->getMsaaSamples();
}

GLFWwindow* Engine::getWindow() const {
    return m_windowSystem->getWindow();
}

bool Engine::hasFramebufferResized() const {
    return m_windowSystem->hasFramebufferResized();
}

void Engine::setFramebufferResized(bool framebufferResized) {
    m_windowSystem->setFramebufferResized(framebufferResized);
}

bool Engine::isInitialized() const {
    return m_instance != VK_NULL_HANDLE;
}

void Engine::createGLFWLibrary() {
    const auto result = glfwInit();
    if (!result) {
        glfwTerminate();

        auto errorMessage = std::string { "Failed to initialize GLFW" };

        throw std::runtime_error { errorMessage };
    }
}

void Engine::createInfoProvider() {
    auto infoProvider = std::make_unique<PlatformInfoProvider>();

    m_infoProvider = std::move(infoProvider);
}

void Engine::createSystemFactory() {
    auto systemFactory = std::make_unique<SystemFactory>();

    m_systemFactory = std::move(systemFactory);
}

void Engine::createInstance() {
    const auto instanceSpecProvider = InstanceSpecProvider { m_enableValidationLayers, m_enableDebuggingExtensions };
    const auto instanceSpec = instanceSpecProvider.createInstanceSpec();
    const auto instance = m_systemFactory->create(instanceSpec);
        
    m_instance = instance;
}

void Engine::createWindowSystem() {
    auto windowSystem = WindowSystem::create(m_instance);

    m_windowSystem = std::move(windowSystem);
}

void Engine::createDebugMessenger() {
    if (!m_enableValidationLayers) {
        return;
    }

    auto debugMessenger = VulkanDebugMessenger::create(m_instance);

    m_debugMessenger = std::move(debugMessenger);
}

void Engine::createWindow(uint32_t width, uint32_t height, const std::string& title) {
    m_windowSystem->createWindow(width, height, title);
       
    this->createRenderSurface();
}

void Engine::createGpuDevice() {
    auto gpuDeviceInitializer = GpuDeviceInitializer { m_instance };
    auto gpuDevice = gpuDeviceInitializer.createGpuDevice();

    m_gpuDevice = std::move(gpuDevice);
}

void Engine::createRenderSurface() {
    auto surfaceProvider = m_windowSystem->createSurfaceProvider();
    const auto surface = m_gpuDevice->createRenderSurface(surfaceProvider);

    m_surface = surface;
}

QueueFamilyIndices Engine::findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
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

SwapChainSupportDetails Engine::querySwapChainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const {
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

VkShaderModule Engine::createShaderModuleFromFile(const std::string& fileName) {
    return m_gpuDevice->createShaderModuleFromFile(fileName);
}

VkShaderModule Engine::createShaderModule(std::istream& stream) {
    return m_gpuDevice->createShaderModule(stream);
}

VkShaderModule Engine::createShaderModule(const std::vector<char>& code) {
    return m_gpuDevice->createShaderModule(code);
}

VkShaderModule Engine::createShaderModule(const std::vector<unsigned char>& code) {
    return m_gpuDevice->createShaderModule(code);
}

std::unique_ptr<Engine> Engine::create(bool enableDebugging) {
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
