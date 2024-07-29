#ifndef _ENGINE_H
#define _ENGINE_H

#include <vulkan/vulkan.h>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <optional>
#include <set>
#include <unordered_set>

#include <fmt/core.h>
#include <fmt/ostream.h>

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

}

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
        );
        ~VulkanInstanceSpec() = default;

        const std::vector<std::string>& instanceExtensions() const;

        const std::vector<std::string>& instanceLayers() const;

        VkInstanceCreateFlags instanceCreateFlags() const;

        const std::string& applicationName() const;

        const std::string& engineName() const;

        bool areValidationLayersEnabled() const;
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
        explicit InstanceSpecProvider(bool enableValidationLayers, bool enableDebuggingExtensions);

        ~InstanceSpecProvider();

        VulkanInstanceSpec createInstanceSpec() const;
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

        VkInstanceCreateFlags minInstanceCreateFlags() const;

        std::vector<std::string> getWindowSystemInstanceRequirements() const;

        std::vector<std::string> getInstanceExtensions() const;

        std::vector<std::string> getInstanceLayers() const;
};

class SystemFactory final {
    public:
        explicit SystemFactory() = default;

        VkInstance create(const VulkanInstanceSpec& instanceSpec);
    private:
        std::unique_ptr<PlatformInfoProvider> m_infoProvider;

        static std::vector<const char*> convertToCStrings(const std::vector<std::string>& strings);
};

class PhysicalDeviceSpec final {
    public:
        explicit PhysicalDeviceSpec() = default;
        explicit PhysicalDeviceSpec(const std::vector<std::string>& requiredExtensions, bool hasGraphicsFamily, bool hasPresentFamily);

        ~PhysicalDeviceSpec() = default;

        const std::vector<std::string>& requiredExtensions() const;

        bool hasGraphicsFamily() const;

        bool hasPresentFamily() const;
    private:
        std::vector<std::string> m_requiredExtensions;
        bool m_hasGraphicsFamily;
        bool m_hasPresentFamily;
};

class PhysicalDeviceSpecProvider final {
    public:
        explicit PhysicalDeviceSpecProvider() = default;

        PhysicalDeviceSpec createPhysicalDeviceSpec() const;
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

        std::vector<std::string> getPhysicalDeviceRequirements() const;
};

class PhysicalDeviceSelector final {
    public:
        explicit PhysicalDeviceSelector(VkInstance instance, std::unique_ptr<PlatformInfoProvider> infoProvider);

        ~PhysicalDeviceSelector();

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const;

        bool checkDeviceExtensionSupport(VkPhysicalDevice physicalDevice, const std::vector<std::string>& requiredExtensions) const;

        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const;

        bool isPhysicalDeviceCompatible(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, const PhysicalDeviceSpec& physicalDeviceSpec) const;

        std::vector<VkPhysicalDevice> findAllPhysicalDevices() const;

        std::vector<VkPhysicalDevice> findCompatiblePhysicalDevices(VkSurfaceKHR surface, const PhysicalDeviceSpec& physicalDeviceSpec) const;

        VkPhysicalDevice selectPhysicalDeviceForSurface(VkSurfaceKHR surface, const PhysicalDeviceSpec& physicalDeviceSpec) const;
    private:
        VkInstance m_instance;
        std::unique_ptr<PlatformInfoProvider> m_infoProvider;
};

class LogicalDeviceSpec final {
    public:
        explicit LogicalDeviceSpec() = default;
        explicit LogicalDeviceSpec(const std::vector<std::string>& requiredExtensions, bool requireSamplerAnisotropy);
        
        ~LogicalDeviceSpec() = default;

        const std::vector<std::string>& requiredExtensions() const;

        bool requireSamplerAnisotropy() const;
    private:
        std::vector<std::string> m_requiredExtensions;
        bool m_requireSamplerAnisotropy;
};

class LogicalDeviceSpecProvider final {
    public:
        explicit LogicalDeviceSpecProvider() = default;
        explicit LogicalDeviceSpecProvider(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);

        ~LogicalDeviceSpecProvider();

        LogicalDeviceSpec createLogicalDeviceSpec() const;
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

        std::vector<std::string> getLogicalDeviceRequirements() const;
};

class LogicalDeviceFactory final {
    public:
        explicit LogicalDeviceFactory(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, std::unique_ptr<PlatformInfoProvider> infoProvider);

        ~LogicalDeviceFactory();

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const;

        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const;

        std::tuple<VkDevice, VkQueue, VkQueue, VkQueue> createLogicalDevice(const LogicalDeviceSpec& logicalDeviceSpec);
    private:
        VkPhysicalDevice m_physicalDevice;
        VkSurfaceKHR m_surface;
        std::unique_ptr<PlatformInfoProvider> m_infoProvider;

        static std::vector<const char*> convertToCStrings(const std::vector<std::string>& strings);
};

class VulkanDebugMessenger final {
    public:
        explicit VulkanDebugMessenger();

        ~VulkanDebugMessenger();

        static std::unique_ptr<VulkanDebugMessenger> create(VkInstance instance);

        static VkResult CreateDebugUtilsMessengerEXT(
            VkInstance instance, 
            const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, 
            const VkAllocationCallbacks* pAllocator, 
            VkDebugUtilsMessengerEXT* pDebugMessenger
        );

        static void DestroyDebugUtilsMessengerEXT(
            VkInstance instance, 
            VkDebugUtilsMessengerEXT debugMessenger, 
            const VkAllocationCallbacks* pAllocator
        );

        static const std::string& messageSeverityToString(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity);

        static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData
        );

        void cleanup();
    private:
        VkInstance m_instance;
        VkDebugUtilsMessengerEXT m_debugMessenger;
};

class SurfaceProvider final {
    public:
        explicit SurfaceProvider() = delete;
        explicit SurfaceProvider(VkInstance instance, GLFWwindow* window);

        ~SurfaceProvider();

        VkSurfaceKHR createSurface();
    private:
        VkInstance m_instance;
        GLFWwindow* m_window;
};

class WindowSystem final {
    public:
        explicit WindowSystem() = default;
        explicit WindowSystem(VkInstance instance);
        
        ~WindowSystem();

        static std::unique_ptr<WindowSystem> create(VkInstance instance);

        void createWindow(uint32_t width, uint32_t height, const std::string& title);

        GLFWwindow* getWindow() const;

        SurfaceProvider createSurfaceProvider();

        bool hasFramebufferResized() const;

        void setFramebufferResized(bool framebufferResized);

        void setWindowTitle(const std::string& title);
    private:
        VkInstance m_instance;
        GLFWwindow* m_window;
        VkSurfaceKHR m_surface;
        VkExtent2D m_windowExtent;
        bool m_framebufferResized;

        static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
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
        );

        ~GpuDevice();

        VkPhysicalDevice getPhysicalDevice() const;

        VkDevice getLogicalDevice() const;

        VkQueue getGraphicsQueue() const;

        VkQueue getComputeQueue() const;

        VkQueue getPresentQueue() const;

        VkCommandPool getCommandPool() const;

        VkSampleCountFlagBits getMsaaSamples() const;

        static VkSampleCountFlagBits getMaxUsableSampleCount(VkPhysicalDevice physicalDevice);

        VkSurfaceKHR createRenderSurface(SurfaceProvider& surfaceProvider);

        VkShaderModule createShaderModuleFromFile(const std::string& fileName);

        VkShaderModule createShaderModule(std::istream& stream);

        VkShaderModule createShaderModule(const std::vector<char>& code);
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

        std::vector<char> loadShader(std::istream& stream);

        std::ifstream openShaderFile(const std::string& fileName);

        std::vector<char> loadShaderFromFile(const std::string& fileName);

        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

class GpuDeviceInitializer final {
    public:
        explicit GpuDeviceInitializer(VkInstance instance);

        ~GpuDeviceInitializer();

        std::unique_ptr<GpuDevice> createGpuDevice();
    private:
        VkInstance m_instance;
        VkSurfaceKHR m_dummySurface;
        VkPhysicalDevice m_physicalDevice;
        VkDevice m_device;
        VkQueue m_graphicsQueue;
        VkQueue m_computeQueue;
        VkQueue m_presentQueue;
        VkCommandPool m_commandPool;

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const;

        void createDummySurface();

        void selectPhysicalDevice();

        void createLogicalDevice();

        void createCommandPool();
};

class Engine final {
    public:
        explicit Engine() = default;
        ~Engine();

        static std::unique_ptr<Engine> createDebugMode();

        static std::unique_ptr<Engine> createReleaseMode();

        VkInstance getInstance() const;

        VkPhysicalDevice getPhysicalDevice() const;

        VkDevice getLogicalDevice() const;

        VkQueue getGraphicsQueue() const;

        VkQueue getComputeQueue() const;

        VkQueue getPresentQueue() const;

        VkCommandPool getCommandPool() const;

        VkSurfaceKHR getSurface() const;

        VkSampleCountFlagBits getMsaaSamples() const;

        GLFWwindow* getWindow() const;

        bool hasFramebufferResized() const;

        void setFramebufferResized(bool framebufferResized);

        bool isInitialized() const;

        void createGLFWLibrary();

        void createInfoProvider();

        void createSystemFactory();

        void createInstance();

        void createWindowSystem();

        void createDebugMessenger();

        void createWindow(uint32_t width, uint32_t height, const std::string& title);

        void createGpuDevice();

        void createRenderSurface();

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const;

        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) const;

        VkShaderModule createShaderModuleFromFile(const std::string& fileName);

        VkShaderModule createShaderModule(std::istream& stream);

        VkShaderModule createShaderModule(std::vector<char>& code);
    private:
        std::unique_ptr<PlatformInfoProvider> m_infoProvider;
        std::unique_ptr<SystemFactory> m_systemFactory;
        VkInstance m_instance;
        std::unique_ptr<VulkanDebugMessenger> m_debugMessenger;
        std::unique_ptr<WindowSystem> m_windowSystem;
        VkSurfaceKHR m_surface;

        std::unique_ptr<GpuDevice> m_gpuDevice;

        bool m_enableValidationLayers; 
        bool m_enableDebuggingExtensions;

        static std::unique_ptr<Engine> create(bool enableDebugging);
};

}

#endif // _ENGINE_H
