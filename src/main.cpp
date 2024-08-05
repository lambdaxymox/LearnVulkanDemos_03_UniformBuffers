#include <vulkan/vulkan.h>

#include "engine.h"

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
#include <unordered_set>

#include <fmt/core.h>
#include <fmt/ostream.h>

#ifndef GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif // GLFW_INCLUDE_VULKAN

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <compile_glsl_shaders/shaders_glsl.h>
#include <compile_hlsl_shaders/shaders_hlsl.h>


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;


using Engine = VulkanEngine::Engine;


struct Vertex {
    glm::vec2 position;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        const auto bindingDescription = VkVertexInputBindingDescription {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        const auto attributeDescriptions = std::array<VkVertexInputAttributeDescription, 2> {
            VkVertexInputAttributeDescription {
                .binding = 0,
                .location = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, position),
            },
            VkVertexInputAttributeDescription {
                .binding = 0,
                .location = 1,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, color),
            },
        };

        return attributeDescriptions;
    }
};

class Mesh final {
    public:
        explicit Mesh() = default;
        explicit Mesh(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices)
            : m_vertices { vertices }
            , m_indices { indices }
        {
        }

        explicit Mesh(std::vector<Vertex>&& vertices, std::vector<uint16_t>&& indices)
            : m_vertices { vertices }
            , m_indices { indices }
        {
        }

        const std::vector<Vertex>& vertices() const {
            return m_vertices;
        }

        const std::vector<uint16_t>& indices() const {
            return m_indices;
        }
    private:
        std::vector<Vertex> m_vertices;
        std::vector<uint16_t> m_indices;
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

class App final {
    public:
        explicit App() = default;

        ~App() {
            this->cleanup();
        }

        void run() {
            this->initApp();
            this->mainLoop();
        }
    private:
        std::unique_ptr<Engine> m_engine;

        std::unordered_map<std::string, std::vector<uint8_t>> m_glslShaders;
        std::unordered_map<std::string, std::vector<uint8_t>> m_hlslShaders;

        Mesh m_mesh;
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
            }
        }

        void createEngine() {
            auto engine = Engine::createDebugMode();
            engine->createWindow(WIDTH, HEIGHT, "Uniform Buffers");

            m_engine = std::move(engine);
        }

        void createShaderBinaries() {
            const auto glslShaders = shaders_glsl::createGlslShaders();
            const auto hlslShaders = shaders_hlsl::createHlslShaders();

            m_glslShaders = std::move(glslShaders);
            m_hlslShaders = std::move(hlslShaders);
        }

        void initApp() {
            this->createEngine();

            this->createShaderBinaries();

            this->createMesh();
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
            this->createRenderingSyncObjects();
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

        std::tuple<VkBuffer, VkDeviceMemory> createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
            const auto bufferInfo = VkBufferCreateInfo {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = size,
                .usage = usage,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };

            auto buffer = VkBuffer {};
            const auto resultCreateBuffer = vkCreateBuffer(m_engine->getLogicalDevice(), &bufferInfo, nullptr, &buffer);
            if (resultCreateBuffer != VK_SUCCESS) {
                throw std::runtime_error("failed to create buffer!");
            }

            auto memRequirements = VkMemoryRequirements {};
            vkGetBufferMemoryRequirements(m_engine->getLogicalDevice(), buffer, &memRequirements);

            const auto allocInfo = VkMemoryAllocateInfo {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = memRequirements.size,
                .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties),
            };

            auto bufferMemory = VkDeviceMemory {};
            const auto resultAllocateMemory = vkAllocateMemory(m_engine->getLogicalDevice(), &allocInfo, nullptr, &bufferMemory);
            if (resultAllocateMemory != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate buffer memory!");
            }

            vkBindBufferMemory(m_engine->getLogicalDevice(), buffer, bufferMemory, 0);

            return std::make_tuple(buffer, bufferMemory);
        }

        void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
            const auto allocInfo = VkCommandBufferAllocateInfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandPool = m_engine->getCommandPool(),
                .commandBufferCount = 1,
            };

            auto commandBuffer = VkCommandBuffer {};
            vkAllocateCommandBuffers(m_engine->getLogicalDevice(), &allocInfo, &commandBuffer);

            const auto beginInfo = VkCommandBufferBeginInfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            vkBeginCommandBuffer(commandBuffer, &beginInfo);

            const auto copyRegion = VkBufferCopy {
                .size = size,
            };
            vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

            vkEndCommandBuffer(commandBuffer);

            const auto submitInfo = VkSubmitInfo {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers = &commandBuffer,
            };

            vkQueueSubmit(m_engine->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(m_engine->getGraphicsQueue());

            vkFreeCommandBuffers(m_engine->getLogicalDevice(), m_engine->getCommandPool(), 1, &commandBuffer);
        }

        void createMesh() {
            auto vertices = std::vector<Vertex> {
                Vertex { { -0.5f, -0.5f}, { 1.0f, 0.0f, 0.0f } },
                Vertex { {  0.5f, -0.5f}, { 0.0f, 1.0f, 0.0f } },
                Vertex { {  0.5f,  0.5f}, { 0.0f, 0.0f, 1.0f } },
                Vertex { { -0.5f,  0.5f}, { 1.0f, 1.0f, 1.0f } }
            };
            auto indices = std::vector<uint16_t> {
                0, 1, 2, 2, 3, 0
            };
            const auto mesh = Mesh { std::move(vertices), std::move(indices) };

            m_mesh = std::move(mesh);
        }

        void createVertexBuffer() {
            const auto bufferSize = VkDeviceSize { sizeof(m_mesh.vertices()[0]) * m_mesh.vertices().size() };
            VkBufferUsageFlags stagingBufferUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VkMemoryPropertyFlags stagingBufferPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            auto [stagingBuffer, stagingBufferMemory] = this->createBuffer(
                bufferSize,
                stagingBufferUsageFlags, 
                stagingBufferPropertyFlags
            );

            void* data;
            vkMapMemory(m_engine->getLogicalDevice(), stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, m_mesh.vertices().data(), (size_t) bufferSize);
            vkUnmapMemory(m_engine->getLogicalDevice(), stagingBufferMemory);

            VkBufferUsageFlags vertexBufferUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            VkMemoryPropertyFlags vertexBufferPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            const auto [vertexBuffer, vertexBufferMemory] = this->createBuffer(
                bufferSize,
                vertexBufferUsageFlags,
                vertexBufferPropertyFlags
            );

            this->copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

            vkDestroyBuffer(m_engine->getLogicalDevice(), stagingBuffer, nullptr);
            vkFreeMemory(m_engine->getLogicalDevice(), stagingBufferMemory, nullptr);

            m_vertexBuffer = vertexBuffer;
            m_vertexBufferMemory = vertexBufferMemory;
        }

        void createIndexBuffer() {
            const auto bufferSize = VkDeviceSize { sizeof(m_mesh.indices()[0]) * m_mesh.indices().size() };
            VkBufferUsageFlags stagingBufferUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VkMemoryPropertyFlags stagingBufferPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            const auto [stagingBuffer, stagingBufferMemory] = this->createBuffer(
                bufferSize,
                stagingBufferUsageFlags,
                stagingBufferPropertyFlags
            );

            void* data;
            vkMapMemory(m_engine->getLogicalDevice(), stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, m_mesh.indices().data(), (size_t) bufferSize);
            vkUnmapMemory(m_engine->getLogicalDevice(), stagingBufferMemory);

            VkBufferUsageFlags vertexBufferUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            VkMemoryPropertyFlags vertexBufferPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            const auto [indexBuffer, indexBufferMemory] = this->createBuffer(bufferSize, vertexBufferUsageFlags, vertexBufferPropertyFlags);

            this->copyBuffer(stagingBuffer, indexBuffer, bufferSize);

            vkDestroyBuffer(m_engine->getLogicalDevice(), stagingBuffer, nullptr);
            vkFreeMemory(m_engine->getLogicalDevice(), stagingBufferMemory, nullptr);

            m_indexBuffer = indexBuffer;
            m_indexBufferMemory = indexBufferMemory;
        }

        void createDescriptorSetLayout() {
            const auto uboLayoutBinding = VkDescriptorSetLayoutBinding {
                .binding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pImmutableSamplers = nullptr,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            };
            const auto layoutInfo = VkDescriptorSetLayoutCreateInfo {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 1,
                .pBindings = &uboLayoutBinding,
            };

            auto descriptorSetLayout = VkDescriptorSetLayout {};
            const auto result = vkCreateDescriptorSetLayout(m_engine->getLogicalDevice(), &layoutInfo, nullptr, &descriptorSetLayout);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create descriptor set layout!");
            }

            m_descriptorSetLayout = descriptorSetLayout;
        }

        void createUniformBuffers() {
            const auto bufferSize = VkDeviceSize { sizeof(UniformBufferObject) };
            auto uniformBuffers = std::vector<VkBuffer> { MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE };
            auto uniformBuffersMemory = std::vector<VkDeviceMemory> { MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE };
            auto uniformBuffersMapped = std::vector<void*> { MAX_FRAMES_IN_FLIGHT, nullptr };

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                VkMemoryPropertyFlags propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            
                const auto [uniformBuffer, uniformBufferMemory] = this->createBuffer(bufferSize, usageFlags, propertyFlags);
                void* uniformBufferMapped;      
                vkMapMemory(m_engine->getLogicalDevice(), uniformBufferMemory, 0, bufferSize, 0, &uniformBufferMapped);

                uniformBuffers[i] = uniformBuffer;
                uniformBuffersMemory[i] = uniformBufferMemory;
                uniformBuffersMapped[i] = uniformBufferMapped;
            }

            m_uniformBuffers = std::move(uniformBuffers);
            m_uniformBuffersMemory = std::move(uniformBuffersMemory);
            m_uniformBuffersMapped = std::move(uniformBuffersMapped);
        }

        void createDescriptorPool() {
            const auto poolSize = VkDescriptorPoolSize {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            };
            const auto poolInfo = VkDescriptorPoolCreateInfo {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .poolSizeCount = 1,
                .pPoolSizes = &poolSize,
                .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            };

            auto descriptorPool = VkDescriptorPool {};
            const auto result = vkCreateDescriptorPool(m_engine->getLogicalDevice(), &poolInfo, nullptr, &descriptorPool);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create descriptor pool!");
            }

            m_descriptorPool = descriptorPool;
        }

        void createDescriptorSets() {
            auto layouts = std::vector<VkDescriptorSetLayout> { MAX_FRAMES_IN_FLIGHT, m_descriptorSetLayout };
        
            const auto allocInfo = VkDescriptorSetAllocateInfo {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = m_descriptorPool,
                .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
                .pSetLayouts = layouts.data(),
            };

            auto descriptorSets = std::vector<VkDescriptorSet> { MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE };
            auto result = vkAllocateDescriptorSets(m_engine->getLogicalDevice(), &allocInfo, descriptorSets.data());
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate descriptor sets!");
            }

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                const auto bufferInfo = VkDescriptorBufferInfo {
                    .buffer = m_uniformBuffers[i],
                    .offset = 0,
                    .range = sizeof(UniformBufferObject),
                };

                const auto descriptorWrite = VkWriteDescriptorSet {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .pBufferInfo = &bufferInfo,
                    .pImageInfo = nullptr,       // Optional
                    .pTexelBufferView = nullptr, // Optional
                };

                vkUpdateDescriptorSets(m_engine->getLogicalDevice(), 1, &descriptorWrite, 0, nullptr);
            }

            m_descriptorSets = std::move(descriptorSets);
        }

        void createCommandBuffers() {
            auto commandBuffers = std::vector<VkCommandBuffer> { MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE };
        
            const auto allocInfo = VkCommandBufferAllocateInfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = m_engine->getCommandPool(),
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = static_cast<uint32_t>(commandBuffers.size()),
            };

            const auto result = vkAllocateCommandBuffers(m_engine->getLogicalDevice(), &allocInfo, commandBuffers.data());
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate command buffers!");
            }

            m_commandBuffers = std::move(commandBuffers);
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
                int _width, _height;
                glfwGetWindowSize(m_engine->getWindow(), &_width, &_height);

                const uint32_t width = std::clamp(
                    static_cast<uint32_t>(_width),
                    capabilities.minImageExtent.width,
                    capabilities.maxImageExtent.width
                );
                const uint32_t height = std::clamp(
                    static_cast<uint32_t>(_height), 
                    capabilities.minImageExtent.height, 
                    capabilities.maxImageExtent.height
                );
                const auto actualExtent = VkExtent2D {
                    .width = width,
                    .height = height,
                };

                return actualExtent;
            }
        }

        void createSwapChain() {
            const auto swapChainSupport = m_engine->querySwapChainSupport(m_engine->getPhysicalDevice(), m_engine->getSurface());
            const auto surfaceFormat = this->selectSwapSurfaceFormat(swapChainSupport.formats);
            const auto presentMode = this->selectSwapPresentMode(swapChainSupport.presentModes);
            const auto extent = this->selectSwapExtent(swapChainSupport.capabilities);
            const auto imageCount = [&swapChainSupport]() {
                uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
                if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
                    return swapChainSupport.capabilities.maxImageCount;
                }

                return imageCount;
            }();
            const auto indices = m_engine->findQueueFamilies(m_engine->getPhysicalDevice(), m_engine->getSurface());
            auto queueFamilyIndices = std::array<uint32_t, 2> { 
                indices.graphicsAndComputeFamily.value(),
                indices.presentFamily.value()
            };
            const auto imageSharingMode = [&indices]() -> VkSharingMode {
                if (indices.graphicsAndComputeFamily != indices.presentFamily) {
                    return VK_SHARING_MODE_CONCURRENT;
                } else {
                    return VK_SHARING_MODE_EXCLUSIVE;
                }
            }();
            const auto [queueFamilyIndicesPtr, queueFamilyIndexCount] = [&indices, &queueFamilyIndices]() -> std::tuple<const uint32_t*, uint32_t> {
                if (indices.graphicsAndComputeFamily != indices.presentFamily) {
                    const auto data = queueFamilyIndices.data();
                    const auto size = static_cast<uint32_t>(queueFamilyIndices.size());
                
                    return std::make_tuple(data, size);
                } else {
                    const auto data = static_cast<uint32_t*>(nullptr);
                    const auto size = static_cast<uint32_t>(0);

                    return std::make_tuple(data, size);
                }
            }();
            
            const auto createInfo = VkSwapchainCreateInfoKHR {
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                .surface = m_engine->getSurface(),
                .minImageCount = imageCount,
                .imageFormat = surfaceFormat.format,
                .imageColorSpace = surfaceFormat.colorSpace,
                .imageExtent = extent,
                .imageArrayLayers = 1,
                .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                .imageSharingMode = imageSharingMode,
                .queueFamilyIndexCount = queueFamilyIndexCount,
                .pQueueFamilyIndices = queueFamilyIndices.data(),
                .preTransform = swapChainSupport.capabilities.currentTransform,
                .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                .presentMode = presentMode,
                .clipped = VK_TRUE,
                .oldSwapchain = VK_NULL_HANDLE,
            };

            auto swapChain = VkSwapchainKHR {};
            const auto result = vkCreateSwapchainKHR(m_engine->getLogicalDevice(), &createInfo, nullptr, &swapChain);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create swap chain!");
            }

            uint32_t swapChainImageCount = 0;
            vkGetSwapchainImagesKHR(m_engine->getLogicalDevice(), swapChain, &swapChainImageCount, nullptr);
        
            auto swapChainImages = std::vector<VkImage> { swapChainImageCount, VK_NULL_HANDLE };
            vkGetSwapchainImagesKHR(m_engine->getLogicalDevice(), swapChain, &swapChainImageCount, swapChainImages.data());

            m_swapChain = swapChain;
            m_swapChainImages = std::move(swapChainImages);
            m_swapChainImageFormat = surfaceFormat.format;
            m_swapChainExtent = extent;
        }

        void createImageViews() {
            auto swapChainImageViews = std::vector<VkImageView> { m_swapChainImages.size(), VK_NULL_HANDLE };
            for (size_t i = 0; i < m_swapChainImages.size(); i++) {
                const auto createInfo = VkImageViewCreateInfo {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .image = m_swapChainImages[i],
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = m_swapChainImageFormat,
                    .components.r = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .components.g = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .components.b = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .components.a = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .subresourceRange.baseMipLevel = 0,
                    .subresourceRange.levelCount = 1,
                    .subresourceRange.baseArrayLayer = 0,
                    .subresourceRange.layerCount = 1,
                };

                auto swapChainImageView = VkImageView {};
                const auto result = vkCreateImageView(m_engine->getLogicalDevice(), &createInfo, nullptr, &swapChainImageView);
                if (result != VK_SUCCESS) {
                    throw std::runtime_error("Failed to create image views!");
                }

                swapChainImageViews[i] = swapChainImageView;
            }

            m_swapChainImageViews = std::move(swapChainImageViews);
        }

        void createRenderPass() {
            const auto colorAttachment = VkAttachmentDescription {
                .format = m_swapChainImageFormat,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            };
            const auto colorAttachmentRef = VkAttachmentReference {
                .attachment = 0,
                .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            };
            const auto subpass = VkSubpassDescription {
                .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                .colorAttachmentCount = 1,
                .pColorAttachments = &colorAttachmentRef,
            };

            const auto renderPassInfo = VkRenderPassCreateInfo {
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                .attachmentCount = 1,
                .pAttachments = &colorAttachment,
                .subpassCount = 1,
                .pSubpasses = &subpass,
            };

            auto renderPass = VkRenderPass {};
            const auto result = vkCreateRenderPass(m_engine->getLogicalDevice(), &renderPassInfo, nullptr, &renderPass);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create render pass!");
            }

            m_renderPass = renderPass;
        }

        void createGraphicsPipeline() {
            const auto vertexShaderModule = m_engine->createShaderModule(m_hlslShaders.at("shader.vert.hlsl"));
            const auto fragmentShaderModule = m_engine->createShaderModule(m_hlslShaders.at("shader.frag.hlsl"));

            const auto vertexShaderStageInfo = VkPipelineShaderStageCreateInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vertexShaderModule,
                .pName = "main",
            };
            const auto fragmentShaderStageInfo = VkPipelineShaderStageCreateInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = fragmentShaderModule,
                .pName = "main",
            };
            const auto shaderStages = std::array<VkPipelineShaderStageCreateInfo, 2> {
                vertexShaderStageInfo,
                fragmentShaderStageInfo
            };
            auto bindingDescription = Vertex::getBindingDescription();
            auto attributeDescriptions = Vertex::getAttributeDescriptions();
            const auto vertexInputInfo = VkPipelineVertexInputStateCreateInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
                .pVertexBindingDescriptions = &bindingDescription,
                .pVertexAttributeDescriptions = attributeDescriptions.data(),
            };
            const auto inputAssembly = VkPipelineInputAssemblyStateCreateInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                .primitiveRestartEnable = VK_FALSE,
            };

            // Without dynamic state, the viewport and scissor rectangle need to be set 
            // in the pipeline using the `VkPipelineViewportStateCreateInfo` struct. This
            // makes the viewport and scissor rectangle for this pipeline immutable.
            // Any changes to these values would require a new pipeline to be created with
            // the new values.
            // ```
            // const auto viewportState = VkPipelineViewportStateCreateInfo {
            //     .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            //     .viewportCount = 1,
            //     .pViewports = &viewport,
            //     .scissorCount = 1,
            //     .pScissors = &scissor,
            // };
            // ```
            const auto viewportState = VkPipelineViewportStateCreateInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1,
                .scissorCount = 1,
            };
            const auto rasterizer = VkPipelineRasterizationStateCreateInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .lineWidth = 1.0f,
                .cullMode = VK_CULL_MODE_BACK_BIT,
                // .frontFace = VK_FRONT_FACE_CLOCKWISE,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .depthBiasEnable = VK_FALSE,
            };
            const auto multisampling = VkPipelineMultisampleStateCreateInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .sampleShadingEnable = VK_FALSE,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                .pSampleMask = nullptr,            // Optional.
                .alphaToCoverageEnable = VK_FALSE, // Optional.
                .alphaToOneEnable = VK_FALSE,      // Optional.
            };
            const auto depthStencil = VkPipelineDepthStencilStateCreateInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE,
            };
            const auto colorBlendAttachment = VkPipelineColorBlendAttachmentState {
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
                .blendEnable = VK_FALSE,
                .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,  // Optional
                .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
                .colorBlendOp = VK_BLEND_OP_ADD,             // Optional
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,  // Optional
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
                .alphaBlendOp = VK_BLEND_OP_ADD,             // Optional
                // // Alpha blending:
                // // finalColor.rgb = newAlpha * newColor + (1 - newAlpha) * oldColor,
                // // finalColor.a = newAlpha.a,
                // .blendEnable = VK_TRUE,
                // .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                // .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                // .colorBlendOp = VK_BLEND_OP_ADD,
                // .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                // .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                // .alphaBlendOp = VK_BLEND_OP_ADD,
            };
            const auto colorBlending = VkPipelineColorBlendStateCreateInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .logicOpEnable = VK_FALSE,
                .logicOp = VK_LOGIC_OP_COPY,
                .attachmentCount = 1,
                .pAttachments = &colorBlendAttachment,
                .blendConstants[0] = 0.0f,
                .blendConstants[1] = 0.0f,
                .blendConstants[2] = 0.0f,
                .blendConstants[3] = 0.0f,
            };

            const auto dynamicStates = std::vector<VkDynamicState> {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR
            };
            const auto dynamicState = VkPipelineDynamicStateCreateInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
                .pDynamicStates = dynamicStates.data(),
            };

            const auto pipelineLayoutInfo = VkPipelineLayoutCreateInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts = &m_descriptorSetLayout,
                .pushConstantRangeCount = 0,    // Optional
                .pPushConstantRanges = nullptr, // Optional
            };

            auto pipelineLayout = VkPipelineLayout {};
            const auto resultCreatePipelineLayout = vkCreatePipelineLayout(m_engine->getLogicalDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout);
            if (resultCreatePipelineLayout != VK_SUCCESS) {
                throw std::runtime_error("failed to create pipeline layout!");
            }

            const auto pipelineInfo = VkGraphicsPipelineCreateInfo {
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = 2,
                .pStages = shaderStages.data(),
                .pVertexInputState = &vertexInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depthStencil,
                .pColorBlendState = &colorBlending,
                .pDynamicState = &dynamicState,
                .layout = pipelineLayout,
                .renderPass = m_renderPass,
                .subpass = 0,
                .basePipelineHandle = VK_NULL_HANDLE, // Optional
                .basePipelineIndex = -1,              // Optional
            };

            auto graphicsPipeline = VkPipeline {};
            const auto resultCreateGraphicsPipeline = vkCreateGraphicsPipelines(
                m_engine->getLogicalDevice(), 
                VK_NULL_HANDLE, 
                1, 
                &pipelineInfo, 
                nullptr, 
                &graphicsPipeline
            );
        
            if (resultCreateGraphicsPipeline != VK_SUCCESS) {
                throw std::runtime_error("failed to create graphics pipeline!");
            }

            /*
            vkDestroyShaderModule(m_engine->getLogicalDevice(), fragmentShaderModule, nullptr);
            vkDestroyShaderModule(m_engine->getLogicalDevice(), vertexShaderModule, nullptr);
            */

            m_pipelineLayout = pipelineLayout;
            m_graphicsPipeline = graphicsPipeline;
        }

        void createFramebuffers() {
            auto swapChainFramebuffers = std::vector<VkFramebuffer> { m_swapChainImageViews.size(), VK_NULL_HANDLE };
            for (size_t i = 0; i < m_swapChainImageViews.size(); i++) {
                const auto attachments = std::array<VkImageView, 1> {
                    m_swapChainImageViews[i]
                };

                const auto framebufferInfo = VkFramebufferCreateInfo {
                    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    .renderPass = m_renderPass,
                    .attachmentCount = static_cast<uint32_t>(attachments.size()),
                    .pAttachments = attachments.data(),
                    .width = m_swapChainExtent.width,
                    .height = m_swapChainExtent.height,
                    .layers = 1,
                };

                auto swapChainFramebuffer = VkFramebuffer {};
                const auto result = vkCreateFramebuffer(
                    m_engine->getLogicalDevice(),
                    &framebufferInfo,
                    nullptr,
                    &swapChainFramebuffer
                );

                if (result != VK_SUCCESS) {
                    throw std::runtime_error("failed to create framebuffer!");
                }

                swapChainFramebuffers[i] = swapChainFramebuffer;
            }

            m_swapChainFramebuffers = std::move(swapChainFramebuffers);
        }

        void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
            const auto beginInfo = VkCommandBufferBeginInfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = 0,                  // Optional.
                .pInheritanceInfo = nullptr, // Optional.
            };

            const auto resultBeginCommandBuffer = vkBeginCommandBuffer(commandBuffer, &beginInfo);
            if (resultBeginCommandBuffer != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            // NOTE: The order of `clearValues` should be identical to the order of the attachments
            // in the render pass.
            const auto clearValues = std::array<VkClearValue, 1> {
                VkClearValue { .color = VkClearColorValue { { 0.0f, 0.0f, 0.0f, 1.0f } } },
            };

            const auto renderPassInfo = VkRenderPassBeginInfo {
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = m_renderPass,
                .framebuffer = m_swapChainFramebuffers[imageIndex],
                .renderArea.offset = VkOffset2D { 0, 0 },
                .renderArea.extent = m_swapChainExtent,
                .clearValueCount = static_cast<uint32_t>(clearValues.size()),
                .pClearValues = clearValues.data(),
            };

            vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

            const auto viewport = VkViewport {
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(m_swapChainExtent.width),
                .height = static_cast<float>(m_swapChainExtent.height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            const auto scissor = VkRect2D {
                .offset = VkOffset2D { 0, 0 },
                .extent = m_swapChainExtent,
            };
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            const auto vertexBuffers = std::array<VkBuffer, 1> { m_vertexBuffer };
            const auto offsets = std::array<VkDeviceSize, 1> { 0 };
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
        
            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(m_mesh.indices().size()), 1, 0, 0, 0);

            vkCmdEndRenderPass(commandBuffer);

            const auto resultEndCommandBuffer = vkEndCommandBuffer(commandBuffer);
            if (resultEndCommandBuffer != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }

        void createRenderingSyncObjects() {
            auto imageAvailableSemaphores = std::vector<VkSemaphore> { MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE };
            auto renderFinishedSemaphores = std::vector<VkSemaphore> { MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE };
            auto inFlightFences = std::vector<VkFence> { MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE };

            const auto semaphoreInfo = VkSemaphoreCreateInfo {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            };

            const auto fenceInfo = VkFenceCreateInfo {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .flags = VK_FENCE_CREATE_SIGNALED_BIT,
            };

            for (size_t i = 0; i < imageAvailableSemaphores.size(); i++) {
                const auto result = vkCreateSemaphore(m_engine->getLogicalDevice(), &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]);
                if (result != VK_SUCCESS) {
                    throw std::runtime_error("failed to create image-available semaphore synchronization object");
                }
            }

            for (size_t i = 0; i < renderFinishedSemaphores.size(); i++) {
                const auto result = vkCreateSemaphore(m_engine->getLogicalDevice(), &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]);
                if (result != VK_SUCCESS) {
                    throw std::runtime_error("failed to create render-finished semaphore synchronization object");
                }
            }

            for (size_t i = 0; i < inFlightFences.size(); i++) {
                const auto result = vkCreateFence(m_engine->getLogicalDevice(), &fenceInfo, nullptr, &inFlightFences[i]);
                if (result != VK_SUCCESS) {
                    throw std::runtime_error("failed to create in-flight fence synchronization object");
                }
            }

            m_imageAvailableSemaphores = std::move(imageAvailableSemaphores);
            m_renderFinishedSemaphores = std::move(renderFinishedSemaphores);
            m_inFlightFences = std::move(inFlightFences);
        }

        void updateUniformBuffer(uint32_t currentImage) {
            static auto startTime = std::chrono::high_resolution_clock::now();

            auto currentTime = std::chrono::high_resolution_clock::now();
            float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

            auto ubo = UniformBufferObject {
                .model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
                .view = glm::lookAt(
                    glm::vec3(2.0f, 2.0f, 2.0f),
                    glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f)
                ),
                .proj = glm::perspective(
                    glm::radians(45.0f),
                    m_swapChainExtent.width / (float) m_swapChainExtent.height,
                    0.1f,
                    10.0f
                ),
            };
            ubo.proj[1][1] *= -1;

            memcpy(m_uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
        }

        void draw() {
            vkWaitForFences(m_engine->getLogicalDevice(), 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

            uint32_t imageIndex;
            const auto resultAcquireNextImageKHR = vkAcquireNextImageKHR(
                m_engine->getLogicalDevice(), 
                m_swapChain, 
                UINT64_MAX, 
                m_imageAvailableSemaphores[m_currentFrame], 
                VK_NULL_HANDLE, 
                &imageIndex
            );

            if (resultAcquireNextImageKHR == VK_ERROR_OUT_OF_DATE_KHR) {
                this->recreateSwapChain();
                return;
            } else if (resultAcquireNextImageKHR != VK_SUCCESS && resultAcquireNextImageKHR != VK_SUBOPTIMAL_KHR) {
                throw std::runtime_error("failed to acquire swap chain image!");
            }

            this->updateUniformBuffer(m_currentFrame);

            vkResetFences(m_engine->getLogicalDevice(), 1, &m_inFlightFences[m_currentFrame]);

            vkResetCommandBuffer(m_commandBuffers[m_currentFrame], /* VkCommandBufferResetFlagBits */ 0);
            this->recordCommandBuffer(m_commandBuffers[m_currentFrame], imageIndex);

            auto waitSemaphores = std::array<VkSemaphore, 1> { m_imageAvailableSemaphores[m_currentFrame] };
            auto waitStages = std::array<VkPipelineStageFlags, 1> { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
            auto signalSemaphores = std::array<VkSemaphore, 1> { m_renderFinishedSemaphores[m_currentFrame] };

            const auto submitInfo = VkSubmitInfo {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = waitSemaphores.data(),
                .pWaitDstStageMask = waitStages.data(),
                .commandBufferCount = 1,
                .pCommandBuffers = &m_commandBuffers[m_currentFrame],
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = signalSemaphores.data(),
            };

            const auto resultQueueSubmit = vkQueueSubmit(m_engine->getGraphicsQueue(), 1, &submitInfo, m_inFlightFences[m_currentFrame]);
            if (resultQueueSubmit != VK_SUCCESS) {
                throw std::runtime_error("failed to submit draw command buffer!");
            }

            const auto swapChains = std::array<VkSwapchainKHR, 1> { m_swapChain };

            const auto presentInfo = VkPresentInfoKHR {
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = signalSemaphores.data(),
                .swapchainCount = 1,
                .pSwapchains = swapChains.data(),
                .pImageIndices = &imageIndex,
            };

            const auto resultQueuePresentKHR = vkQueuePresentKHR(m_engine->getPresentQueue(), &presentInfo);
            if (resultQueuePresentKHR == VK_ERROR_OUT_OF_DATE_KHR || resultQueuePresentKHR == VK_SUBOPTIMAL_KHR || m_engine->hasFramebufferResized()) {
                m_engine->setFramebufferResized(false);
                this->recreateSwapChain();
            } else if (resultQueuePresentKHR != VK_SUCCESS) {
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
    auto app = App {};

    try {
        app.run();
    } catch (const std::exception& exception) {
        fmt::println(std::cerr, "{}", exception.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
