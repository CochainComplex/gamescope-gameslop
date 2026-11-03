#include "vulkan_engine.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace gamescope::framegen::compute
{
namespace
{

void check( VkResult result, const char *operation )
{
	if ( result != VK_SUCCESS )
		throw std::runtime_error( std::string( operation ) + ": VkResult=" + std::to_string( result ) );
}

template<typename T> T vk_info( VkStructureType type )
{
	T value{};
	value.sType = type;
	return value;
}

template<typename T, typename F> std::vector<T> enumerate( F function )
{
	uint32_t count = 0;
	check( function( &count, static_cast<T *>( nullptr ) ), "enumerate count" );
	std::vector<T> result;
	VkResult status;
	do
	{
		result.resize( count );
		status = function( &count, result.data() );
		if ( status == VK_INCOMPLETE )
			check( function( &count, static_cast<T *>( nullptr ) ), "enumerate recount" );
	} while ( status == VK_INCOMPLETE );
	check( status, "enumerate" );
	result.resize( count );
	return result;
}

struct Instance
{
	VkInstance handle = VK_NULL_HANDLE;
	VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
	std::atomic<uint32_t> validationErrors{ 0 };

	static VKAPI_ATTR VkBool32 VKAPI_CALL message( VkDebugUtilsMessageSeverityFlagBitsEXT severity,
		VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT *data, void *user )
	{
		if ( severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT )
			static_cast<Instance *>( user )->validationErrors.fetch_add( 1 );
		std::fprintf( stderr, "Vulkan validation: %s\n", data->pMessage );
		return VK_FALSE;
	}

	void create( bool validation )
	{
		const char *layer = "VK_LAYER_KHRONOS_validation";
		const char *extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
		if ( validation )
		{
			const auto layers = enumerate<VkLayerProperties>( []( auto *count, auto *data ) {
				return vkEnumerateInstanceLayerProperties( count, data ); } );
			if ( std::none_of( layers.begin(), layers.end(), [&]( const auto &entry ) {
				return !std::strcmp( entry.layerName, layer ); } ) )
				throw std::runtime_error( "requested Vulkan validation layer unavailable" );
		}
		auto debug = vk_info<VkDebugUtilsMessengerCreateInfoEXT>( VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT );
		debug.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debug.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		debug.pfnUserCallback = message;
		debug.pUserData = this;
		auto application = vk_info<VkApplicationInfo>( VK_STRUCTURE_TYPE_APPLICATION_INFO );
		application.pApplicationName = "gamescope native compute conformance PoC";
		application.apiVersion = VK_API_VERSION_1_2;
		auto info = vk_info<VkInstanceCreateInfo>( VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO );
		info.pApplicationInfo = &application;
		if ( validation )
		{
			info.enabledLayerCount = info.enabledExtensionCount = 1;
			info.ppEnabledLayerNames = &layer;
			info.ppEnabledExtensionNames = &extension;
			info.pNext = &debug;
		}
		check( vkCreateInstance( &info, nullptr, &handle ), "vkCreateInstance" );
		if ( validation )
		{
			auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
				vkGetInstanceProcAddr( handle, "vkCreateDebugUtilsMessengerEXT" ) );
			if ( !create ) throw std::runtime_error( "debug messenger entry point unavailable" );
			check( create( handle, &debug, nullptr, &messenger ), "create debug messenger" );
		}
	}
	void destroy()
	{
		if ( messenger )
		{
			auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
				vkGetInstanceProcAddr( handle, "vkDestroyDebugUtilsMessengerEXT" ) );
			if ( destroy ) destroy( handle, messenger, nullptr );
			messenger = VK_NULL_HANDLE;
		}
		if ( handle ) vkDestroyInstance( handle, nullptr );
		handle = VK_NULL_HANDLE;
	}
	~Instance() { destroy(); }
};

std::vector<VkPhysicalDevice> physical_devices( VkInstance instance )
{
	return enumerate<VkPhysicalDevice>( [&]( auto *count, auto *data ) {
		return vkEnumeratePhysicalDevices( instance, count, data ); } );
}

Device describe( VkPhysicalDevice physical )
{
	const auto extensions = enumerate<VkExtensionProperties>( [&]( auto *count, auto *data ) {
		return vkEnumerateDeviceExtensionProperties( physical, nullptr, count, data ); } );
	const bool hasPci = std::any_of( extensions.begin(), extensions.end(), []( const auto &entry ) {
		return !std::strcmp( entry.extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME ); } );
	auto pci = vk_info<VkPhysicalDevicePCIBusInfoPropertiesEXT>( VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT );
	auto identity = vk_info<VkPhysicalDeviceIDProperties>( VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES );
	identity.pNext = hasPci ? &pci : nullptr;
	auto properties = vk_info<VkPhysicalDeviceProperties2>( VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 );
	properties.pNext = &identity;
	vkGetPhysicalDeviceProperties2( physical, &properties );
	Device result;
	std::copy_n( identity.deviceUUID, result.uuid.size(), result.uuid.begin() );
	result.name = properties.properties.deviceName;
	result.vendor = properties.properties.vendorID;
	result.device = properties.properties.deviceID;
	result.software = properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
	if ( hasPci )
	{
		char address[32];
		std::snprintf( address, sizeof( address ), "%04x:%02x:%02x.%x",
			pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction );
		result.pci = address;
	}
	return result;
}

struct Buffer
{
	VkBuffer handle = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	void *mapped = nullptr;
	bool coherent = false;
	void destroy( VkDevice device )
	{
		if ( mapped ) vkUnmapMemory( device, memory );
		if ( handle ) vkDestroyBuffer( device, handle, nullptr );
		if ( memory ) vkFreeMemory( device, memory, nullptr );
		*this = {};
	}
};

} // namespace

struct VulkanEngine::Impl
{
	Instance instance;
	Device identity;
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties properties{};
	VkDevice device = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	VkCommandPool pool = VK_NULL_HANDLE;
	VkCommandBuffer command = VK_NULL_HANDLE;
	VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	VkDescriptorSet descriptors = VK_NULL_HANDLE;
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	struct Kernel { VkShaderModule shader = VK_NULL_HANDLE; VkPipeline pipeline = VK_NULL_HANDLE; };
	std::vector<Kernel> kernels;
	VkSemaphore timeline = VK_NULL_HANDLE;
	std::array<Buffer, 3> buffers{};
	std::array<VkDeviceSize, 3> bytes{};
	size_t pixels = 0;
	Batch batch;
	Ticket active{};
	bool poisoned = false;
	bool retirementFailed = false;

	void destroy()
	{
		if ( !device ) { instance.destroy(); return; }
		// Standalone shutdown only. A production adapter must use the existing
		// renderer retirement owner and must never drain on the present thread.
		const VkResult retired = vkDeviceWaitIdle( device );
		if ( retired != VK_SUCCESS )
		{
			retirementFailed = true;
			std::fprintf( stderr, "compute device retirement: VkResult=%d\n", retired );
		}
		for ( auto &buffer : buffers ) buffer.destroy( device );
		if ( timeline ) vkDestroySemaphore( device, timeline, nullptr );
		for ( const auto &kernel : kernels )
		{
			if ( kernel.pipeline ) vkDestroyPipeline( device, kernel.pipeline, nullptr );
			if ( kernel.shader ) vkDestroyShaderModule( device, kernel.shader, nullptr );
		}
		if ( pipelineLayout ) vkDestroyPipelineLayout( device, pipelineLayout, nullptr );
		if ( descriptorPool ) vkDestroyDescriptorPool( device, descriptorPool, nullptr );
		if ( descriptorLayout ) vkDestroyDescriptorSetLayout( device, descriptorLayout, nullptr );
		if ( pool ) vkDestroyCommandPool( device, pool, nullptr );
		vkDestroyDevice( device, nullptr );
		device = VK_NULL_HANDLE;
		instance.destroy();
	}
	~Impl() { destroy(); }

	void healthy() const
	{
		if ( poisoned ) throw std::runtime_error( "compute engine failed; no further admission" );
		if ( !device ) throw std::runtime_error( "compute engine is closed" );
		if ( instance.validationErrors.load() ) throw std::runtime_error( "Vulkan validation error" );
	}

	void initialize( DeviceUuid uuid, const std::string &path, bool validation )
	{
		if ( std::all_of( uuid.begin(), uuid.end(), []( uint8_t b ) { return b == 0; } ) )
			throw std::runtime_error( "presenter UUID must be explicit and nonzero" );
		instance.create( validation );
		uint32_t matches = 0;
		for ( auto candidate : physical_devices( instance.handle ) )
		{
			Device description = describe( candidate );
			if ( description.uuid == uuid )
			{
				physical = candidate;
				identity = std::move( description );
				++matches;
			}
		}
		if ( matches != 1 ) throw std::runtime_error( "presenter UUID must match exactly one Vulkan device" );
		if ( identity.software ) throw std::runtime_error( "software Vulkan is not presenter-GPU evidence" );
		vkGetPhysicalDeviceProperties( physical, &properties );
		if ( properties.apiVersion < VK_API_VERSION_1_2 )
			throw std::runtime_error( "Vulkan 1.2 required for this conformance backend" );
		auto features = vk_info<VkPhysicalDeviceTimelineSemaphoreFeatures>( VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES );
		auto supported = vk_info<VkPhysicalDeviceFeatures2>( VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 );
		supported.pNext = &features;
		vkGetPhysicalDeviceFeatures2( physical, &supported );
		if ( !features.timelineSemaphore ) throw std::runtime_error( "timeline semaphores unavailable" );
		uint32_t count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties( physical, &count, nullptr );
		std::vector<VkQueueFamilyProperties> families( count );
		vkGetPhysicalDeviceQueueFamilyProperties( physical, &count, families.data() );
		uint32_t family = UINT32_MAX;
		for ( uint32_t i = 0; i < count; ++i )
		{
			if ( families[i].queueCount && ( families[i].queueFlags & VK_QUEUE_COMPUTE_BIT ) )
			{
				if ( family == UINT32_MAX ) family = i;
				if ( !( families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT ) ) { family = i; break; }
			}
		}
		if ( family == UINT32_MAX ) throw std::runtime_error( "no compute queue on presenter device" );
		const float priority = 0.0f;
		auto queueInfo = vk_info<VkDeviceQueueCreateInfo>( VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO );
		queueInfo.queueFamilyIndex = family;
		queueInfo.queueCount = 1;
		queueInfo.pQueuePriorities = &priority;
		auto create = vk_info<VkDeviceCreateInfo>( VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO );
		create.pNext = &features;
		create.queueCreateInfoCount = 1;
		create.pQueueCreateInfos = &queueInfo;
		check( vkCreateDevice( physical, &create, nullptr, &device ), "vkCreateDevice" );
		vkGetDeviceQueue( device, family, 0, &queue );
		auto poolInfo = vk_info<VkCommandPoolCreateInfo>( VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO );
		poolInfo.queueFamilyIndex = family;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		check( vkCreateCommandPool( device, &poolInfo, nullptr, &pool ), "vkCreateCommandPool" );
		auto allocate = vk_info<VkCommandBufferAllocateInfo>( VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO );
		allocate.commandPool = pool;
		allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocate.commandBufferCount = 1;
		check( vkAllocateCommandBuffers( device, &allocate, &command ), "vkAllocateCommandBuffers" );
		auto type = vk_info<VkSemaphoreTypeCreateInfo>( VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO );
		type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
		auto semaphore = vk_info<VkSemaphoreCreateInfo>( VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO );
		semaphore.pNext = &type;
		check( vkCreateSemaphore( device, &semaphore, nullptr, &timeline ), "vkCreateSemaphore" );

		std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
		for ( uint32_t i = 0; i < bindings.size(); ++i )
			bindings[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
		auto layout = vk_info<VkDescriptorSetLayoutCreateInfo>( VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO );
		layout.bindingCount = bindings.size();
		layout.pBindings = bindings.data();
		check( vkCreateDescriptorSetLayout( device, &layout, nullptr, &descriptorLayout ), "descriptor layout" );
		VkDescriptorPoolSize descriptorSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
		auto descriptorInfo = vk_info<VkDescriptorPoolCreateInfo>( VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO );
		descriptorInfo.maxSets = 1;
		descriptorInfo.poolSizeCount = 1;
		descriptorInfo.pPoolSizes = &descriptorSize;
		check( vkCreateDescriptorPool( device, &descriptorInfo, nullptr, &descriptorPool ), "descriptor pool" );
		auto descriptorAllocate = vk_info<VkDescriptorSetAllocateInfo>( VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO );
		descriptorAllocate.descriptorPool = descriptorPool;
		descriptorAllocate.descriptorSetCount = 1;
		descriptorAllocate.pSetLayouts = &descriptorLayout;
		check( vkAllocateDescriptorSets( device, &descriptorAllocate, &descriptors ), "descriptor set" );
		VkPushConstantRange push{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 128 };
		auto pipelineInfo = vk_info<VkPipelineLayoutCreateInfo>( VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO );
		pipelineInfo.setLayoutCount = 1;
		pipelineInfo.pSetLayouts = &descriptorLayout;
		pipelineInfo.pushConstantRangeCount = 1;
		pipelineInfo.pPushConstantRanges = &push;
		check( vkCreatePipelineLayout( device, &pipelineInfo, nullptr, &pipelineLayout ), "pipeline layout" );
		addKernel( path );
		healthy();
	}

	uint32_t addKernel( const std::string &path )
	{
		std::ifstream stream( path, std::ios::binary | std::ios::ate );
		const auto size = stream.tellg();
		if ( !stream || size <= 0 || size % 4 != 0 || size > 1024 * 1024 )
			throw std::runtime_error( "invalid/missing conformance SPIR-V: " + path );
		std::vector<uint32_t> code( static_cast<size_t>( size ) / 4 );
		stream.seekg( 0 );
		if ( !stream.read( reinterpret_cast<char *>( code.data() ), size ) || code[0] != 0x07230203 )
			throw std::runtime_error( "invalid conformance SPIR-V data" );
		auto shaderInfo = vk_info<VkShaderModuleCreateInfo>( VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO );
		shaderInfo.codeSize = static_cast<size_t>( size );
		shaderInfo.pCode = code.data();
		kernels.emplace_back();
		auto &kernel = kernels.back();
		auto &shader = kernel.shader;
		auto &pipeline = kernel.pipeline;
		check( vkCreateShaderModule( device, &shaderInfo, nullptr, &shader ), "shader module" );
		auto compute = vk_info<VkComputePipelineCreateInfo>( VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO );
		compute.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		compute.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		compute.stage.module = shader;
		compute.stage.pName = "main";
		compute.layout = pipelineLayout;
		check( vkCreateComputePipelines( device, VK_NULL_HANDLE, 1, &compute, nullptr, &pipeline ), "compute pipeline" );
		return uint32_t( kernels.size() - 1 );
	}

	void resize( std::array<VkDeviceSize, 3> sizes )
	{
		if ( bytes == sizes ) return;
		for ( auto &buffer : buffers ) buffer.destroy( device );
		bytes = {};
		VkPhysicalDeviceMemoryProperties memoryProperties;
		vkGetPhysicalDeviceMemoryProperties( physical, &memoryProperties );
		for ( size_t slot = 0; slot < buffers.size(); ++slot )
		{
			auto &buffer = buffers[slot];
			auto create = vk_info<VkBufferCreateInfo>( VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO );
			create.size = sizes[slot];
			create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			check( vkCreateBuffer( device, &create, nullptr, &buffer.handle ), "vkCreateBuffer" );
			VkMemoryRequirements requirements;
			vkGetBufferMemoryRequirements( device, buffer.handle, &requirements );
			uint32_t type = UINT32_MAX;
			for ( uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i )
			{
				const auto flags = memoryProperties.memoryTypes[i].propertyFlags;
				if ( ( requirements.memoryTypeBits & ( 1u << i ) ) && ( flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ) )
				{
					if ( type == UINT32_MAX ) type = i;
					if ( flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ) { type = i; break; }
				}
			}
			if ( type == UINT32_MAX ) throw std::runtime_error( "no host-visible buffer memory" );
			buffer.coherent = memoryProperties.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			auto allocation = vk_info<VkMemoryAllocateInfo>( VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO );
			allocation.allocationSize = requirements.size;
			allocation.memoryTypeIndex = type;
			check( vkAllocateMemory( device, &allocation, nullptr, &buffer.memory ), "vkAllocateMemory" );
			check( vkBindBufferMemory( device, buffer.handle, buffer.memory, 0 ), "vkBindBufferMemory" );
			check( vkMapMemory( device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped ), "vkMapMemory" );
		}
		std::array<VkDescriptorBufferInfo, 3> info{};
		std::array<VkWriteDescriptorSet, 3> writes{};
		for ( uint32_t i = 0; i < buffers.size(); ++i )
		{
			info[i] = { buffers[i].handle, 0, sizes[i] };
			writes[i] = vk_info<VkWriteDescriptorSet>( VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET );
			writes[i].dstSet = descriptors;
			writes[i].dstBinding = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writes[i].pBufferInfo = &info[i];
		}
		vkUpdateDescriptorSets( device, writes.size(), writes.data(), 0, nullptr );
		bytes = sizes;
	}

	void flush( const Buffer &buffer, bool invalidate )
	{
		if ( buffer.coherent ) return;
		auto range = vk_info<VkMappedMemoryRange>( VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE );
		range.memory = buffer.memory;
		range.size = VK_WHOLE_SIZE;
		check( invalidate ? vkInvalidateMappedMemoryRanges( device, 1, &range )
			: vkFlushMappedMemoryRanges( device, 1, &range ), "mapped memory visibility" );
	}
	Ticket submitRaw( uint64_t epoch, const std::array<std::span<const std::byte>, 3> &initial,
		std::span<const BufferStep> steps, bool preserve = false );
	bool readBuffers( const std::array<BufferReadback, 3> &output );
};

std::vector<Device> VulkanEngine::devices()
{
	Instance instance;
	instance.create( false );
	std::vector<Device> result;
	for ( auto physical : physical_devices( instance.handle ) ) result.push_back( describe( physical ) );
	return result;
}

VulkanEngine::VulkanEngine( DeviceUuid presenter, const std::string &shaderPath, bool validation )
	: m( std::make_unique<Impl>() )
{
	m->initialize( presenter, shaderPath, validation );
}
VulkanEngine::~VulkanEngine() = default;
const Device &VulkanEngine::device() const { return m->identity; }
uint64_t VulkanEngine::epoch() const { return m->batch.epoch(); }

Ticket VulkanEngine::submit( const Request &request )
{
	m->healthy();
	const auto placement = phase( request.pair );
	const uint64_t count = uint64_t( request.width ) * request.height;
	if ( m->batch.busy() || !placement || request.pair.epoch != epoch()
		|| request.width == 0 || request.height == 0
		|| request.width > INT32_MAX || request.height > INT32_MAX
		|| count > m->properties.limits.maxStorageBufferRange / sizeof( Pixel )
		|| ( count + 63 ) / 64 > m->properties.limits.maxComputeWorkGroupCount[0]
		|| request.previous.size() != count || request.current.size() != count
		|| ( request.method != Method::Blend && request.method != Method::Translation )
		|| !std::isfinite( request.dx ) || !std::isfinite( request.dy ) ) return {};
	const auto finite = []( const Pixel &pixel ) {
		return std::isfinite( pixel.r ) && std::isfinite( pixel.g )
			&& std::isfinite( pixel.b ) && std::isfinite( pixel.a ); };
	if ( !std::all_of( request.previous.begin(), request.previous.end(), finite )
		|| !std::all_of( request.current.begin(), request.current.end(), finite ) ) return {};
	std::vector<Pixel> empty( count );
	const Parameters parameters{ request.width, request.height, uint32_t( request.method ),
		*placement, request.dx, request.dy };
	const BufferStep step{ 0, std::as_bytes( std::span( &parameters, 1 ) ), uint32_t( ( count + 63 ) / 64 ) };
	const auto ticket = m->submitRaw( request.pair.epoch,
		{ std::as_bytes( request.previous ), std::as_bytes( request.current ),
			std::as_bytes( std::span( empty ) ) }, std::span( &step, 1 ) );
	if ( ticket.serial ) m->pixels = static_cast<size_t>( count );
	return ticket;
}

Ticket VulkanEngine::Impl::submitRaw( uint64_t epoch,
	const std::array<std::span<const std::byte>, 3> &initial, std::span<const BufferStep> steps, bool preserve )
{
	healthy();
	if ( batch.busy() || epoch != batch.epoch() || steps.empty() ) return {};
	std::array<VkDeviceSize, 3> sizes{};
	for ( size_t i = 0; i < initial.size(); ++i )
	{
		if ( preserve )
		{
			if ( !bytes[i] || ( !initial[i].empty() && initial[i].size() != bytes[i] ) ) return {};
			sizes[i] = bytes[i];
		}
		else
		{
			if ( initial[i].empty() || initial[i].size() > properties.limits.maxStorageBufferRange ) return {};
			sizes[i] = initial[i].size();
		}
	}
	for ( const auto &step : steps )
	{
		if ( step.kernel >= kernels.size() || !kernels[step.kernel].pipeline
			|| step.constants.empty() || step.constants.size() > 128 || step.constants.size() % 4
			|| !step.groupsX || !step.groupsY || !step.groupsZ
			|| step.groupsX > properties.limits.maxComputeWorkGroupCount[0]
			|| step.groupsY > properties.limits.maxComputeWorkGroupCount[1]
			|| step.groupsZ > properties.limits.maxComputeWorkGroupCount[2] ) return {};
	}
	resize( sizes );
	pixels = 0;
	for ( size_t i = 0; i < buffers.size(); ++i )
	{
		if ( initial[i].empty() ) continue;
		std::memcpy( buffers[i].mapped, initial[i].data(), bytes[i] );
		flush( buffers[i], false );
	}
	check( vkResetCommandPool( device, pool, 0 ), "vkResetCommandPool" );
	auto begin = vk_info<VkCommandBufferBeginInfo>( VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO );
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	check( vkBeginCommandBuffer( command, &begin ), "vkBeginCommandBuffer" );
	auto before = vk_info<VkMemoryBarrier>( VK_STRUCTURE_TYPE_MEMORY_BARRIER );
	before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
	VkPipelineStageFlags sourceStages = VK_PIPELINE_STAGE_HOST_BIT;
	if ( preserve )
	{
		before.srcAccessMask |= VK_ACCESS_SHADER_WRITE_BIT;
		sourceStages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	}
	before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	vkCmdPipelineBarrier( command, sourceStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 1, &before, 0, nullptr, 0, nullptr );
	vkCmdBindDescriptorSets( command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
		0, 1, &descriptors, 0, nullptr );
	for ( const auto &step : steps )
	{
		vkCmdBindPipeline( command, VK_PIPELINE_BIND_POINT_COMPUTE, kernels[step.kernel].pipeline );
		vkCmdPushConstants( command, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
			0, uint32_t( step.constants.size() ), step.constants.data() );
		vkCmdDispatch( command, step.groupsX, step.groupsY, step.groupsZ );
		auto between = vk_info<VkMemoryBarrier>( VK_STRUCTURE_TYPE_MEMORY_BARRIER );
		between.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		between.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier( command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0, 1, &between, 0, nullptr, 0, nullptr );
	}
	auto after = vk_info<VkMemoryBarrier>( VK_STRUCTURE_TYPE_MEMORY_BARRIER );
	after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
	vkCmdPipelineBarrier( command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
		0, 1, &after, 0, nullptr, 0, nullptr );
	check( vkEndCommandBuffer( command ), "vkEndCommandBuffer" );
	healthy();
	active = batch.begin( epoch );
	if ( active.serial == 0 ) throw std::runtime_error( "compute serial exhausted" );
	auto values = vk_info<VkTimelineSemaphoreSubmitInfo>( VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO );
	values.signalSemaphoreValueCount = 1;
	values.pSignalSemaphoreValues = &active.serial;
	auto submit = vk_info<VkSubmitInfo>( VK_STRUCTURE_TYPE_SUBMIT_INFO );
	submit.pNext = &values;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &command;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &timeline;
	const VkResult result = vkQueueSubmit( queue, 1, &submit, VK_NULL_HANDLE );
	if ( result != VK_SUCCESS ) poisoned = true;
	check( result, "vkQueueSubmit" );
	return active;
}

Status VulkanEngine::poll( Ticket ticket )
{
	m->healthy();
	if ( !m->batch.owns( ticket ) ) return Status::Stale;
	uint64_t value = 0;
	check( vkGetSemaphoreCounterValue( m->device, m->timeline, &value ), "timeline poll" );
	if ( value < ticket.serial ) return Status::Pending;
	m->batch.complete( ticket );
	return m->batch.discarded( ticket ) ? Status::Discarded : Status::Complete;
}

void VulkanEngine::discard( Ticket ticket ) { m->batch.discard( ticket ); }

bool VulkanEngine::collect( Ticket ticket, std::span<Pixel> output )
{
	if ( !m->pixels || output.size() != m->pixels || poll( ticket ) != Status::Complete ) return false;
	m->flush( m->buffers[2], true );
	const auto *pixels = static_cast<const Pixel *>( m->buffers[2].mapped );
	for ( size_t i = 0; i < m->pixels; ++i )
		if ( !std::isfinite( pixels[i].r ) || !std::isfinite( pixels[i].g )
			|| !std::isfinite( pixels[i].b ) || !std::isfinite( pixels[i].a ) )
			throw std::runtime_error( "nonfinite compute output" );
	std::memcpy( output.data(), pixels, m->bytes[2] );
	return m->batch.retire( ticket );
}

bool VulkanEngine::retire( Ticket ticket )
{
	if ( poll( ticket ) == Status::Pending ) return false;
	return m->batch.retire( ticket );
}

bool VulkanEngine::reset() { m->healthy(); return m->batch.reset(); }

uint32_t VulkanEngine::addKernel( const std::string &path )
{
	m->healthy();
	if ( m->batch.busy() ) throw std::runtime_error( "cannot add a kernel while busy" );
	try { return m->addKernel( path ); }
	catch ( ... ) { m->poisoned = true; throw; }
}

Ticket VulkanEngine::submitBuffers( uint64_t epoch,
	const std::array<std::span<const float>, 3> &initial, std::span<const BufferStep> steps )
{
	return m->submitRaw( epoch, { std::as_bytes( initial[0] ), std::as_bytes( initial[1] ),
		std::as_bytes( initial[2] ) }, steps );
}

bool VulkanEngine::collectBuffers( Ticket ticket, const std::array<std::span<float>, 3> &output )
{
	for ( size_t i = 0; i < output.size(); ++i )
		if ( !output[i].empty() && output[i].size_bytes() != m->bytes[i] ) return false;
	const std::array<BufferReadback, 3> ranges{{
		{ 0, std::as_writable_bytes( output[0] ) },
		{ 0, std::as_writable_bytes( output[1] ) },
		{ 0, std::as_writable_bytes( output[2] ) } }};
	return collectBufferRanges( ticket, ranges );
}

bool VulkanEngine::Impl::readBuffers( const std::array<BufferReadback, 3> &output )
{
	healthy();
	// Validate every range before copying any output. Subtraction avoids overflow.
	for ( size_t i = 0; i < output.size(); ++i )
		if ( !output[i].output.empty() && ( output[i].offsetBytes > bytes[i]
			|| output[i].output.size_bytes() > bytes[i] - output[i].offsetBytes ) ) return false;
	for ( size_t i = 0; i < output.size(); ++i )
		if ( !output[i].output.empty() )
		{
			// Keep allocation-wide invalidation for noncoherent memory. The mapped
			// range is valid/aligned even when the requested diagnostic is tiny.
			flush( buffers[i], true );
			std::memcpy( output[i].output.data(),
				static_cast<const std::byte *>( buffers[i].mapped ) + output[i].offsetBytes,
				output[i].output.size_bytes() );
		}
	return true;
}

bool VulkanEngine::collectBufferRanges( Ticket ticket, const std::array<BufferReadback, 3> &output )
{
	if ( poll( ticket ) != Status::Complete || !m->readBuffers( output ) ) return false;
	return m->batch.retire( ticket );
}

Ticket VulkanEngine::updateBuffers( uint64_t epoch,
	const std::array<std::span<const float>, 3> &updates, std::span<const BufferStep> steps )
{
	return m->submitRaw( epoch, { std::as_bytes( updates[0] ), std::as_bytes( updates[1] ),
		std::as_bytes( updates[2] ) }, steps, true );
}

bool VulkanEngine::readIdleBuffers( const std::array<BufferReadback, 3> &output )
{
	m->healthy();
	return !m->batch.busy() && m->readBuffers( output );
}

void VulkanEngine::shutdown()
{
	if ( m->batch.busy() ) throw std::runtime_error( "retire outstanding ticket before shutdown" );
	m->destroy();
	if ( m->instance.validationErrors.load() || m->retirementFailed )
		throw std::runtime_error( "compute teardown failed validation or retirement" );
}

} // namespace gamescope::framegen::compute
