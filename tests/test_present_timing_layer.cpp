#define VK_USE_PLATFORM_WAYLAND_KHR
#define VK_USE_PLATFORM_XCB_KHR
#define VK_USE_PLATFORM_XLIB_KHR
#include "vkroots.h"

// Exercise the actual layer and its loader interface with a fake driver.
#include "../layer/VkLayer_FROG_gamescope_wsi.cpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <limits>
#include <sys/socket.h>

namespace {

// Non-dispatchable handles are uint64_t rather than pointers on 32-bit targets.
template <typename T>
T fakeHandle(uint64_t value) { return T(value); }

uint32_t destroyedInstances = 0;
uint32_t destroyedDevices = 0;

void VKAPI_CALL destroyInstance(VkInstance instance, const VkAllocationCallbacks *) {
  REQUIRE(vkroots::LookupDispatch(instance) == nullptr);
  destroyedInstances++;
}

void VKAPI_CALL destroyDevice(VkDevice device, const VkAllocationCallbacks *) {
  REQUIRE(vkroots::LookupDispatch(device) == nullptr);
  destroyedDevices++;
}

VkResult VKAPI_CALL enumeratePhysicalDevices(VkInstance, uint32_t *count, VkPhysicalDevice *) {
  *count = 0;
  return VK_SUCCESS;
}

PFN_vkVoidFunction VKAPI_CALL getInstanceProc(VkInstance, const char *name) {
  if (std::strcmp(name, "vkGetInstanceProcAddr") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(getInstanceProc);
  if (std::strcmp(name, "vkDestroyInstance") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(destroyInstance);
  if (std::strcmp(name, "vkEnumeratePhysicalDevices") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(enumeratePhysicalDevices);
  return nullptr;
}

struct RefusedPresent {
  uint32_t submits = 0;
  uint32_t idles = 0;
  std::vector<VkSwapchainKHR> released;
  VkResult idleResult = VK_SUCCESS;
  uint32_t failSubmit = 0;
  std::vector<VkSemaphore> waits;
  std::vector<VkFence> fences;
  std::vector<VkFrameBoundaryFlagsEXT> boundaries;
} refused;

VkResult VKAPI_CALL submit(VkQueue, uint32_t count, const VkSubmitInfo *infos, VkFence fence) {
  refused.submits++;
  if (refused.submits == refused.failSubmit)
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  for (uint32_t i = 0; i < count; i++) {
    if (auto *boundary = vkroots::FindInChain<VkFrameBoundaryEXT>(&infos[i]))
      refused.boundaries.push_back(boundary->flags);
    for (uint32_t j = 0; j < infos[i].waitSemaphoreCount; j++)
      refused.waits.push_back(infos[i].pWaitSemaphores[j]);
  }
  if (fence)
    refused.fences.push_back(fence);
  return VK_SUCCESS;
}

VkResult VKAPI_CALL waitIdle(VkQueue) {
  refused.idles++;
  return refused.idleResult;
}

VkResult VKAPI_CALL releaseImages(VkDevice, const VkReleaseSwapchainImagesInfoKHR *info) {
  REQUIRE(info->imageIndexCount == 1);
  refused.released.push_back(info->swapchain);
  return VK_SUCCESS;
}

struct PresentedSwapchain {
  VkSwapchainKHR swapchain;
  uint32_t image;
  uint64_t id;
  uint64_t target;
  VkPresentStageFlagsEXT stages;
  VkFence fence;
};
std::vector<PresentedSwapchain> presented;
uint32_t failPresent = 0;
uint32_t presentCalls = 0;

VkResult VKAPI_CALL present(VkQueue, const VkPresentInfoKHR *info) {
  presentCalls++;
  if (presentCalls == failPresent)
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  if (auto *boundary = vkroots::FindInChain<VkFrameBoundaryEXT>(info))
    refused.boundaries.push_back(boundary->flags);
  auto *ids = vkroots::FindInChain<VkPresentId2KHR>(info);
  auto *timings = vkroots::FindInChain<VkPresentTimingsInfoEXT>(info);
  auto *fences = vkroots::FindInChain<VkSwapchainPresentFenceInfoEXT>(info);
  if (auto *modes = vkroots::FindInChain<VkSwapchainPresentModeInfoEXT>(info)) {
    REQUIRE(modes->swapchainCount == info->swapchainCount);
    REQUIRE(modes->swapchainCount > 0);
    for (uint32_t i = 0; i < modes->swapchainCount; i++)
      CHECK(modes->pPresentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR);
  }
  for (uint32_t i = 0; i < info->waitSemaphoreCount; i++)
    refused.waits.push_back(info->pWaitSemaphores[i]);
  for (uint32_t i = 0; i < info->swapchainCount; i++) {
    const auto *timing = timings && timings->pTimingInfos ? &timings->pTimingInfos[i] : nullptr;
    presented.push_back({info->pSwapchains[i], info->pImageIndices[i], ids ? ids->pPresentIds[i] : 0,
      timing ? timing->targetTime : 0, timing ? timing->presentStageQueries : 0,
      fences ? fences->pFences[i] : VK_NULL_HANDLE});
    if (info->pResults)
      info->pResults[i] = VK_SUCCESS;
  }
  return VK_SUCCESS;
}

PFN_vkVoidFunction VKAPI_CALL getDeviceProc(VkDevice, const char *name) {
  if (std::strcmp(name, "vkGetDeviceProcAddr") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(getDeviceProc);
  if (std::strcmp(name, "vkDestroyDevice") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(destroyDevice);
  if (std::strcmp(name, "vkQueuePresentKHR") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(present);
  if (std::strcmp(name, "vkQueueSubmit") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(submit);
  if (std::strcmp(name, "vkQueueWaitIdle") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(waitIdle);
  if (std::strcmp(name, "vkReleaseSwapchainImagesEXT") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(releaseImages);
  return nullptr;
}

struct DriverCalibration {
  bool ext = false;
  bool khr = false;
  uint32_t count = 0;
  uint32_t calls = 0;
  const VkCalibratedTimestampInfoKHR *original = nullptr;
  std::vector<VkTimeDomainKHR> domains;
  VkResult result = VK_SUCCESS;
  uint64_t deviation = 7;
} driver;

VkResult VKAPI_CALL calibrate(VkDevice, uint32_t count, const VkCalibratedTimestampInfoKHR *infos,
                              uint64_t *times, uint64_t *deviation) {
  driver.calls++;
  driver.count = count;
  driver.original = infos;
  for (uint32_t i = 0; i < count; i++) {
    driver.domains.push_back(infos[i].timeDomain);
    times[i] = 1000 + i;
  }
  *deviation = driver.deviation;
  return driver.result;
}

PFN_vkVoidFunction VKAPI_CALL getCalibrationProc(VkDevice device, const char *name) {
  if (std::strcmp(name, "vkGetDeviceProcAddr") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(getCalibrationProc);
  if (std::strcmp(name, "vkGetCalibratedTimestampsEXT") == 0)
    return driver.ext ? reinterpret_cast<PFN_vkVoidFunction>(calibrate) : nullptr;
  if (std::strcmp(name, "vkGetCalibratedTimestampsKHR") == 0)
    return driver.khr ? reinterpret_cast<PFN_vkVoidFunction>(calibrate) : nullptr;
  return getDeviceProc(device, name);
}

VkResult VKAPI_CALL calibrationDomains(VkPhysicalDevice, uint32_t *count, VkTimeDomainKHR *domains) {
  const std::array values{ VK_TIME_DOMAIN_DEVICE_KHR, VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR };
  return vkroots::array(values, count, domains);
}

VkResult VKAPI_CALL calibrationExtensions(VkPhysicalDevice, const char *, uint32_t *count, VkExtensionProperties *properties) {
  std::vector<VkExtensionProperties> extensions{
    { VK_KHR_PRESENT_ID_2_EXTENSION_NAME, VK_KHR_PRESENT_ID_2_SPEC_VERSION },
  };
  if (driver.khr)
    extensions.push_back({ VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME, VK_KHR_CALIBRATED_TIMESTAMPS_SPEC_VERSION });
  if (driver.ext)
    extensions.push_back({ VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME, VK_EXT_CALIBRATED_TIMESTAMPS_SPEC_VERSION });
  return vkroots::array(extensions, count, properties);
}

PFN_vkVoidFunction VKAPI_CALL getCalibrationInstanceProc(VkInstance instance, const char *name) {
  if (std::strcmp(name, "vkGetInstanceProcAddr") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(getCalibrationInstanceProc);
  if (std::strcmp(name, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT") == 0)
    return driver.ext ? reinterpret_cast<PFN_vkVoidFunction>(calibrationDomains) : nullptr;
  if (std::strcmp(name, "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR") == 0)
    return driver.khr ? reinterpret_cast<PFN_vkVoidFunction>(calibrationDomains) : nullptr;
  if (std::strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(calibrationExtensions);
  return getInstanceProc(instance, name);
}

const VkSwapchainKHR layerSwapchain = fakeHandle<VkSwapchainKHR>(1);
const VkSwapchainKHR driverSwapchain = fakeHandle<VkSwapchainKHR>(2);

struct Calibration {
  VkSwapchainCalibratedTimestampInfoEXT local = {
    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO_EXT,
    .swapchain = layerSwapchain,
    .presentStage = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT,
    .timeDomainId = 1,
  };
  VkSwapchainCalibratedTimestampInfoEXT foreign = {
    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO_EXT,
    .swapchain = driverSwapchain,
    .presentStage = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT,
    .timeDomainId = 8,
  };
  std::array<VkCalibratedTimestampInfoKHR, 4> infos = {{
    { VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, &local, VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT },
    { VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, nullptr, VK_TIME_DOMAIN_DEVICE_KHR },
    { VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, &local, VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT },
    { VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, &foreign, VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT },
  }};
  std::array<uint64_t, 4> timestamps{};
  uint64_t deviation = 0;
  uint64_t before = 0;
  uint64_t after = 0;

  Calibration() {
    driver = {};
    GamescopeWSILayer::gamescopeSwapchains.erase(layerSwapchain);
    REQUIRE(GamescopeWSILayer::gamescopeSwapchains.create(layerSwapchain, GamescopeWSILayer::GamescopeSwapchainData{}) != nullptr);
  }

  ~Calibration() {
    GamescopeWSILayer::gamescopeSwapchains.erase(layerSwapchain);
  }

  VkResult run(uint32_t count) {
    before = GamescopeWSILayer::getTimeMonotonic();
    VkResult result = GamescopeWSILayer::calibratePresentTimestamps(calibrate, VK_NULL_HANDLE, count,
      infos.data(), timestamps.data(), &deviation);
    after = GamescopeWSILayer::getTimeMonotonic();
    return result;
  }

  bool sampledDuringCall(size_t i) const {
    return timestamps[i] >= before && timestamps[i] <= after;
  }
};

}

namespace {
void VKAPI_CALL driverTimingFeatures(VkPhysicalDevice, VkPhysicalDeviceFeatures2 *features) {
  if (auto *timing = vkroots::FindInChainMutable<VkPhysicalDevicePresentTimingFeaturesEXT>(features))
    timing->presentTiming = timing->presentAtAbsoluteTime = timing->presentAtRelativeTime = VK_TRUE;
}

PFN_vkVoidFunction VKAPI_CALL getFeatureInstanceProc(VkInstance instance, const char *name) {
  if (std::strcmp(name, "vkGetPhysicalDeviceFeatures2") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(driverTimingFeatures);
  return getCalibrationInstanceProc(instance, name);
}

struct AdvertisedTiming {
  VkPhysicalDevicePresentTimingFeaturesEXT features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT};
  bool extension = false;
};

AdvertisedTiming queryAdvertisedTiming(VkInstance instance) {
  vkroots::VkInstanceDispatch instanceDispatch(getFeatureInstanceProc, instance, nullptr);
  vkroots::VkPhysicalDeviceDispatch dispatch(VK_NULL_HANDLE, &instanceDispatch);
  AdvertisedTiming advertised;
  VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &advertised.features};
  GamescopeWSILayer::VkInstanceOverrides::GetPhysicalDeviceFeatures2(dispatch, VK_NULL_HANDLE, &features);
  uint32_t count = 0;
  REQUIRE(GamescopeWSILayer::VkInstanceOverrides::EnumerateDeviceExtensionProperties(dispatch, VK_NULL_HANDLE, nullptr, &count, nullptr) == VK_SUCCESS);
  std::vector<VkExtensionProperties> extensions(count);
  REQUIRE(GamescopeWSILayer::VkInstanceOverrides::EnumerateDeviceExtensionProperties(dispatch, VK_NULL_HANDLE, nullptr, &count, extensions.data()) == VK_SUCCESS);
  advertised.extension = GamescopeWSILayer::hasExtension(extensions, VK_EXT_PRESENT_TIMING_EXTENSION_NAME);
  return advertised;
}
}

TEST_CASE("Present timing is advertised only for a timing-capable gamescope server", "[present_timing_layer]") {
  driver = {};
  driver.khr = true;
  const auto instance = reinterpret_cast<VkInstance>(uintptr_t(1101));

  SECTION("a foreign instance keeps the driver's features") {
    const auto advertised = queryAdvertisedTiming(instance);
    CHECK(advertised.features.presentTiming);
    CHECK(advertised.features.presentAtAbsoluteTime);
    CHECK(advertised.features.presentAtRelativeTime);
    CHECK_FALSE(advertised.extension);
  }

  SECTION("a version 1 server exposes nothing") {
    REQUIRE(GamescopeWSILayer::gamescopeInstances.create(instance, GamescopeWSILayer::GamescopeInstanceData{.swapchainFactoryVersion = 1}));
    const auto advertised = queryAdvertisedTiming(instance);
    CHECK_FALSE(advertised.features.presentTiming);
    CHECK_FALSE(advertised.features.presentAtAbsoluteTime);
    CHECK_FALSE(advertised.features.presentAtRelativeTime);
    CHECK_FALSE(advertised.extension);
  }

  SECTION("a version 2 server exposes the extension") {
    REQUIRE(GamescopeWSILayer::gamescopeInstances.create(instance, GamescopeWSILayer::GamescopeInstanceData{.swapchainFactoryVersion = 2}));
    const auto advertised = queryAdvertisedTiming(instance);
    CHECK(advertised.features.presentTiming);
    CHECK(advertised.extension);
  }

  SECTION("a version 2 server still needs the driver's dependencies") {
    driver.khr = false;
    REQUIRE(GamescopeWSILayer::gamescopeInstances.create(instance, GamescopeWSILayer::GamescopeInstanceData{.swapchainFactoryVersion = 2}));
    const auto advertised = queryAdvertisedTiming(instance);
    CHECK_FALSE(advertised.features.presentTiming);
    CHECK_FALSE(advertised.extension);
  }

  GamescopeWSILayer::gamescopeInstances.erase(instance);
}

namespace {
struct MaintenanceDriver {
  uint32_t api = VK_API_VERSION_1_1;
  bool extension = true;
  bool feature = true;
  uint32_t coreCalls = 0;
  uint32_t khrCalls = 0;
  bool enabled = false;
  bool forced = false;
} maintenance;

void VKAPI_CALL maintenanceProperties(VkPhysicalDevice, VkPhysicalDeviceProperties *properties) {
  properties->apiVersion = maintenance.api;
}

VkResult VKAPI_CALL maintenanceExtensions(VkPhysicalDevice, const char *, uint32_t *count, VkExtensionProperties *properties) {
  std::vector<VkExtensionProperties> extensions;
  if (maintenance.extension)
    extensions.push_back({VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME, VK_EXT_SWAPCHAIN_MAINTENANCE_1_SPEC_VERSION});
  return vkroots::array(extensions, count, properties);
}

void maintenanceFeatures(VkPhysicalDeviceFeatures2 *features) {
  if (auto *feature = vkroots::FindInChainMutable<VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT>(features))
    feature->swapchainMaintenance1 = maintenance.feature;
}

void VKAPI_CALL maintenanceCore(VkPhysicalDevice, VkPhysicalDeviceFeatures2 *features) {
  maintenance.coreCalls++;
  maintenanceFeatures(features);
}

void VKAPI_CALL maintenanceKHR(VkPhysicalDevice, VkPhysicalDeviceFeatures2 *features) {
  maintenance.khrCalls++;
  maintenanceFeatures(features);
}

VkResult VKAPI_CALL maintenanceCreateDevice(VkPhysicalDevice, const VkDeviceCreateInfo *info,
                                          const VkAllocationCallbacks *, VkDevice *) {
  const std::vector<const char *> extensions(info->ppEnabledExtensionNames,
    info->ppEnabledExtensionNames + info->enabledExtensionCount);
  maintenance.enabled = GamescopeWSILayer::contains(extensions, VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
  if (auto *feature = vkroots::FindInChain<VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT>(info))
    maintenance.forced = feature->swapchainMaintenance1;
  // Stop after inspecting the forwarded request; no real device is allocated.
  return VK_ERROR_INITIALIZATION_FAILED;
}

PFN_vkVoidFunction VKAPI_CALL maintenanceProc(VkInstance instance, const char *name) {
  if (std::strcmp(name, "vkGetPhysicalDeviceProperties") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(maintenanceProperties);
  if (std::strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(maintenanceExtensions);
  if (std::strcmp(name, "vkGetPhysicalDeviceFeatures2") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(maintenanceCore);
  if (std::strcmp(name, "vkGetPhysicalDeviceFeatures2KHR") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(maintenanceKHR);
  if (std::strcmp(name, "vkCreateDevice") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(maintenanceCreateDevice);
  return getInstanceProc(instance, name);
}
}

TEST_CASE("Maintenance injection respects enabled instance dependencies and query aliases", "[present_timing_layer]") {
  maintenance = {};
  VkApplicationInfo app{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
  std::vector<const char *> extensions{VK_KHR_SURFACE_EXTENSION_NAME,
    VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME, VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME};
  uint32_t expectedCore = 1;
  uint32_t expectedKHR = 0;
  bool expected = true;
  bool deviceSwapchain = true;
  bool appInfo = true;

  SECTION("core 1.1") {}
  SECTION("API 1.0 without properties2 never queries features") {
    app.apiVersion = VK_API_VERSION_1_0;
    expected = false;
    expectedCore = 0;
  }
  SECTION("API 1.0 with properties2 uses only KHR") {
    app.apiVersion = VK_API_VERSION_1_0;
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    expectedCore = 0;
    expectedKHR = 1;
  }
  SECTION("physical API 1.0 without KHR never queries features") {
    maintenance.api = VK_API_VERSION_1_0;
    expected = false;
    expectedCore = 0;
  }
  SECTION("physical API 1.0 with KHR uses only KHR") {
    maintenance.api = VK_API_VERSION_1_0;
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    expectedCore = 0;
    expectedKHR = 1;
  }
  SECTION("an omitted API version defaults to 1.0") {
    app.apiVersion = 0;
    expected = false;
    expectedCore = 0;
  }
  SECTION("omitted application info defaults to 1.0") {
    appInfo = false;
    expected = false;
    expectedCore = 0;
  }
  SECTION("each missing instance dependency disables injection") {
    const auto index = GENERATE(0, 1, 2);
    extensions.erase(extensions.begin() + index);
    expected = false;
    expectedCore = 0;
  }
  SECTION("driver extension absent") {
    maintenance.extension = false;
    expected = false;
    expectedCore = 0;
  }
  SECTION("driver feature absent") {
    maintenance.feature = false;
    expected = false;
  }
  SECTION("a device without swapchain does not acquire its dependencies") {
    deviceSwapchain = false;
    expected = false;
    expectedCore = 0;
  }

  const auto instance = reinterpret_cast<VkInstance>(uintptr_t(1201));
  auto *state = GamescopeWSILayer::gamescopeInstances.create(instance, GamescopeWSILayer::GamescopeInstanceData{});
  REQUIRE(state);
  const VkInstanceCreateInfo info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = appInfo ? &app : nullptr,
    .enabledExtensionCount = uint32_t(extensions.size()), .ppEnabledExtensionNames = extensions.data()};
  state->setVulkanCapabilities(info);
  vkroots::VkInstanceDispatch instanceDispatch(maintenanceProc, instance, nullptr);
  vkroots::VkPhysicalDeviceDispatch dispatch(VK_NULL_HANDLE, &instanceDispatch);
  const char *swapchain = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
  VkLayerDeviceLink link{.pfnNextGetDeviceProcAddr = getDeviceProc};
  VkLayerDeviceCreateInfo layer{.sType = VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO, .function = VK_LAYER_LINK_INFO,
    .u = {.pLayerInfo = &link}};
  const VkDeviceCreateInfo deviceInfo{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &layer,
    .enabledExtensionCount = uint32_t(deviceSwapchain), .ppEnabledExtensionNames = &swapchain};
  VkDevice device{};
  REQUIRE(GamescopeWSILayer::VkInstanceOverrides::CreateDevice(dispatch, VK_NULL_HANDLE, &deviceInfo, nullptr, &device) == VK_ERROR_INITIALIZATION_FAILED);
  CHECK(maintenance.enabled == expected);
  CHECK(maintenance.forced == expected);
  CHECK(maintenance.coreCalls == expectedCore);
  CHECK(maintenance.khrCalls == expectedKHR);
  GamescopeWSILayer::gamescopeInstances.erase(instance);
}

namespace {
struct SurfaceDriver {
  VkSurfaceCapabilitiesKHR capabilities{
    .minImageCount = 3, .maxImageCount = 0,
    .currentExtent = {640, 480}, .minImageExtent = {64, 64}, .maxImageExtent = {4096, 4096},
    .maxImageArrayLayers = 2,
    .supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
    .currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
    .supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    .supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
  };
  VkResult result = VK_SUCCESS;
  VkSurfaceKHR surface{};
  VkSwapchainCreateInfoKHR creation{};
  std::vector<VkPresentModeKHR> modes;
  uint32_t creates = 0;
  uint32_t acquires = 0;
  VkAcquireNextImageInfoKHR acquire{};
} surfaceDriver;

VkResult VKAPI_CALL surfaceCapabilities(VkPhysicalDevice, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR *caps) {
  surfaceDriver.surface = surface;
  if (surfaceDriver.result == VK_SUCCESS)
    *caps = surfaceDriver.capabilities;
  return surfaceDriver.result;
}

VkResult VKAPI_CALL surfaceCapabilities2(VkPhysicalDevice, const VkPhysicalDeviceSurfaceInfo2KHR *info,
                                        VkSurfaceCapabilities2KHR *caps) {
  if (auto *protectedCaps = vkroots::FindInChainMutable<VkSurfaceProtectedCapabilitiesKHR>(caps))
    protectedCaps->supportsProtected = VK_TRUE;
  return surfaceCapabilities(VK_NULL_HANDLE, info->surface, &caps->surfaceCapabilities);
}

VkResult VKAPI_CALL surfaceFormats(VkPhysicalDevice, VkSurfaceKHR, uint32_t *count, VkSurfaceFormatKHR *formats) {
  const std::array values{VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}};
  return vkroots::array(values, count, formats);
}

PFN_vkVoidFunction VKAPI_CALL surfaceContractProc(VkInstance instance, const char *name) {
  if (std::strcmp(name, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(surfaceCapabilities);
  if (std::strcmp(name, "vkGetPhysicalDeviceSurfaceCapabilities2KHR") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(surfaceCapabilities2);
  if (std::strcmp(name, "vkGetPhysicalDeviceSurfaceFormatsKHR") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(surfaceFormats);
  return maintenanceProc(instance, name);
}

VkResult VKAPI_CALL createContractSwapchain(VkDevice, const VkSwapchainCreateInfoKHR *info,
                                           const VkAllocationCallbacks *, VkSwapchainKHR *) {
  surfaceDriver.creates++;
  surfaceDriver.creation = *info;
  surfaceDriver.creation.pNext = nullptr;
  if (auto *modes = vkroots::FindInChain<VkSwapchainPresentModesCreateInfoEXT>(info))
    surfaceDriver.modes.assign(modes->pPresentModes, modes->pPresentModes + modes->presentModeCount);
  // Inspect the actual forwarded request without allocating driver or protocol objects.
  return VK_ERROR_INITIALIZATION_FAILED;
}

VkResult VKAPI_CALL acquireContractImage(VkDevice, const VkAcquireNextImageInfoKHR *info, uint32_t *index) {
  surfaceDriver.acquires++;
  surfaceDriver.acquire = *info;
  *index = 2;
  return VK_SUBOPTIMAL_KHR;
}

PFN_vkVoidFunction VKAPI_CALL surfaceContractDeviceProc(VkDevice device, const char *name) {
  if (std::strcmp(name, "vkCreateSwapchainKHR") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(createContractSwapchain);
  if (std::strcmp(name, "vkAcquireNextImage2KHR") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(acquireContractImage);
  return getDeviceProc(device, name);
}

struct SurfaceContract {
  VkInstance instance = reinterpret_cast<VkInstance>(uintptr_t(1301));
  VkSurfaceKHR surface = fakeHandle<VkSurfaceKHR>(1302);
  vkroots::VkInstanceDispatch instanceDispatch{surfaceContractProc, instance, nullptr};
  vkroots::VkPhysicalDeviceDispatch physical{VK_NULL_HANDLE, &instanceDispatch};
  VkDeviceCreateInfo info{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  vkroots::VkDeviceDispatch device{surfaceContractDeviceProc, VK_NULL_HANDLE, VK_NULL_HANDLE, &physical, &info};

  explicit SurfaceContract(bool tracked = true) {
    surfaceDriver = {};
    maintenance = {};
    REQUIRE(GamescopeWSILayer::gamescopeInstances.create(instance, GamescopeWSILayer::GamescopeInstanceData{
      .apiVersion = VK_API_VERSION_1_1, .surfaceMaintenance1 = true}));
    if (tracked) {
      auto *state = GamescopeWSILayer::gamescopeSurfaces.create(surface, GamescopeWSILayer::GamescopeSurfaceData{
        .instance = instance});
      REQUIRE(state);
      state->waylandObjects.limiterState = std::make_shared<GamescopeWSILayer::GamescopeLimiterState>();
      state->waylandObjects.limiterState->state = 0;
    }
  }

  ~SurfaceContract() {
    GamescopeWSILayer::gamescopeSurfaces.erase(surface);
    GamescopeWSILayer::gamescopeInstances.erase(instance);
  }
};
}

TEST_CASE("Surface capability queries preserve driver fields and caller chains", "[present_timing_layer][surface_counts]") {
  const bool tracked = GENERATE(false, true);
  const bool caps2 = GENERATE(false, true);
  SurfaceContract fixture(tracked);
  const auto result = GENERATE(VK_SUCCESS, VK_ERROR_SURFACE_LOST_KHR);
  surfaceDriver.result = result;
  VkSurfaceCapabilitiesKHR caps{};
  caps.minImageCount = 17;
  if (caps2) {
    VkSurfacePresentModeEXT mode{.sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_EXT,
      .presentMode = VK_PRESENT_MODE_MAILBOX_KHR};
    const VkPhysicalDeviceSurfaceInfo2KHR info{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
      .pNext = &mode, .surface = fixture.surface};
    VkSurfaceProtectedCapabilitiesKHR protectedCaps{.sType = VK_STRUCTURE_TYPE_SURFACE_PROTECTED_CAPABILITIES_KHR};
    VkSurfaceCapabilities2KHR output{.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR,
      .pNext = &protectedCaps, .surfaceCapabilities = caps};
    CHECK(GamescopeWSILayer::VkInstanceOverrides::GetPhysicalDeviceSurfaceCapabilities2KHR(
      fixture.physical, VK_NULL_HANDLE, &info, &output) == result);
    CHECK(info.pNext == &mode);
    CHECK(mode.pNext == nullptr);
    CHECK(mode.presentMode == VK_PRESENT_MODE_MAILBOX_KHR);
    CHECK(output.pNext == &protectedCaps);
    CHECK(protectedCaps.pNext == nullptr);
    CHECK(protectedCaps.supportsProtected == VK_TRUE);
    caps = output.surfaceCapabilities;
  } else {
    CHECK(GamescopeWSILayer::VkInstanceOverrides::GetPhysicalDeviceSurfaceCapabilitiesKHR(
      fixture.physical, VK_NULL_HANDLE, fixture.surface, &caps) == result);
  }
  CHECK(surfaceDriver.surface == fixture.surface);
  if (result == VK_SUCCESS) {
    const auto &expected = surfaceDriver.capabilities;
    CHECK(caps.minImageCount == expected.minImageCount);
    CHECK(caps.maxImageCount == expected.maxImageCount);
    CHECK(caps.currentExtent.width == expected.currentExtent.width);
    CHECK(caps.currentExtent.height == expected.currentExtent.height);
    CHECK(caps.minImageExtent.width == expected.minImageExtent.width);
    CHECK(caps.minImageExtent.height == expected.minImageExtent.height);
    CHECK(caps.maxImageExtent.width == expected.maxImageExtent.width);
    CHECK(caps.maxImageExtent.height == expected.maxImageExtent.height);
    CHECK(caps.maxImageArrayLayers == expected.maxImageArrayLayers);
    CHECK(caps.supportedTransforms == expected.supportedTransforms);
    CHECK(caps.currentTransform == expected.currentTransform);
    CHECK(caps.supportedCompositeAlpha == expected.supportedCompositeAlpha);
    CHECK(caps.supportedUsageFlags == expected.supportedUsageFlags);
  } else
    CHECK(caps.minImageCount == 17);
}

TEST_CASE("Swapchain creation forwards a valid count and maintenance modes without changing the caller", "[present_timing_layer][surface_counts]") {
  const auto variant = GENERATE(0, 1, 2);
  SurfaceContract fixture;
  maintenance.extension = variant != 0;
  const VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
  VkSwapchainPresentModesCreateInfoEXT modes{.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT,
    .presentModeCount = 1, .pPresentModes = &mode};
  const VkSwapchainCreateInfoKHR info{.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
    .pNext = variant == 2 ? &modes : nullptr, .surface = fixture.surface, .minImageCount = 4,
    .imageFormat = VK_FORMAT_B8G8R8A8_UNORM, .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
    .imageExtent = {640, 480}, .imageArrayLayers = 1, .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    .presentMode = mode, .clipped = VK_TRUE};
  const auto original = info;
  VkSwapchainKHR swapchain{};
  CHECK(GamescopeWSILayer::VkDeviceOverrides::CreateSwapchainKHR(
    fixture.device, VK_NULL_HANDLE, &info, nullptr, &swapchain) == VK_ERROR_INITIALIZATION_FAILED);
  REQUIRE(surfaceDriver.creates == 1);
  CHECK(surfaceDriver.creation.surface == fixture.surface);
  CHECK(surfaceDriver.creation.minImageCount == 4);
  CHECK(surfaceDriver.creation.presentMode == VK_PRESENT_MODE_MAILBOX_KHR);
  CHECK(surfaceDriver.modes == (variant == 0 ? std::vector<VkPresentModeKHR>{} :
    std::vector<VkPresentModeKHR>{VK_PRESENT_MODE_MAILBOX_KHR}));
  CHECK(info.pNext == original.pNext);
  CHECK(info.flags == original.flags);
  CHECK(info.surface == original.surface);
  CHECK(info.minImageCount == original.minImageCount);
  CHECK(info.imageColorSpace == original.imageColorSpace);
  CHECK(info.presentMode == original.presentMode);
  CHECK(modes.pNext == nullptr);
  CHECK(modes.presentModeCount == 1);
  CHECK(modes.pPresentModes == &mode);
  CHECK(mode == VK_PRESENT_MODE_FIFO_KHR);
}

TEST_CASE("Acquire aliases forward active requests and reject retired swapchains first", "[present_timing_layer][surface_counts]") {
  const bool retired = GENERATE(false, true);
  const bool legacy = GENERATE(false, true);
  SurfaceContract fixture;
  const auto handle = fakeHandle<VkSwapchainKHR>(1303);
  auto *state = GamescopeWSILayer::gamescopeSwapchains.create(handle, GamescopeWSILayer::GamescopeSwapchainData{});
  REQUIRE(state);
  *state->retired = retired;
  const VkAcquireNextImageInfoKHR info{.sType = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR,
    .swapchain = handle, .timeout = 42, .semaphore = fakeHandle<VkSemaphore>(1304), .deviceMask = 1};
  uint32_t index = 99;
  const auto result = legacy ? GamescopeWSILayer::VkDeviceOverrides::AcquireNextImageKHR(
    fixture.device, VK_NULL_HANDLE, handle, info.timeout, info.semaphore, info.fence, &index) :
    GamescopeWSILayer::VkDeviceOverrides::AcquireNextImage2KHR(fixture.device, VK_NULL_HANDLE, &info, &index);
  CHECK(result == (retired ? VK_ERROR_OUT_OF_DATE_KHR : VK_SUBOPTIMAL_KHR));
  CHECK(surfaceDriver.acquires == (retired ? 0 : 1));
  CHECK(index == (retired ? 99 : 2));
  if (!retired) {
    CHECK(surfaceDriver.acquire.swapchain == handle);
    CHECK(surfaceDriver.acquire.timeout == info.timeout);
    CHECK(surfaceDriver.acquire.semaphore == info.semaphore);
    CHECK(surfaceDriver.acquire.fence == info.fence);
    CHECK(surfaceDriver.acquire.deviceMask == info.deviceMask);
  }
  GamescopeWSILayer::gamescopeSwapchains.erase(handle);
}

TEST_CASE("Features2 aliases retain their driver entry point", "[present_timing_layer]") {
  maintenance = {};
  vkroots::VkInstanceDispatch instanceDispatch(maintenanceProc, VK_NULL_HANDLE, nullptr);
  vkroots::VkPhysicalDeviceDispatch dispatch(VK_NULL_HANDLE, &instanceDispatch);
  VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  SECTION("KHR") {
    GamescopeWSILayer::VkInstanceOverrides::GetPhysicalDeviceFeatures2KHR(dispatch, VK_NULL_HANDLE, &features);
    CHECK(maintenance.khrCalls == 1);
    CHECK(maintenance.coreCalls == 0);
  }
  SECTION("core") {
    GamescopeWSILayer::VkInstanceOverrides::GetPhysicalDeviceFeatures2(dispatch, VK_NULL_HANDLE, &features);
    CHECK(maintenance.coreCalls == 1);
    CHECK(maintenance.khrCalls == 0);
  }
}

TEST_CASE("Local-only present calibration does not call the driver", "[present_timing_layer]") {
  Calibration calibration;
  REQUIRE(calibration.run(1) == VK_SUCCESS);
  REQUIRE(driver.calls == 0);
  REQUIRE(calibration.sampledDuringCall(0));
  REQUIRE(calibration.deviation <= calibration.after - calibration.before);
}

TEST_CASE("Timestamp lookup preserves enabled aliases for Zink fallback", "[present_timing_layer][calibration_lookup]") {
  Calibration calibration;
  SECTION("KHR only") { driver.khr = true; }
  SECTION("EXT only") { driver.ext = true; }
  SECTION("Both aliases") { driver.khr = driver.ext = true; }
  SECTION("Neither alias") {}

  VkInstance instance = reinterpret_cast<VkInstance>(uintptr_t(201));
  VkDevice device = reinterpret_cast<VkDevice>(uintptr_t(202));
  VkDeviceCreateInfo createInfo{ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  auto *instanceDispatch = vkroots::tables::InstanceDispatches.create(instance, getInstanceProc, instance, getInstanceProc);
  REQUIRE(instanceDispatch != nullptr);
  REQUIRE(vkroots::tables::DeviceDispatches.create(device, getCalibrationProc, device,
    nullptr, nullptr, &createInfo) != nullptr);
  struct Cleanup {
    VkDevice device;
    VkInstance instance;
    ~Cleanup() {
      vkroots::tables::DestroyDispatchTable(device);
      vkroots::tables::DestroyDispatchTable(instance);
    }
  } cleanup{ device, instance };

  VkNegotiateLayerInterface loader{ .sType = LAYER_NEGOTIATE_INTERFACE_STRUCT, .loaderLayerInterfaceVersion = 2 };
  REQUIRE(vkNegotiateLoaderLayerInterfaceVersion(&loader) == VK_SUCCESS);
  auto instanceProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(loader.pfnGetInstanceProcAddr(instance, "vkGetInstanceProcAddr"));
  REQUIRE(instanceProc != nullptr);
  const std::array lookups = {
    loader.pfnGetDeviceProcAddr,
    reinterpret_cast<PFN_vkGetDeviceProcAddr>(loader.pfnGetInstanceProcAddr(instance, "vkGetDeviceProcAddr")),
    reinterpret_cast<PFN_vkGetDeviceProcAddr>(loader.pfnGetDeviceProcAddr(device, "vkGetDeviceProcAddr")),
    reinterpret_cast<PFN_vkGetDeviceProcAddr>(instanceProc(instance, "vkGetDeviceProcAddr")),
  };
  for (auto lookup : lookups) {
    CHECK(lookup != nullptr);
    if (!lookup)
      continue;
    CHECK(lookup(device, "vkGetPastPresentationTimingEXT") != nullptr);
    for (const auto &[name, enabled] : std::array{
        std::pair{ "vkGetCalibratedTimestampsEXT", driver.ext },
        std::pair{ "vkGetCalibratedTimestampsKHR", driver.khr } }) {
      auto timestamp = reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(lookup(device, name));
      CHECK((timestamp != nullptr) == enabled);
      if (!enabled || !timestamp)
        continue;

      // Enabled aliases still pass through the layer for mixed local/device queries.
      calibration.before = GamescopeWSILayer::getTimeMonotonic();
      REQUIRE(timestamp(device, 2, calibration.infos.data(), calibration.timestamps.data(), &calibration.deviation) == VK_SUCCESS);
      calibration.after = GamescopeWSILayer::getTimeMonotonic();
      CHECK(calibration.sampledDuringCall(0));
      CHECK(calibration.timestamps[1] == 1000);
    }
  }
}

TEST_CASE("Time-domain lookup preserves available aliases and local domains", "[present_timing_layer][calibration_lookup][calibration_domains]") {
  driver = {};
  SECTION("KHR only") { driver.khr = true; }
  SECTION("EXT only") { driver.ext = true; }
  SECTION("Both aliases") { driver.khr = driver.ext = true; }
  SECTION("Neither alias") {}

  VkInstance instance = reinterpret_cast<VkInstance>(uintptr_t(203));
  VkPhysicalDevice physicalDevice = reinterpret_cast<VkPhysicalDevice>(uintptr_t(204));
  auto *dispatch = vkroots::tables::InstanceDispatches.create(instance, getCalibrationInstanceProc, instance, getCalibrationInstanceProc);
  REQUIRE(dispatch != nullptr);
  REQUIRE(vkroots::tables::AssignDispatchTable(physicalDevice, dispatch) != nullptr);
  REQUIRE(GamescopeWSILayer::gamescopeInstances.create(instance, GamescopeWSILayer::GamescopeInstanceData{
    .swapchainFactoryVersion = 2,
  }) != nullptr);
  struct Cleanup {
    VkPhysicalDevice physicalDevice;
    VkInstance instance;
    ~Cleanup() {
      GamescopeWSILayer::gamescopeInstances.erase(instance);
      vkroots::tables::UnassignDispatchTable(physicalDevice);
      vkroots::tables::DestroyDispatchTable(instance);
    }
  } cleanup{ physicalDevice, instance };

  VkNegotiateLayerInterface loader{ .sType = LAYER_NEGOTIATE_INTERFACE_STRUCT, .loaderLayerInterfaceVersion = 2 };
  REQUIRE(vkNegotiateLoaderLayerInterfaceVersion(&loader) == VK_SUCCESS);
  const std::array lookups = {
    loader.pfnGetInstanceProcAddr,
    reinterpret_cast<PFN_vkGetInstanceProcAddr>(loader.pfnGetInstanceProcAddr(instance, "vkGetInstanceProcAddr")),
    loader.pfnGetPhysicalDeviceProcAddr,
    reinterpret_cast<PFN_GetPhysicalDeviceProcAddr>(loader.pfnGetInstanceProcAddr(instance, "vk_layerGetPhysicalDeviceProcAddr")),
  };
  for (auto lookup : lookups) {
    REQUIRE(lookup != nullptr);
    for (const auto &[name, available] : std::array{
        std::pair{ "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT", driver.ext },
        std::pair{ "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR", driver.khr } }) {
      auto query = reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(lookup(instance, name));
      CHECK((query != nullptr) == available);
      if (!available || !query)
        continue;

      uint32_t count = 0;
      REQUIRE(query(physicalDevice, &count, nullptr) == VK_SUCCESS);
      CHECK(count == (driver.khr ? 3u : 2u));
      std::array<VkTimeDomainKHR, 3> domains{};
      count = domains.size();
      REQUIRE(query(physicalDevice, &count, domains.data()) == VK_SUCCESS);
      CHECK(domains[0] == VK_TIME_DOMAIN_DEVICE_KHR);
      CHECK(domains[1] == VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR);
      if (driver.khr)
        CHECK(domains[2] == VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT);
    }
  }
}

TEST_CASE("Present calibration merges both local aliases with driver timestamps", "[present_timing_layer]") {
  Calibration calibration;
  REQUIRE(calibration.run(4) == VK_SUCCESS);
  REQUIRE(driver.calls == 1);
  REQUIRE(driver.count == 2);
  REQUIRE(driver.domains == std::vector<VkTimeDomainKHR>{VK_TIME_DOMAIN_DEVICE_KHR, VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT});
  REQUIRE(calibration.sampledDuringCall(0));
  REQUIRE(calibration.sampledDuringCall(2));
  REQUIRE(calibration.timestamps[0] == calibration.timestamps[2]);
  REQUIRE(calibration.timestamps[1] == 1000);
  REQUIRE(calibration.timestamps[3] == 1001);
  REQUIRE(calibration.deviation >= 7);
  REQUIRE(calibration.infos[0].pNext == &calibration.local);
  REQUIRE(calibration.infos[3].pNext == &calibration.foreign);
}

TEST_CASE("Non-layer calibration is forwarded untouched", "[present_timing_layer]") {
  Calibration calibration;
  calibration.infos[0].pNext = &calibration.foreign;
  REQUIRE(calibration.run(1) == VK_SUCCESS);
  REQUIRE(driver.original == calibration.infos.data());
  REQUIRE(calibration.timestamps[0] == 1000);
  REQUIRE(calibration.deviation == 7);
}

TEST_CASE("Present calibration preserves driver errors", "[present_timing_layer]") {
  Calibration calibration;
  driver.result = VK_ERROR_DEVICE_LOST;
  REQUIRE(calibration.run(2) == VK_ERROR_DEVICE_LOST);
  REQUIRE(calibration.timestamps == std::array<uint64_t, 4>{});
  REQUIRE(calibration.deviation == 0);
}

TEST_CASE("Present calibration deviation saturates", "[present_timing_layer]") {
  Calibration calibration;
  driver.deviation = std::numeric_limits<uint64_t>::max();
  REQUIRE(calibration.run(2) == VK_SUCCESS);
  REQUIRE(calibration.deviation == std::numeric_limits<uint64_t>::max());
}

TEST_CASE("timespecToNanos does not wrap on 32-bit", "[present_timing_layer]") {
  timespec spec{5, 1};
  CHECK(GamescopeWSILayer::timespecToNanos(spec) == 5'000'000'001);
}

TEST_CASE("A refused present keeps every image acquired and reports each", "[present_timing_layer]") {
  refused = {};
  VkDeviceCreateInfo createInfo{ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  vkroots::VkDeviceDispatch dispatch(getDeviceProc, VK_NULL_HANDLE, VK_NULL_HANDLE, nullptr, &createInfo);
  vkroots::VkQueueDispatch queueDispatch(VK_NULL_HANDLE, &dispatch);
  std::array<VkSwapchainKHR, 2> swapchains{fakeHandle<VkSwapchainKHR>(1), fakeHandle<VkSwapchainKHR>(2)};
  std::array<uint32_t, 2> images{0, 1};
  std::array<VkResult, 2> results{};
  VkSemaphore semaphore = fakeHandle<VkSemaphore>(5);
  VkPresentInfoKHR info{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, nullptr, 1, &semaphore, 2,
    swapchains.data(), images.data(), results.data() };

  REQUIRE(GamescopeWSILayer::VkDeviceOverrides::PresentWithoutImage(queueDispatch, VK_NULL_HANDLE, &info,
    VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT) == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT);
  REQUIRE(refused.submits == 1);
  REQUIRE(refused.idles == 0);
  REQUIRE(refused.released.empty());
  REQUIRE(results[0] == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT);
  REQUIRE(results[1] == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT);
}

TEST_CASE("Layer instance teardown does not read the erased dispatch", "[present_timing_layer]") {
  destroyedInstances = 0;
  VkInstance instance = reinterpret_cast<VkInstance>(uintptr_t(101));
  auto *dispatch = vkroots::tables::InstanceDispatches.create(instance, getInstanceProc, instance, getInstanceProc);
  REQUIRE(dispatch != nullptr);
  GamescopeWSILayer::VkInstanceOverrides::DestroyInstance(*dispatch, instance, nullptr);
  REQUIRE(destroyedInstances == 1);
  REQUIRE(vkroots::LookupDispatch(instance) == nullptr);
}

TEST_CASE("Layer device teardown does not read the erased dispatch", "[present_timing_layer]") {
  destroyedDevices = 0;
  VkDevice device = reinterpret_cast<VkDevice>(uintptr_t(102));
  VkDeviceCreateInfo createInfo{ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  auto *dispatch = vkroots::tables::DeviceDispatches.create(device, getDeviceProc, device,
    nullptr, nullptr, &createInfo);
  REQUIRE(dispatch != nullptr);
  GamescopeWSILayer::VkDeviceOverrides::DestroyDevice(*dispatch, device, nullptr);
  REQUIRE(destroyedDevices == 1);
  REQUIRE(vkroots::LookupDispatch(device) == nullptr);
}

namespace {

struct TimingRecord {
  VkPresentStageTimeEXT stage{};
  VkPastPresentationTimingEXT timing{.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT, .presentStageCount = 1, .pPresentStages = &stage};
  VkPastPresentationTimingInfoEXT info{.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT};
  VkPastPresentationTimingPropertiesEXT properties{.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT,
    .presentationTimingCount = 1, .pPresentationTimings = &timing};

  VkResult query(GamescopeWSILayer::GamescopeSwapchainData &swapchain) {
    std::lock_guard lock(*swapchain.presentTimingMutex);
    return swapchain.timingQueue.getPast(&info, &properties);
  }
};

struct PresentBatch {
  int sockets[2];
  wl_display *display;
  wl_registry *registry;
  wl_event_queue *globalsQueue;
  gamescope_swapchain_factory_v2 *factory;
  std::array<VkSwapchainKHR, 2> swapchains{fakeHandle<VkSwapchainKHR>(501), fakeHandle<VkSwapchainKHR>(502)};
  std::array<VkSurfaceKHR, 2> surfaces{fakeHandle<VkSurfaceKHR>(601), fakeHandle<VkSurfaceKHR>(602)};
  std::array<GamescopeWSILayer::GamescopeSwapchainData *, 2> states{};
  std::array<wl_surface *, 2> wlSurfaces{};
  std::array<uint32_t, 2> images{2, 1};
  std::array<uint64_t, 2> ids{71, 92};
  std::array<VkFence, 2> fences{fakeHandle<VkFence>(701), fakeHandle<VkFence>(702)};
  std::array<VkPresentTimingInfoEXT, 2> timings{{
    {.sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT, .presentStageQueries = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT},
    {.sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT, .targetTime = 9000, .presentStageQueries = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT},
  }};
  std::array<VkResult, 2> results{};
  VkPresentTimingsInfoEXT timingInfo{VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT, nullptr, 2, timings.data()};
  VkPresentId2KHR idInfo{VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR, &timingInfo, 2, ids.data()};
  VkSwapchainPresentFenceInfoEXT fenceInfo{VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT, &idInfo, 2, fences.data()};
  VkSemaphore semaphore = fakeHandle<VkSemaphore>(801);
  VkPresentInfoKHR info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, &fenceInfo, 1, &semaphore, 2, swapchains.data(), images.data(), results.data()};
  VkDeviceCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  vkroots::VkDeviceDispatch device{getDeviceProc, VK_NULL_HANDLE, VK_NULL_HANDLE, nullptr, &createInfo};
  vkroots::VkQueueDispatch queue{VK_NULL_HANDLE, &device};

  explicit PresentBatch(bool foreign = false) {
    refused = {};
    presented.clear();
    presentCalls = failPresent = 0;
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    display = wl_display_connect_to_fd(sockets[0]);
    REQUIRE(display);
    registry = wl_display_get_registry(display);
    globalsQueue = wl_display_create_queue(display);
    factory = static_cast<gamescope_swapchain_factory_v2 *>(wl_registry_bind(registry, 1, &gamescope_swapchain_factory_v2_interface, 2));
    for (size_t i = 0; i < (foreign ? 1u : 2u); i++) {
      auto *eventQueue = wl_display_create_queue(display);
      auto *wrapper = static_cast<gamescope_swapchain_factory_v2 *>(wl_proxy_create_wrapper(factory));
      wl_proxy_set_queue(reinterpret_cast<wl_proxy *>(wrapper), eventQueue);
      wlSurfaces[i] = reinterpret_cast<wl_surface *>(wl_proxy_create(reinterpret_cast<wl_proxy *>(display), &wl_surface_interface));
      auto *object = gamescope_swapchain_factory_v2_create_swapchain(wrapper, wlSurfaces[i]);
      wl_proxy_wrapper_destroy(wrapper);
      // No X server is needed: the surface's native path makes the size/bypass check deterministic.
      REQUIRE(GamescopeWSILayer::gamescopeSurfaces.create(surfaces[i], GamescopeWSILayer::GamescopeSurfaceData{}) != nullptr);
      states[i] = GamescopeWSILayer::gamescopeSwapchains.create(swapchains[i], GamescopeWSILayer::GamescopeSwapchainData{
        .object = object, .display = display, .globalsQueue = globalsQueue, .eventQueue = eventQueue, .surface = surfaces[i],
        .isBypassingXWayland = true, .presentMode = VK_PRESENT_MODE_FIFO_KHR, .presentTimingEnabled = true});
      REQUIRE(states[i]);
      states[i]->timingQueue.setSize(4);
    }
  }

  ~PresentBatch() {
    for (size_t i = 0; i < states.size(); i++) {
      if (!states[i]) continue;
      gamescope_swapchain_destroy(states[i]->object);
      wl_event_queue_destroy(states[i]->eventQueue);
      GamescopeWSILayer::gamescopeSwapchains.erase(swapchains[i]);
      GamescopeWSILayer::gamescopeSurfaces.erase(surfaces[i]);
      wl_surface_destroy(wlSurfaces[i]);
    }
    gamescope_swapchain_factory_v2_destroy(factory);
    wl_registry_destroy(registry);
    wl_event_queue_destroy(globalsQueue);
    wl_display_disconnect(display);
    close(sockets[1]);
  }

  VkResult run() { return GamescopeWSILayer::VkDeviceOverrides::QueuePresentKHR(queue, VK_NULL_HANDLE, &info); }
};

}

TEST_CASE("Maintenance presents preserve per-swapchain counts and caller chains", "[present_timing_layer][present_batch]") {
  PresentBatch batch;
  for (auto *state : batch.states)
    state->hasSwapchainMaintenance1 = true;
  CHECK(batch.run() == VK_SUCCESS);
  CHECK(presented.size() == 2);
  CHECK(batch.info.swapchainCount == 2);
  CHECK(batch.info.pNext == &batch.fenceInfo);
  CHECK(batch.fenceInfo.pNext == &batch.idInfo);
  CHECK(batch.idInfo.pNext == &batch.timingInfo);
}

TEST_CASE("Mixed presents preserve the driver's timing request", "[present_timing_layer][present_batch]") {
  PresentBatch batch(true);
  CHECK(batch.run() == VK_SUCCESS);
  CHECK(presented.size() == 2);
  CHECK(presented[0].target == 0);
  CHECK(presented[1].target == batch.timings[1].targetTime);
  CHECK(presented[1].stages == batch.timings[1].presentStageQueries);
  CHECK(presented[1].id == 92);
  CHECK(presented[1].image == 1);
  CHECK(presented[1].fence == batch.fences[1]);
  CHECK(refused.waits == std::vector<VkSemaphore>{batch.semaphore});
  CHECK(batch.info.pNext == &batch.fenceInfo);
  CHECK(batch.fenceInfo.pNext == &batch.idInfo);
  CHECK(batch.idInfo.pNext == &batch.timingInfo);
}

TEST_CASE("A full timing queue does not reject the other hooked swapchain", "[present_timing_layer][present_batch]") {
  PresentBatch batch;
  batch.states[0]->timingQueue.setSize(1);
  REQUIRE(batch.states[0]->timingQueue.reserve(1, 0, VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT));
  batch.states[1]->timingQueue.setSize(1);
  CHECK(batch.run() == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT);
  // The refused swapchain still holds one entry, the presented one took its slot.
  CHECK(batch.states[0]->timingQueue.setSize(1) == VK_SUCCESS);
  CHECK(!batch.states[1]->timingQueue.hasSlot());
  CHECK(batch.results[0] == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT);
  CHECK(batch.results[1] == VK_SUCCESS);
  CHECK(refused.released.empty());
  CHECK(presented.size() == 1);
  if (!presented.empty()) {
    CHECK(presented[0].swapchain == batch.swapchains[1]);
    CHECK(presented[0].id == 92);
    CHECK(presented[0].fence == batch.fences[1]);
  }
  CHECK(refused.fences == std::vector<VkFence>{batch.fences[0]});
  CHECK(refused.waits == std::vector<VkSemaphore>{batch.semaphore});
}

TEST_CASE("An untimed swapchain survives another swapchain's full queue", "[present_timing_layer][present_batch]") {
  PresentBatch batch;
  batch.states[1]->timingQueue.setSize(0);
  batch.timings[0].presentStageQueries = 0;
  CHECK(batch.run() == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT);
  CHECK(batch.results[0] == VK_SUCCESS);
  CHECK(batch.results[1] == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT);
  CHECK(presented.size() == 1);
  if (!presented.empty())
    CHECK(presented[0].swapchain == batch.swapchains[0]);
  CHECK(refused.released.empty());
}

TEST_CASE("A target on a swapchain without bypass presents untimed", "[present_timing_layer][present_batch]") {
  PresentBatch batch;
  batch.states[1]->isBypassingXWayland = false;
  GamescopeWSILayer::gamescopeSurfaces.find(batch.surfaces[1])->flags = GamescopeLayerClient::Flag::NoSuboptimal;
  CHECK(batch.run() == VK_SUCCESS);
  CHECK(batch.results[0] == VK_SUCCESS);
  CHECK(batch.results[1] == VK_SUCCESS);
  CHECK(presented.size() == 2);
  CHECK(refused.released.empty());
  TimingRecord record;
  CHECK(record.query(*batch.states[1]) == VK_SUCCESS);
  CHECK(record.properties.presentationTimingCount == 1);
  CHECK(record.timing.presentStageCount == 1);
  CHECK(record.stage.time == 0);
}

TEST_CASE("A lost gamescope connection only refuses swapchains that need it", "[present_timing_layer][present_batch]") {
  PresentBatch batch;
  batch.states[0]->isBypassingXWayland = false;
  GamescopeWSILayer::gamescopeSurfaces.find(batch.surfaces[0])->flags = GamescopeLayerClient::Flag::NoSuboptimal;
  close(batch.sockets[1]);
  batch.sockets[1] = -1;
  CHECK(batch.run() == VK_ERROR_OUT_OF_DATE_KHR);
  CHECK(batch.results[0] == VK_SUCCESS);
  CHECK(batch.results[1] == VK_ERROR_OUT_OF_DATE_KHR);
  CHECK(presented.size() == 1);
  if (!presented.empty())
    CHECK(presented[0].swapchain == batch.swapchains[0]);
  CHECK(wl_display_get_error(batch.display) != 0);
}

TEST_CASE("Split presents preserve allocation failure before any submission", "[present_timing_layer][present_batch]") {
  PresentBatch batch(true);
  refused.failSubmit = 1;
  batch.results.fill(VK_NOT_READY);
  CHECK(batch.run() == VK_ERROR_OUT_OF_HOST_MEMORY);
  CHECK(refused.waits.empty());
  CHECK(refused.fences.empty());
  CHECK(refused.released.empty());
  CHECK(presented.empty());
  CHECK(batch.results == std::array<VkResult, 2>{VK_NOT_READY, VK_NOT_READY});
}

TEST_CASE("Split presents convert allocation failure after waits to device loss", "[present_timing_layer][present_batch]") {
  PresentBatch batch(true);
  refused.idleResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
  CHECK(batch.run() == VK_ERROR_DEVICE_LOST);
  CHECK(refused.waits == std::vector<VkSemaphore>{batch.semaphore});
  CHECK(refused.fences.empty());
  CHECK(refused.released.empty());
  CHECK(presented.empty());
}

TEST_CASE("A native present failing after a hooked present cannot return OOM", "[present_timing_layer][present_batch]") {
  PresentBatch batch(true);
  batch.info.waitSemaphoreCount = 0;
  batch.info.pWaitSemaphores = nullptr;
  failPresent = 2;
  CHECK(batch.run() == VK_ERROR_DEVICE_LOST);
  CHECK(presented.size() == 1);
  CHECK(batch.results[0] == VK_SUCCESS);
  CHECK(batch.results[1] == VK_ERROR_DEVICE_LOST);
}

TEST_CASE("Healthy hooked batches keep the single driver present", "[present_timing_layer][present_batch]") {
  PresentBatch batch;
  CHECK(batch.run() == VK_SUCCESS);
  CHECK(presentCalls == 1);
  CHECK(refused.submits == 0);
  CHECK(refused.idles == 0);
  CHECK(presented.size() == 2);
}

TEST_CASE("A failed extra fence submit cannot return OOM after the first succeeds", "[present_timing_layer][present_batch]") {
  PresentBatch batch;
  refused.failSubmit = 2;
  CHECK(GamescopeWSILayer::VkDeviceOverrides::PresentWithoutImage(batch.queue, VK_NULL_HANDLE, &batch.info,
    VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT) == VK_ERROR_DEVICE_LOST);
  CHECK(refused.submits == 2);
  CHECK(refused.waits == std::vector<VkSemaphore>{batch.semaphore});
  CHECK(refused.fences == std::vector<VkFence>{batch.fences[0]});
  CHECK(refused.released.empty());
}

TEST_CASE("Present slicing preserves all per-swapchain arrays and scalar metadata", "[present_timing_layer][present_batch]") {
  PresentBatch batch(true);
  std::array<VkPresentRegionKHR, 2> regions{{{0, nullptr}, {1, nullptr}}};
  VkRectLayerKHR rect{{1, 2}, {3, 4}, 0};
  regions[1].pRectangles = &rect;
  std::array<uint32_t, 2> masks{1, 2};
  std::array<VkPresentTimeGOOGLE, 2> times{{{11, 100}, {22, 200}}};
  std::array<VkPresentModeKHR, 2> modes{VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR};
  VkDisplayPresentInfoKHR display{.sType = VK_STRUCTURE_TYPE_DISPLAY_PRESENT_INFO_KHR, .pNext = batch.info.pNext,
    .srcRect = {{1, 2}, {3, 4}}, .dstRect = {{5, 6}, {7, 8}}, .persistent = VK_TRUE};
  VkFrameBoundaryEXT boundary{.sType = VK_STRUCTURE_TYPE_FRAME_BOUNDARY_EXT, .pNext = &display,
    .flags = VK_FRAME_BOUNDARY_FRAME_END_BIT_EXT, .frameID = 99};
  VkPresentRegionsKHR regionInfo{VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR, &boundary, 2, regions.data()};
  VkDeviceGroupPresentInfoKHR group{VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_INFO_KHR, &regionInfo, 2, masks.data(), VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR};
  VkPresentTimesInfoGOOGLE google{VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE, &group, 2, times.data()};
  VkPresentIdKHR ids{VK_STRUCTURE_TYPE_PRESENT_ID_KHR, &google, 2, batch.ids.data()};
  VkSwapchainPresentModeInfoEXT modeInfo{VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_EXT, &ids, 2, modes.data()};
  batch.info.pNext = &modeInfo;
  for (uint32_t i = 0; i < 2; i++) {
    VkResult result;
    GamescopeWSILayer::PresentInfoSlice slice(batch.info, i, &result);
    CHECK(slice.info.swapchainCount == 1);
    CHECK(slice.info.pSwapchains[0] == batch.swapchains[i]);
    CHECK(slice.info.pImageIndices[0] == batch.images[i]);
    CHECK(slice.info.pResults == &result);
    CHECK(vkroots::FindInChain<VkPresentRegionsKHR>(&slice.info)->pRegions == &regions[i]);
    CHECK(vkroots::FindInChain<VkDeviceGroupPresentInfoKHR>(&slice.info)->pDeviceMasks == &masks[i]);
    CHECK(vkroots::FindInChain<VkPresentTimesInfoGOOGLE>(&slice.info)->pTimes == &times[i]);
    CHECK(vkroots::FindInChain<VkPresentIdKHR>(&slice.info)->pPresentIds == &batch.ids[i]);
    CHECK(vkroots::FindInChain<VkPresentId2KHR>(&slice.info)->pPresentIds == &batch.ids[i]);
    CHECK(vkroots::FindInChain<VkSwapchainPresentModeInfoEXT>(&slice.info)->pPresentModes == &modes[i]);
    CHECK(vkroots::FindInChain<VkSwapchainPresentFenceInfoEXT>(&slice.info)->pFences == &batch.fences[i]);
    CHECK(vkroots::FindInChain<VkPresentTimingsInfoEXT>(&slice.info)->pTimingInfos == &batch.timings[i]);
    CHECK(vkroots::FindInChain<VkDisplayPresentInfoKHR>(&slice.info)->dstRect.extent.width == 7);
    CHECK(vkroots::FindInChain<VkFrameBoundaryEXT>(&slice.info)->frameID == 99);
    CHECK(vkroots::FindInChain<VkFrameBoundaryEXT>(&slice.info)->flags == (i ? VK_FRAME_BOUNDARY_FRAME_END_BIT_EXT : 0));
  }
  CHECK(batch.info.pNext == &modeInfo);
  CHECK(modeInfo.pNext == &ids);
  CHECK(ids.pNext == &google);
  CHECK(google.pNext == &group);
  CHECK(group.pNext == &regionInfo);
  CHECK(boundary.flags == VK_FRAME_BOUNDARY_FRAME_END_BIT_EXT);
  CHECK(batch.fenceInfo.swapchainCount == 2);

  group.swapchainCount = 0;
  group.pDeviceMasks = nullptr;
  batch.timingInfo.pTimingInfos = nullptr;
  GamescopeWSILayer::PresentInfoSlice slice(batch.info, 1, nullptr);
  CHECK(vkroots::FindInChain<VkDeviceGroupPresentInfoKHR>(&slice.info)->swapchainCount == 0);
  CHECK(vkroots::FindInChain<VkDeviceGroupPresentInfoKHR>(&slice.info)->pDeviceMasks == nullptr);
  CHECK(vkroots::FindInChain<VkPresentTimingsInfoEXT>(&slice.info)->pTimingInfos == nullptr);
}

TEST_CASE("Mixed batches without timing entries need no split", "[present_timing_layer][present_batch]") {
  PresentBatch batch(true);
  batch.timingInfo.pTimingInfos = nullptr;
  CHECK(batch.run() == VK_SUCCESS);
  CHECK(presentCalls == 1);
  CHECK(refused.submits == 0);
  CHECK(refused.idles == 0);
  CHECK(presented.size() == 2);
}

TEST_CASE("A retired swapchain does not reject the other hooked swapchain", "[present_timing_layer][present_batch]") {
  PresentBatch batch;
  *batch.states[0]->retired = true;
  CHECK(batch.run() == VK_ERROR_OUT_OF_DATE_KHR);
  CHECK(batch.results[0] == VK_ERROR_OUT_OF_DATE_KHR);
  CHECK(batch.results[1] == VK_SUCCESS);
  CHECK(presented.size() == 1);
  if (!presented.empty())
    CHECK(presented[0].swapchain == batch.swapchains[1]);
  CHECK(refused.waits == std::vector<VkSemaphore>{batch.semaphore});
  CHECK(refused.fences == std::vector<VkFence>{batch.fences[0]});
}

TEST_CASE("An unknown chain entry keeps the whole-batch refusal", "[present_timing_layer][present_batch]") {
  PresentBatch batch;
  batch.states[0]->timingQueue.setSize(0);
  VkBaseInStructure unknown{static_cast<VkStructureType>(0x7fffffff), static_cast<const VkBaseInStructure *>(batch.info.pNext)};
  batch.info.pNext = &unknown;
  CHECK(batch.run() == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT);
  CHECK(refused.released.empty());
  CHECK(presented.empty());
}

TEST_CASE("A rejected last swapchain still ends the frame", "[present_timing_layer][present_batch]") {
  PresentBatch batch;
  batch.states[1]->timingQueue.setSize(0);
  VkFrameBoundaryEXT boundary{.sType = VK_STRUCTURE_TYPE_FRAME_BOUNDARY_EXT, .pNext = batch.info.pNext,
    .flags = VK_FRAME_BOUNDARY_FRAME_END_BIT_EXT, .frameID = 23};
  batch.info.pNext = &boundary;
  SECTION("with waits and fences") {}
  SECTION("without waits or fences") {
    batch.info.waitSemaphoreCount = 0;
    boundary.pNext = &batch.idInfo;
  }
  CHECK(batch.run() == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT);
  CHECK(refused.boundaries == std::vector<VkFrameBoundaryFlagsEXT>{0, VK_FRAME_BOUNDARY_FRAME_END_BIT_EXT});
}

TEST_CASE("A failing middle present marks the remaining swapchains lost", "[present_timing_layer][present_batch]") {
  PresentBatch batch(true);
  std::array<VkSwapchainKHR, 3> swapchains{batch.swapchains[0], batch.swapchains[1], fakeHandle<VkSwapchainKHR>(503)};
  std::array<uint32_t, 3> images{2, 1, 0};
  std::array<uint64_t, 3> ids{71, 92, 93};
  std::array<VkPresentTimingInfoEXT, 3> timings{{batch.timings[0], batch.timings[1], batch.timings[1]}};
  std::array<VkResult, 3> results{};
  VkPresentTimingsInfoEXT timingInfo{VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT, nullptr, 3, timings.data()};
  VkPresentId2KHR idInfo{VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR, &timingInfo, 3, ids.data()};
  VkFrameBoundaryEXT boundary{.sType = VK_STRUCTURE_TYPE_FRAME_BOUNDARY_EXT, .pNext = &idInfo,
    .flags = VK_FRAME_BOUNDARY_FRAME_END_BIT_EXT, .frameID = 7};
  VkPresentInfoKHR info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, &boundary, 1, &batch.semaphore, 3, swapchains.data(), images.data(), results.data()};
  failPresent = 2;
  CHECK(GamescopeWSILayer::VkDeviceOverrides::QueuePresentKHR(batch.queue, VK_NULL_HANDLE, &info) == VK_ERROR_DEVICE_LOST);
  CHECK(presented.size() == 1);
  CHECK(results == std::array<VkResult, 3>{VK_SUCCESS, VK_ERROR_DEVICE_LOST, VK_ERROR_DEVICE_LOST});
  CHECK(refused.boundaries == std::vector<VkFrameBoundaryFlagsEXT>{0});
}

#include "wayland_test_server.hpp"

TEST_CASE("Surface globals do not dispatch another surface's limiter", "[present_timing_layer][surface_globals]") {
  WaylandTestServer peer;
  auto first = GamescopeWSILayer::GamescopeWaylandObjects::get(peer.display);
  std::optional second{GamescopeWSILayer::GamescopeWaylandObjects::get(peer.display)};
  REQUIRE(first.valid());
  REQUIRE(second->valid());
  auto *timingQueue = wl_display_create_queue(peer.display);
  peer.SendState(1);
  // Read all events without dispatching either surface's globals.
  auto *readQueue = wl_display_create_queue(peer.display);
  REQUIRE(wl_display_roundtrip_queue(peer.display, readQueue) >= 0);
  REQUIRE(GamescopeWSILayer::waylandPumpEvents(peer.display, timingQueue, first.queue.get()) >= 0);
  CHECK(first.limiterState->state == 1);
  CHECK(second->limiterState->state == 0);
  second.reset();
  peer.SendState(0);
  REQUIRE(wl_display_roundtrip_queue(peer.display, readQueue) >= 0);
  REQUIRE(GamescopeWSILayer::waylandPumpEvents(peer.display, timingQueue, first.queue.get()) >= 0);
  CHECK(first.limiterState->state == 0);
  wl_event_queue_destroy(timingQueue);
  wl_event_queue_destroy(readQueue);
}

namespace {
VkResult surfaceResult = VK_ERROR_OUT_OF_HOST_MEMORY;
uint64_t surfaceSerial = 10000;
std::vector<VkSurfaceKHR> destroyedSurfaces;
VkResult VKAPI_CALL createTestSurface(VkInstance, const VkWaylandSurfaceCreateInfoKHR *,
                                    const VkAllocationCallbacks *, VkSurfaceKHR *surface) {
  if (surfaceResult == VK_SUCCESS)
    *surface = fakeHandle<VkSurfaceKHR>(++surfaceSerial);
  return surfaceResult;
}
bool destroyedWhileTracked = false;
void VKAPI_CALL destroyTestSurface(VkInstance, VkSurfaceKHR surface, const VkAllocationCallbacks *) {
  if (surface)
    destroyedSurfaces.push_back(surface);
  destroyedWhileTracked |= GamescopeWSILayer::gamescopeSurfaces.find(surface) != nullptr;
}
PFN_vkVoidFunction getSurfaceProc(VkInstance instance, const char *name) {
  if (std::strcmp(name, "vkCreateWaylandSurfaceKHR") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(createTestSurface);
  if (std::strcmp(name, "vkDestroySurfaceKHR") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(destroyTestSurface);
  return getInstanceProc(instance, name);
}
}

TEST_CASE("Failed native surface creation releases bound globals", "[present_timing_layer][surface_globals]") {
  WaylandTestServer peer;
  auto instance = reinterpret_cast<VkInstance>(uintptr_t(1001));
  REQUIRE(GamescopeWSILayer::gamescopeInstances.create(instance, GamescopeWSILayer::GamescopeInstanceData{
    .display = peer.display}) != nullptr);
  vkroots::VkInstanceDispatch dispatch(getSurfaceProc, instance, nullptr);
  auto *appSurface = peer.CreateSurface();
  VkWaylandSurfaceCreateInfoKHR info{.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
    .display = peer.display, .surface = appSurface};
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  surfaceResult = VK_ERROR_OUT_OF_HOST_MEMORY;
  destroyedSurfaces.clear();
  CHECK(GamescopeWSILayer::VkInstanceOverrides::CreateWaylandSurfaceKHR(dispatch, instance, &info, nullptr, &surface) == VK_ERROR_OUT_OF_HOST_MEMORY);
  peer.Sync();
  CHECK(peer.FactoryCount() == 0);
  CHECK(peer.LimiterCount() == 0);
  CHECK(peer.SurfaceCount() == 1);
  CHECK(destroyedSurfaces.empty());

  surfaceResult = VK_SUCCESS;
  REQUIRE(GamescopeWSILayer::VkInstanceOverrides::CreateWaylandSurfaceKHR(dispatch, instance, &info, nullptr, &surface) == VK_SUCCESS);
  REQUIRE(GamescopeWSILayer::gamescopeSurfaces.find(surface));
  destroyedWhileTracked = false;
  GamescopeWSILayer::VkInstanceOverrides::DestroySurfaceKHR(dispatch, instance, surface, nullptr);
  peer.Sync();
  CHECK(peer.FactoryCount() == 0);
  CHECK(destroyedSurfaces == std::vector<VkSurfaceKHR>{surface});
  // A surface created concurrently may reuse the handle once the driver frees it.
  CHECK_FALSE(destroyedWhileTracked);
  // The layer must leave the application's native surface alive.
  CHECK(peer.SurfaceCount() == 1);
  wl_surface_destroy(appSurface);
  peer.Sync();
  CHECK(peer.SurfaceCount() == 0);
  GamescopeWSILayer::gamescopeInstances.erase(instance);
}

TEST_CASE("Missing compositor globals unwind their other bindings", "[present_timing_layer][surface_globals]") {
  WaylandTestServer peer(false);
  {
    auto objects = GamescopeWSILayer::GamescopeWaylandObjects::get(peer.display);
    CHECK_FALSE(objects.valid());
  }
  peer.Sync();
  CHECK(peer.FactoryCount() == 0);
  CHECK(peer.LimiterCount() == 0);
}

namespace {
void VKAPI_CALL destroyTestSwapchain(VkDevice, VkSwapchainKHR, const VkAllocationCallbacks *) {}
PFN_vkVoidFunction getSurfaceDeviceProc(VkDevice device, const char *name) {
  if (std::strcmp(name, "vkDestroySwapchainKHR") == 0)
    return reinterpret_cast<PFN_vkVoidFunction>(destroyTestSwapchain);
  return getDeviceProc(device, name);
}
}

TEST_CASE("Retiring an old swapchain preserves its surface globals and successor", "[present_timing_layer][surface_globals]") {
  using namespace GamescopeWSILayer;
  WaylandTestServer peer;
  auto owner = GamescopeWaylandObjects::get(peer.display);
  REQUIRE(owner.valid());
  auto *appSurface = peer.CreateSurface();
  VkDeviceCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  vkroots::VkDeviceDispatch dispatch(getSurfaceDeviceProc, VK_NULL_HANDLE, VK_NULL_HANDLE, nullptr, &createInfo);
  const std::array<VkSwapchainKHR, 2> handles{
    fakeHandle<VkSwapchainKHR>(2001), fakeHandle<VkSwapchainKHR>(2002)};
  std::array<GamescopeSwapchainData *, 2> states;
  for (size_t i = 0; i < states.size(); ++i) {
    auto *queue = wl_display_create_queue(peer.display);
    auto *wrapper = static_cast<gamescope_swapchain_factory_v2 *>(wl_proxy_create_wrapper(owner.gamescopeSwapchainFactory.get()));
    wl_proxy_set_queue(reinterpret_cast<wl_proxy *>(wrapper), queue);
    auto *object = gamescope_swapchain_factory_v2_create_swapchain(wrapper, appSurface);
    wl_proxy_wrapper_destroy(wrapper);
    states[i] = gamescopeSwapchains.create(handles[i], GamescopeSwapchainData{
      .object = object, .display = peer.display, .globalsQueue = owner.queue.get(), .eventQueue = queue});
    REQUIRE(states[i]);
    gamescope_swapchain_add_listener(object, &s_swapchainListener, states[i]);
  }
  peer.Sync();
  peer.SendState(1);
  peer.SendRefreshCycle(8'333'333);
  peer.Sync();
  REQUIRE(waylandPumpEvents(peer.display, states[1]->eventQueue, states[1]->globalsQueue) >= 0);
  CHECK(owner.limiterState->state == 1);
  CHECK(states[1]->refreshCycle == 8'333'333);
  CHECK(states[0]->refreshCycle == 16'666'666);

  VkDeviceOverrides::DestroySwapchainKHR(dispatch, VK_NULL_HANDLE, handles[0], nullptr);
  peer.Sync();
  peer.SendState(0);
  peer.SendRefreshCycle(16'666'667);
  peer.Sync();
  REQUIRE(waylandPumpEvents(peer.display, states[1]->eventQueue, states[1]->globalsQueue) >= 0);
  CHECK(owner.limiterState->state == 0);
  CHECK(states[1]->refreshCycle == 16'666'667);
  VkDeviceOverrides::DestroySwapchainKHR(dispatch, VK_NULL_HANDLE, handles[1], nullptr);
  wl_surface_destroy(appSurface);
}

TEST_CASE("A timed present round-trips through the gamescope protocol", "[present_timing_layer][surface_globals]") {
  using namespace GamescopeWSILayer;
  WaylandTestServer peer;
  auto owner = GamescopeWaylandObjects::get(peer.display);
  REQUIRE(owner.valid());
  auto *appSurface = peer.CreateSurface();
  const auto surface = fakeHandle<VkSurfaceKHR>(2101);
  const auto handle = fakeHandle<VkSwapchainKHR>(2102);
  REQUIRE(gamescopeSurfaces.create(surface, GamescopeSurfaceData{}));
  auto *eventQueue = wl_display_create_queue(peer.display);
  auto *wrapper = static_cast<gamescope_swapchain_factory_v2 *>(wl_proxy_create_wrapper(owner.gamescopeSwapchainFactory.get()));
  wl_proxy_set_queue(reinterpret_cast<wl_proxy *>(wrapper), eventQueue);
  auto *object = gamescope_swapchain_factory_v2_create_swapchain_with_timing(wrapper, appSurface);
  wl_proxy_wrapper_destroy(wrapper);
  auto *state = gamescopeSwapchains.create(handle, GamescopeSwapchainData{
    .object = object, .display = peer.display, .globalsQueue = owner.queue.get(), .eventQueue = eventQueue,
    .surface = surface, .isBypassingXWayland = true, .presentMode = VK_PRESENT_MODE_FIFO_KHR, .presentTimingEnabled = true});
  REQUIRE(state);
  gamescope_swapchain_add_listener(object, &s_swapchainListener, state);
  state->timingQueue.setSize(4);

  presented.clear();
  presentCalls = failPresent = 0;
  const uint64_t target = 0x1'0000'0005;
  const VkPresentTimingInfoFlagsEXT flags =
    VK_PRESENT_TIMING_INFO_PRESENT_AT_RELATIVE_TIME_BIT_EXT | VK_PRESENT_TIMING_INFO_PRESENT_AT_NEAREST_REFRESH_CYCLE_BIT_EXT;
  VkPresentTimingInfoEXT timing{.sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT, .flags = flags, .targetTime = target,
    .presentStageQueries = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT};
  VkPresentTimingsInfoEXT timings{VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT, nullptr, 1, &timing};
  uint64_t id = 42;
  VkPresentId2KHR ids{VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR, &timings, 1, &id};
  uint32_t image = 0;
  VkResult result = VK_NOT_READY;
  VkPresentInfoKHR info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, &ids, 0, nullptr, 1, &handle, &image, &result};
  VkDeviceCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  vkroots::VkDeviceDispatch device(getSurfaceDeviceProc, VK_NULL_HANDLE, VK_NULL_HANDLE, nullptr, &createInfo);
  vkroots::VkQueueDispatch queue(VK_NULL_HANDLE, &device);
  REQUIRE(VkDeviceOverrides::QueuePresentKHR(queue, VK_NULL_HANDLE, &info) == VK_SUCCESS);
  CHECK(presented.size() == 1);
  peer.Sync();

  const auto requests = peer.TimingRequests();
  REQUIRE(requests.size() == 1);
  CHECK(requests[0].target == target);
  CHECK(requests[0].flags == flags);

  peer.SendTimingProperties(8'333'333, 8'333'333);
  peer.SendPresentTiming(requests[0].serial, 0x2'0000'0010, 0x2'0000'0020, 0x2'0000'0030);
  peer.Sync();

  VkSwapchainTimingPropertiesEXT properties{.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_TIMING_PROPERTIES_EXT};
  uint64_t counter = 0;
  CHECK(VkDeviceOverrides::GetSwapchainTimingPropertiesEXT(device, VK_NULL_HANDLE, handle, &properties, &counter) == VK_SUCCESS);
  CHECK(properties.refreshDuration == 8'333'333);
  CHECK(counter != 0);

  VkPresentStageTimeEXT stage{};
  VkPastPresentationTimingEXT past{.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT, .presentStageCount = 1, .pPresentStages = &stage};
  VkPastPresentationTimingInfoEXT pastInfo{.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT, .swapchain = handle};
  VkPastPresentationTimingPropertiesEXT pastProperties{.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT,
    .presentationTimingCount = 1, .pPresentationTimings = &past};
  CHECK(VkDeviceOverrides::GetPastPresentationTimingEXT(device, VK_NULL_HANDLE, &pastInfo, &pastProperties) == VK_SUCCESS);
  CHECK(pastProperties.presentationTimingCount == 1);
  CHECK(past.presentId == id);
  CHECK(past.reportComplete);
  CHECK(stage.stage == VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT);
  CHECK(stage.time == 0x2'0000'0030);

  VkDeviceOverrides::DestroySwapchainKHR(device, VK_NULL_HANDLE, handle, nullptr);
  gamescopeSurfaces.erase(surface);
  wl_surface_destroy(appSurface);
}
