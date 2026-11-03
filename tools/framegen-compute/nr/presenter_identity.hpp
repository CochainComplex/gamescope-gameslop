// Offscreen Wine/DXVK host only. No compositor or vendor compute SDK dependency.
// Interop declarations follow DXVK's MIT-licensed src/dxgi/dxgi_interfaces.h.
#pragma once

#include <vulkan_core.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

namespace nr_poc
{
struct InteropAdapter : IUnknown
{
    virtual void STDMETHODCALLTYPE GetVulkanHandles( VkInstance *, VkPhysicalDevice * ) = 0;
};
struct InteropFactory : IUnknown
{
    virtual void STDMETHODCALLTYPE GetVulkanInstance( VkInstance *, PFN_vkGetInstanceProcAddr * ) = 0;
};
inline constexpr GUID AdapterId{ 0x3a6d8f2c, 0xb0e8, 0x4ab4, { 0xb4, 0xdc, 0x4f, 0xd2, 0x48, 0x91, 0xbf, 0xa5 } };
inline constexpr GUID FactoryId{ 0x4c5e1b0d, 0xb0c8, 0x4131, { 0xbf, 0xd8, 0x9b, 0x24, 0x76, 0xf7, 0xf4, 0x08 } };

inline bool matches( IDXGIFactory4 *factory, IDXGIAdapter1 *adapter )
{
    char wanted[33] = {}, expectedPci[32] = {};
    if ( GetEnvironmentVariableA( "GSC_NR_VULKAN_UUID", wanted, sizeof wanted ) != 32
        || !GetEnvironmentVariableA( "GSC_NR_PCI", expectedPci, sizeof expectedPci ) )
        return false;
    std::array<unsigned char, 16> uuid{};
    const auto digit = []( char c ) -> int {
        if ( c >= '0' && c <= '9' ) return c - '0';
        if ( c >= 'a' && c <= 'f' ) return c - 'a' + 10;
        return -1;
    };
    for ( size_t i = 0; i < uuid.size(); ++i )
    {
        const int a = digit( wanted[2*i] ), b = digit( wanted[2*i+1] );
        if ( a < 0 || b < 0 ) return false;
        uuid[i] = unsigned( 16*a + b );
    }
    InteropAdapter *physical = nullptr;
    InteropFactory *instance = nullptr;
    if ( FAILED( adapter->QueryInterface( AdapterId, reinterpret_cast<void **>( &physical ) ) ) ) return false;
    if ( FAILED( factory->QueryInterface( FactoryId, reinterpret_cast<void **>( &instance ) ) ) )
    {
        physical->Release();
        return false;
    }
    VkInstance vkInstance = VK_NULL_HANDLE, factoryInstance = VK_NULL_HANDLE;
    VkPhysicalDevice vkPhysical = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr getProc = nullptr;
    physical->GetVulkanHandles( &vkInstance, &vkPhysical );
    instance->GetVulkanInstance( &factoryInstance, &getProc );
    const auto properties = getProc ? reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        getProc( vkInstance, "vkGetPhysicalDeviceProperties2" ) ) : nullptr;
    const auto extensions = getProc ? reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        getProc( vkInstance, "vkEnumerateDeviceExtensionProperties" ) ) : nullptr;
    uint32_t count = 0;
    bool hasPci = false;
    if ( extensions && vkPhysical && vkInstance == factoryInstance
        && extensions( vkPhysical, nullptr, &count, nullptr ) == VK_SUCCESS )
    {
        std::vector<VkExtensionProperties> list( count );
        if ( extensions( vkPhysical, nullptr, &count, list.data() ) == VK_SUCCESS )
            for ( const auto &extension : list )
                hasPci |= !std::strcmp( extension.extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME );
    }
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT };
    VkPhysicalDeviceIDProperties id{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, &pci };
    VkPhysicalDeviceProperties2 props{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id };
    // Both APIs must identify the exact presenter. No ordinal/name fallback.
    bool matched = false;
    if ( properties && hasPci && vkInstance == factoryInstance && vkPhysical )
    {
        properties( vkPhysical, &props );
        char pciText[32];
        std::snprintf( pciText, sizeof pciText, "%04x:%02x:%02x.%x", pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction );
        matched = !std::memcmp( id.deviceUUID, uuid.data(), uuid.size() ) && !std::strcmp( pciText, expectedPci );
        if ( matched )
            std::printf( "NR_POC_PRESENTER uuid=%s pci=%s vendor=%04x device=%04x\n",
                wanted, pciText, props.properties.vendorID, props.properties.deviceID );
    }
    instance->Release();
    physical->Release();
    return matched;
}
} // namespace nr_poc
