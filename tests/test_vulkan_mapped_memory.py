#!/usr/bin/env python3
"""Exercise production image allocation/addressing with CPU-only Vulkan fixtures."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def between(source, start, end):
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/rendervulkan.cpp").read_text()
    header = (root / "src/rendervulkan.hpp").read_text()
    compositor = (root / "src/steamcompmgr.cpp").read_text()
    init = between(source, "bool CVulkanTexture::BInit(",
                   "bool CVulkanTexture::BInitFromSwapchain(")
    properties = between(init, "\tif ( flags.bMappable == true )",
                         "\tif ( flags.bOutputImage == true )")
    allocation_info = between(init, "\tVkMemoryAllocateInfo allocInfo = {",
                              "\n\tm_size = allocInfo.allocationSize;")
    # Compile the ordinary-allocation branch itself, including its early return,
    # followed by the actual allocation call. The imported/reused paths are not
    # substituted with a second implementation of the allocation policy.
    selection = between(init, "\t\t\t// HOST_CACHED improves CPU readback,",
                        "\n\t\t}\n\t\t\n\t\tres = g_device.vk.AllocateMemory")
    allocate = next(line for line in init.splitlines()
                    if "res = g_device.vk.AllocateMemory(" in line)
    create = between(init, "\tres = g_device.vk.CreateImage(",
                     "\n\tVkMemoryRequirements memRequirements;")
    allocate_tail = between(init, allocate,
                            "\n\t}\n\telse\n\t{\n\t\tvk_log.infof")
    allocate_tail = allocate_tail[len(allocate):]
    reuse = between(init, '\t\tvk_log.infof("%d vs %d!",',
                    "\n\t}\n\t\n\tres = g_device.vk.BindImageMemory")
    destructor = between(source, "CVulkanTexture::~CVulkanTexture( void )",
                         "\nint CVulkanTexture::memoryFence()")
    swapchain = between(source, "bool CVulkanTexture::BInitFromSwapchain(",
                        "\ngamescope::Rc<CVulkanTexture> CVulkanTexture::AcquireDeviceLocalStagingImage()")
    find_type = between(source, "int32_t CVulkanDevice::findMemoryType(",
                        "\nstd::unique_ptr<CVulkanCmdBuffer> CVulkanDevice::commandBuffer(")
    texture_header = between(header, "class CVulkanTexture ", "\n};")
    members = [line for line in texture_header.splitlines()
               if line.startswith(("\tinline uint8_t *mappedData()",
                                   "\tinline uint8_t *mappedPixels()",
                                   "\tinline uint32_t rowPitch()",
                                   "\tinline VkDeviceSize totalSize()",
                                   "\tuint32_t m_unRowPitch =",
                                   "\tVkDeviceSize m_mappedOffset =",
                                   "\tVkDeviceSize m_size =",
                                   "\tuint32_t m_lumaOffset =",
                                   "\tuint32_t m_lumaPitch =",
                                   "\tuint32_t m_chromaOffset =",
                                   "\tuint32_t m_chromaPitch =",
                                   "\tbool m_bInitialized =",
                                   "\tbool m_bOwnsImage =",
                                   "\tbool m_bDeviceLocal =",
                                   "\tbool m_bOutputImage =",
                                   "\tVkImage m_vkImage =",
                                   "\tVkDeviceMemory m_vkImageMemory =",
                                   "\tVkImageView m_",
                                   "\tuint32_t m_drmFormat =",
                                   "\tuint32_t m_width =",
                                   "\tuint32_t m_height =",
                                   "\tuint32_t m_depth =",
                                   "\tuint32_t m_contentWidth =",
                                   "\tuint32_t m_contentHeight =",
                                   "\tVkFormat m_format =",
                                   "\tuint8_t *m_pMappedData ="))]
    layout = between(init, "\tif ( flags.bMappable == true )\n\t{\n\t\tassert( tiling",
                     "\n\tif ( flags.bExportable == true )")
    # Capture enables export independently of CPU mapping. Execute its actual
    # nonmodifier query and assignments instead of reproducing their policy.
    export_layout = between(init, "\n\t\t\tconst VkImageSubresource subresource = {\n",
                            "\n\t\t}\n\n\t\tm_dmabuf = dmabuf;")
    mapping = between(init, "\tif ( flags.bMappable )\n",
                      "\n\tm_bInitialized = true;")
    upload = between(source, "static void framegen_net_stage_weights(",
                     "\nstatic void framegen_record_net(")
    # The completed-readback prefix owns addressing and row copies. Later
    # profile validation/persistence is outside this mapped-memory regression.
    readback = between(source, "static void framegen_net_profile_consume()\n{",
                       "\n\tfor ( uint32_t i = 0; i < k_uFramegenNetFloats; i++ )") + "\n}\n"
    screenshot_data = between(compositor, "\t\t\t\tconst uint8_t *mappedData = pScreenshotTexture->",
                              "\n\t\t\t\tconst uint32_t width =")
    screenshot_write = next(line for line in compositor.splitlines()
                            if "fwrite(" in line and "pScreenshotTexture->totalSize()" in line)
    with tempfile.TemporaryDirectory(prefix="gamescope-vk-memory-") as directory:
        directory = Path(directory)
        (directory / "memory_type.inc").write_text(find_type)
        (directory / "image_allocation.inc").write_text(
            properties + allocation_info + selection + "\n" + allocate + "\n")
        (directory / "image_create.inc").write_text(create)
        (directory / "image_allocation_tail.inc").write_text(allocate_tail)
        (directory / "image_reuse.inc").write_text(reuse)
        (directory / "image_destructor.inc").write_text(destructor)
        (directory / "image_swapchain.inc").write_text(swapchain)
        (directory / "mapped_members.inc").write_text("\n".join(members) + "\n")
        (directory / "mapped_layout.inc").write_text(layout)
        (directory / "export_layout.inc").write_text(export_layout)
        (directory / "mapped_mapping.inc").write_text(mapping)
        (directory / "mapped_upload.inc").write_text(upload)
        (directory / "mapped_readback.inc").write_text(readback)
        (directory / "screenshot_data.inc").write_text(screenshot_data + "\n")
        (directory / "screenshot_write.inc").write_text(screenshot_write + "\n")
        binary = directory / "test"
        subprocess.run(shlex.split(os.environ.get("CXX", "c++"))
                       + ["-std=c++20", "-Wall", "-Wextra", "-Werror",
                          "-Wno-missing-field-initializers", "-I", str(directory),
                          str(root / "tests/test_vulkan_mapped_memory.cpp"),
                          "-o", str(binary)], check=True, timeout=60)
        subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == "__main__":
    main()
