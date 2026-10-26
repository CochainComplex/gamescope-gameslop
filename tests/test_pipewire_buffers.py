#!/usr/bin/env python3
"""Exercise production PipeWire DMA-BUF descriptors with real SPA ABI types."""
import hashlib
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import tempfile


FIXTURE = r'''
#include <spa/buffer/buffer.h>
#include <spa/param/video/raw.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <limits>
#include <sys/types.h>
#include <vector>

struct Dmabuf {
    int n_planes = 1;
    int fd[1] = {-1};
    uint32_t offset[1] = {};
    uint32_t stride[1] = {};
};
struct CVulkanTexture {
    uint32_t w = 5, h = 3, y = 7, yp = 11, uv = 72, uvp = 9;
    Dmabuf attributes;
    const Dmabuf &dmabuf() const { return attributes; }
    uint32_t width() const { return w; }
    uint32_t height() const { return h; }
    uint32_t lumaOffset() const { return y; }
    uint32_t lumaRowPitch() const { return yp; }
    uint32_t chromaOffset() const { return uv; }
    uint32_t chromaRowPitch() const { return uvp; }
};

#include "pipewire_buffers.inc"

struct Buffer {
    std::array<spa_chunk, 3> chunks{};
    std::array<spa_data, 3> data{};
    spa_buffer buffer{};
    explicit Buffer(uint32_t blocks = 2) {
        buffer.n_datas = blocks;
        buffer.datas = data.data();
        for (size_t i = 0; i < data.size(); i++) {
            data[i].type = 1u << SPA_DATA_DmaBuf;
            data[i].chunk = &chunks[i];
        }
    }
};

static void descriptors(const Buffer &buffer, int fd, uint32_t size) {
    for (uint32_t i = 0; i < buffer.buffer.n_datas; i++) {
        const auto &data = buffer.data[i];
        assert(data.type == SPA_DATA_DmaBuf);
        assert(data.fd == fd && fcntl(fd, F_GETFD) != -1);
        assert(data.flags == SPA_DATA_FLAG_READABLE && data.mapoffset == 0);
        assert(data.maxsize == size && data.data == nullptr);
        assert(data.chunk->flags == 0);
    }
}

static void reconstruct(int fd) {
    CVulkanTexture tex;
    tex.attributes.fd[0] = fd;
    Buffer buffer;
    assert(setup_dmabuf_buffer(&buffer.buffer, &tex, SPA_VIDEO_FORMAT_NV12, 128));
    descriptors(buffer, fd, 128);
    // Seed known physical row locations, independent of descriptor arithmetic.
    // Deliberately use a nonzero Y offset, unequal pitches and an inter-plane gap.
    std::vector<uint8_t> allocation(128, 0xee);
    const std::vector<uint8_t> y = {1,2,3,4,5, 6,7,8,9,10, 11,12,13,14,15};
    const std::vector<uint8_t> uv = {101,102,103,104,105,106, 107,108,109,110,111,112};
    const std::array<size_t, 3> y_rows = {7,18,29};
    const std::array<size_t, 2> uv_rows = {72,81};
    for (size_t row = 0; row < y_rows.size(); row++)
        std::copy_n(y.data() + row * 5, 5, allocation.data() + y_rows[row]);
    for (size_t row = 0; row < uv_rows.size(); row++)
        std::copy_n(uv.data() + row * 6, 6, allocation.data() + uv_rows[row]);

    // A consumer concatenates each block's physical chunk into its logical buffer.
    // Read pixels using the resulting logical plane starts and advertised strides.
    std::vector<uint8_t> logical;
    std::array<size_t, 2> starts{};
    for (size_t plane = 0; plane < 2; plane++) {
        const auto &data = buffer.data[plane];
        assert(data.chunk->offset <= data.maxsize);
        const size_t begin = data.mapoffset + data.chunk->offset;
        const size_t end = begin + data.chunk->size;
        assert(end <= allocation.size());
        starts[plane] = logical.size();
        logical.insert(logical.end(), allocation.begin() + begin, allocation.begin() + end);
    }
    std::vector<uint8_t> reconstructed_y, reconstructed_uv;
    for (size_t row = 0; row < 3; row++) {
        const size_t begin = starts[0] + row * buffer.chunks[0].stride;
        assert(begin + 5 <= logical.size());
        reconstructed_y.insert(reconstructed_y.end(), logical.begin() + begin, logical.begin() + begin + 5);
    }
    for (size_t row = 0; row < 2; row++) {
        const size_t begin = starts[1] + row * buffer.chunks[1].stride;
        assert(begin + 6 <= logical.size());
        reconstructed_uv.insert(reconstructed_uv.end(), logical.begin() + begin, logical.begin() + begin + 6);
    }
    assert(reconstructed_y == y && reconstructed_uv == uv);
}

static void rejections(int fd) {
    CVulkanTexture good;
    good.attributes.fd[0] = fd;
    for (uint32_t blocks : {0u, 1u, 3u}) {
        Buffer buffer(blocks);
        assert(!setup_dmabuf_buffer(&buffer.buffer, &good, SPA_VIDEO_FORMAT_NV12, 128));
    }
    for (size_t plane = 0; plane < 2; plane++) {
        Buffer wrong_type;
        wrong_type.data[plane].type = 1u << SPA_DATA_MemFd;
        assert(!setup_dmabuf_buffer(&wrong_type.buffer, &good, SPA_VIDEO_FORMAT_NV12, 128));
        Buffer no_chunk;
        no_chunk.data[plane].chunk = nullptr;
        assert(!setup_dmabuf_buffer(&no_chunk.buffer, &good, SPA_VIDEO_FORMAT_NV12, 128));
    }
    for (off_t size : {off_t{-1}, off_t{0}, off_t{86}, off_t{UINT32_MAX} + 1}) {
        Buffer buffer;
        assert(!setup_dmabuf_buffer(&buffer.buffer, &good, SPA_VIDEO_FORMAT_NV12, size));
    }
    const auto rejected = [](CVulkanTexture tex) {
        Buffer buffer;
        assert(!setup_dmabuf_buffer(&buffer.buffer, &tex, SPA_VIDEO_FORMAT_NV12, 128));
    };
    auto tex = good; tex.attributes.n_planes = 2; rejected(tex);
    tex = good; tex.attributes.fd[0] = -1; rejected(tex);
    tex = good; tex.w = 0; rejected(tex);
    tex = good; tex.h = 0; rejected(tex);
    tex = good; tex.yp = 0; rejected(tex);
    tex = good; tex.yp = 4; rejected(tex);
    tex = good; tex.uvp = 0; rejected(tex);
    tex = good; tex.uvp = 5; rejected(tex);
    tex = good; tex.y = UINT32_MAX; rejected(tex);
    tex = good; tex.uv = UINT32_MAX; rejected(tex);
    tex = good; tex.h = UINT32_MAX; rejected(tex);
    tex = good; tex.w = UINT32_MAX; rejected(tex);
    tex = good; tex.uv = 20; rejected(tex);
    // A one-row plane can fit while a uint32 pitch overflows SPA's int32 stride.
    tex = good; tex.h = 1; tex.yp = UINT32_MAX; rejected(tex);
}

int main() {
    assert(capture_buffer_blocks(true, SPA_VIDEO_FORMAT_NV12) == 2);
    assert(capture_buffer_blocks(true, SPA_VIDEO_FORMAT_BGRx) == 1);
    assert(capture_buffer_blocks(false, SPA_VIDEO_FORMAT_NV12) == 1);
    assert(capture_buffer_blocks(false, SPA_VIDEO_FORMAT_BGRx) == 1);
    FILE *file = std::tmpfile();
    assert(file);
    const int fd = fileno(file);
    reconstruct(fd);
    CVulkanTexture tex;
    tex.attributes.fd[0] = fd;
    tex.w = 800; tex.h = 600; tex.y = 0; tex.yp = 896; tex.uv = 589824; tex.uvp = 896;
    Buffer hardware;
    assert(setup_dmabuf_buffer(&hardware.buffer, &tex, SPA_VIDEO_FORMAT_NV12, 917504));
    descriptors(hardware, fd, 917504);
    assert(hardware.chunks[0].offset == 0 && hardware.chunks[0].stride == 896);
    assert(hardware.chunks[1].offset == 589824 && hardware.chunks[1].stride == 896);
    assert(hardware.chunks[0].size == 537504 && hardware.chunks[1].size == 268704);
    tex.w = 4; tex.h = 2; tex.attributes.offset[0] = 12; tex.attributes.stride[0] = 20;
    Buffer rgb(1);
    assert(setup_dmabuf_buffer(&rgb.buffer, &tex, SPA_VIDEO_FORMAT_BGRx, 64));
    descriptors(rgb, fd, 64);
    assert(rgb.chunks[0].offset == 12 && rgb.chunks[0].stride == 20 && rgb.chunks[0].size == 40);
    rejections(fd);
    assert(fcntl(fd, F_GETFD) != -1); // Descriptors borrow, rather than close or replace, the texture FD.
    std::fclose(file);
}
'''


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/pipewire.cpp").read_text()
    begin = source.index("static uint32_t capture_buffer_blocks(")
    helpers = source[begin:source.index("\nstatic void destroy_buffer(", begin)]
    print(f"source SHA256: {hashlib.sha256(source.encode()).hexdigest()}", flush=True)
    print(f"helpers SHA256: {hashlib.sha256(helpers.encode()).hexdigest()}", flush=True)
    cflags = shlex.split(subprocess.check_output(
        ["pkg-config", "--cflags", "libpipewire-0.3"], text=True))
    variants = [("production", helpers, True)]
    if sys.argv[1:] == ["--mutants"]:
        single = helpers.replace("? 2 : 1", "? 1 : 1", 1)
        packed = helpers.replace("tex->chromaOffset()", "tex->width() * tex->height()", 1)
        assert single != helpers and packed != helpers
        variants += [("single-block-mutant", single, False), ("packed-uv-mutant", packed, False)]
    with tempfile.TemporaryDirectory(prefix="gamescope-pipewire-buffers-") as directory:
        directory = Path(directory)
        fixture = directory / "test.cpp"
        fixture.write_text(FIXTURE)
        for name, body, expected in variants:
            (directory / "pipewire_buffers.inc").write_text(body)
            binary = directory / name
            subprocess.run(shlex.split(os.environ.get("CXX", "c++"))
                           + ["-std=c++20", "-Wall", "-Wextra", "-Werror", *cflags,
                              "-I", str(directory), str(fixture), "-o", str(binary)],
                           check=True, timeout=60)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10,
                                    preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
            if (result.returncode == 0) != expected:
                raise RuntimeError(f"{name}: unexpected exit {result.returncode}\n{result.stderr}")
            if not expected:
                assert result.returncode == -6 and "Assertion" in result.stderr, result.stderr
            print(f"{name}: {'PASS' if expected else 'caught at runtime'} (exit {result.returncode})")
            if not expected:
                print(result.stderr.strip())


if __name__ == "__main__":
    main()
