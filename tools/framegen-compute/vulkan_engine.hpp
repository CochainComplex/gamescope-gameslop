#pragma once

#include "framegen/compute_job.hpp"

#include <memory>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace gamescope::framegen::compute
{

struct Device
{
	DeviceUuid uuid{};
	std::string name;
	std::string pci;
	uint32_t vendor = 0;
	uint32_t device = 0;
	bool software = false;
};

struct Request
{
	Pair pair;
	uint32_t width;
	uint32_t height;
	Method method;
	float dx = 0.0f;
	float dy = 0.0f;
	std::span<const Pixel> previous;
	std::span<const Pixel> current;
};

enum class Status { Stale, Pending, Complete, Discarded };

// The current second job is a training parity replay. Reuse this standalone
// context's three float storage buffers and retirement owner; it requires no
// extra image owner or production renderer changes.
struct BufferStep
{
	uint32_t kernel;
	std::span<const std::byte> constants;
	uint32_t groupsX, groupsY = 1, groupsZ = 1;
};

struct BufferReadback
{
	size_t offsetBytes = 0;
	std::span<std::byte> output;
};

// Standalone conformance owner. Host-staged float32 RGBA is intentional here;
// this is not a production image transport or a learned-generation backend.
// A compositor adapter must borrow CVulkanDevice and its existing image/queue
// owner, rather than instantiate this separate device in the hot path.
class VulkanEngine
{
public:
	static std::vector<Device> devices();
	VulkanEngine( DeviceUuid presenter, const std::string &shaderPath, bool validation = false );
	~VulkanEngine();
	VulkanEngine( const VulkanEngine & ) = delete;
	VulkanEngine &operator=( const VulkanEngine & ) = delete;
	[[nodiscard]] const Device &device() const;
	[[nodiscard]] uint64_t epoch() const;
	// Invalid/busy requests return an empty ticket without changing ownership.
	// Vulkan errors throw; a failed queue submission poisons this engine.
	[[nodiscard]] Ticket submit( const Request &request );
	[[nodiscard]] Status poll( Ticket ticket );
	void discard( Ticket ticket );
	// Only a completed ticket can be collected/retired. Discarded data is never
	// copied to output; completion must still be observed before reuse.
	[[nodiscard]] bool collect( Ticket ticket, std::span<Pixel> output );
	[[nodiscard]] bool retire( Ticket ticket );
	[[nodiscard]] bool reset();
	[[nodiscard]] uint32_t addKernel( const std::string &shaderPath );
	[[nodiscard]] Ticket submitBuffers( uint64_t epoch,
		const std::array<std::span<const float>, 3> &initial,
		std::span<const BufferStep> steps );
	[[nodiscard]] bool collectBuffers( Ticket ticket,
		const std::array<std::span<float>, 3> &output );
	// Resident jobs reuse the same allocations. Empty updates preserve a buffer;
	// every nonempty update must match its existing extent. No implicit resize.
	[[nodiscard]] Ticket updateBuffers( uint64_t epoch,
		const std::array<std::span<const float>, 3> &updates,
		std::span<const BufferStep> steps );
	[[nodiscard]] bool collectBufferRanges( Ticket ticket,
		const std::array<BufferReadback, 3> &output );
	// Explicit checkpoint/debug read, exclusively owned by the idle caller.
	// Never waits: outstanding (even completed but unretired) work refuses it.
	[[nodiscard]] bool readIdleBuffers( const std::array<BufferReadback, 3> &output );
	// Standalone finalization, after all tickets retire. Reports validation
	// errors through teardown as failures; the destructor is the fallback drain.
	void shutdown();
private:
	struct Impl;
	std::unique_ptr<Impl> m;
};

} // namespace gamescope::framegen::compute
