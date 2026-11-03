#pragma once

#include "vulkan_engine.hpp"
#include "framegen/training_report.hpp"

#include <mutex>

namespace gamescope::framegen::compute
{
struct TrainingSample
{
	std::span<const float> source, destination, forward, reverse;
};

struct TrainingCheckpoint
{
	uint64_t epoch;
	uint32_t key, acceptedStep;
	// [fast,m,v,EMA,served], all from one accepted GPU revision.
	std::vector<float> values;
	// Empty unless an explicit full diagnostic snapshot was requested.
	std::vector<float> gradients;
};

// Experimental offscreen owner. No compositor adapter or external-memory ABI.
// Fixed tensor shape and key per context; callers assign a unique nonzero model
// epoch across replacements. All methods serialize access to the GPU owner.
// Accepted GPU state is private: training writes a candidate, then full-model
// validation publishes it or rolls the candidate back to the accepted bank.
class ResidentTrainer
{
public:
	ResidentTrainer( DeviceUuid presenter, uint64_t modelEpoch, uint32_t key,
		uint32_t width, uint32_t height, std::span<const float> prior, bool validation = false );
	~ResidentTrainer();
	ResidentTrainer( const ResidentTrainer & ) = delete;
	ResidentTrainer &operator=( const ResidentTrainer & ) = delete;
	// Busy admission returns an empty ticket. Input errors throw before mutation.
	// Adam's step advances only on acceptance; attempt identity advances on every
	// submitted job. This API does not offer cancellation of GPU publication.
	[[nodiscard]] Ticket submit( const TrainingSample &sample, uint32_t flags = 0,
		uint32_t sceneCut = 0, float lr = 3e-4f, float ema = 0.0625f, float decay = 0.0005f );
	[[nodiscard]] Status poll( Ticket ticket );
	// Pending/stale returns nullopt; malformed reports poison the context.
	// A valid rejected report is returned with accepted=false.
	[[nodiscard]] std::optional<TrainingReport> collect( Ticket ticket );
	// Explicit CPU snapshot only. Never waits for outstanding training. Holds
	// exclusive ownership across memory invalidation and copying. No file I/O.
	[[nodiscard]] TrainingCheckpoint checkpoint( bool fullDiagnostics = false );
	void shutdown();
private:
	void healthy() const;
	std::mutex m_mutex;
	VulkanEngine m_engine;
	uint64_t m_epoch;
	uint32_t m_key, m_width, m_height;
	uint32_t m_optimizer, m_health, m_publish;
	uint32_t m_acceptedStep = 0, m_attempt = 0;
	bool m_initialized = false, m_failed = false, m_closed = false;
	Ticket m_ticket{}, m_engineTicket{};
	TrainingReportIdentity m_expected{};
	std::vector<float> m_input, m_initial, m_zeroGradients;
};
} // namespace gamescope::framegen::compute
