#pragma once
#include "inplace_probability_pipeline.h"

// Isolated storage variant. The full producer/replay owner is shared with the
// tagged candidate; only the score-to-probability representation differs.
namespace qrt_packed_probability_pipeline {
inline constexpr auto probability = qrt_inplace_probability_pipeline::probability_storage<true>;
inline constexpr auto launch = qrt_inplace_probability_pipeline::launch_storage<true>;
} // namespace qrt_packed_probability_pipeline
