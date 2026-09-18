#ifndef QRT_FLA_COMPLETION_GUARD_H
#define QRT_FLA_COMPLETION_GUARD_H

#include <chrono>
#include <cmath>

namespace qrt_fla_completion {

constexpr double kLimitMs = 100.0;
enum class ClockSource { unavailable, gpu, host };

struct Observation {
    bool completed = false;
    double gpu_ms = 0.0;
    double host_ms = 0.0;
};

struct Decision {
    ClockSource source;
    double milliseconds;
    bool accepted() const { return source != ClockSource::unavailable; }
};

inline bool bounded(double milliseconds) {
    return std::isfinite(milliseconds) && milliseconds >= 0.0 &&
           milliseconds <= kLimitMs;
}

// Called only after successful end-event synchronization. A monotonic host
// interval enclosing submission and completion independently bounds all GPU
// work in that interval, including queue and scheduling delays. It can prove
// the same limit when the GPU event clock reports an invalid or larger value.
// This strict nominal bound remains available to standalone diagnostics.
inline Decision evaluate(double gpu_ms, double host_ms) {
    if (bounded(gpu_ms)) return {ClockSource::gpu, gpu_ms};
    if (bounded(host_ms)) return {ClockSource::host, host_ms};
    return {ClockSource::unavailable, 0.0};
}

// A completed HIP operation does not become a runtime failure because its
// elapsed interval exceeds a nominal latency. The event interval can include
// scheduling and memory delays; the enclosing process deadline still bounds
// execution. Keep the original preferred clocks for ordinary intervals, then
// retain a finite completed interval for reporting a latency outlier. Callers
// must first check launch, end-event synchronization and elapsed-time API status.
inline Decision evaluate_completed(double gpu_ms, double host_ms) {
    const auto nominal = evaluate(gpu_ms, host_ms);
    if (nominal.accepted()) return nominal;
    if (std::isfinite(gpu_ms) && gpu_ms >= 0.0) return {ClockSource::gpu, gpu_ms};
    if (std::isfinite(host_ms) && host_ms >= 0.0) return {ClockSource::host, host_ms};
    return {ClockSource::unavailable, 0.0};
}

template<class Clock = std::chrono::steady_clock>
class Timer {
    static_assert(Clock::is_steady, "completion guard requires a monotonic clock");
    typename Clock::time_point begin_ = Clock::now();
public:
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - begin_).count();
    }
};

} // namespace qrt_fla_completion
#endif
