#ifndef QRT_FLA_COMPLETION_GUARD_H
#define QRT_FLA_COMPLETION_GUARD_H

#include <chrono>
#include <cmath>

namespace qrt_fla_completion {

constexpr double kLimitMs = 100.0;
enum class ClockSource { unavailable, gpu, host };

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
// No timeout is increased and no unfinished work is accepted.
inline Decision evaluate(double gpu_ms, double host_ms) {
    if (bounded(gpu_ms)) return {ClockSource::gpu, gpu_ms};
    if (bounded(host_ms)) return {ClockSource::host, host_ms};
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
