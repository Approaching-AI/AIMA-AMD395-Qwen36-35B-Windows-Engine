#include "../../native/providers/gdn/completion_guard.h"
#include <cassert>
#include <cstdio>
#include <limits>

struct FakeClock {
    using duration = std::chrono::duration<double, std::milli>;
    using time_point = std::chrono::time_point<FakeClock>;
    static constexpr bool is_steady = true;
    static double value;
    static time_point now() { return time_point{duration{value}}; }
};
double FakeClock::value = 0.0;

int main() {
    using namespace qrt_fla_completion;
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    struct Case { double value; bool acceptable; };
    const Case cases[] = {
        {-inf,false},{-1,false},{-0.0,true},{0,true},{1,true},
        {std::nextafter(100.0,0.0),true},{100,true},
        {std::nextafter(100.0,inf),false},{101,false},{inf,false},{nan,false}
    };
    unsigned comparisons = 0, host_fallbacks = 0, rejections = 0;
    for (const auto gpu : cases) for (const auto host : cases) {
        const auto d = evaluate(gpu.value,host.value);
        assert(d.accepted() == (gpu.acceptable || host.acceptable));
        if (gpu.acceptable) {
            assert(d.source == ClockSource::gpu && d.milliseconds == gpu.value);
        } else if (host.acceptable) {
            assert(d.source == ClockSource::host && d.milliseconds == host.value);
            ++host_fallbacks;
        } else {
            assert(d.source == ClockSource::unavailable);
            ++rejections;
        }
        ++comparisons;
    }
    FakeClock::value = 7.0;
    const Timer<FakeClock> timer;
    FakeClock::value = 107.0;
    assert(timer.elapsed_ms() == 100.0 && evaluate(inf,timer.elapsed_ms()).accepted());
    FakeClock::value = std::nextafter(107.0,inf);
    assert(timer.elapsed_ms() > 100.0 && !evaluate(inf,timer.elapsed_ms()).accepted());
    FakeClock::value = 6.0;
    assert(!evaluate(inf,timer.elapsed_ms()).accepted());
    const Timer<> real_timer;
    assert(std::isfinite(real_timer.elapsed_ms()) && real_timer.elapsed_ms() >= 0.0);
    std::printf("{\"kind\":\"fla_completion_guard_host\",\"comparisons\":%u,\"host_fallbacks\":%u,\"rejections\":%u,\"timer_checks\":4,\"guard_ms\":100,\"pass\":true}\n",
                comparisons,host_fallbacks,rejections);
}
