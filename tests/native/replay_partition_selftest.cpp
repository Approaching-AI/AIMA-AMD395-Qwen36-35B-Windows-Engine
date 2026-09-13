#include "../../native/providers/moe_accumulator/sm121_replay_partition.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

constexpr unsigned guard = 65u, sentinel = 0xa5a5a5a5u;
void check(hipError_t x) { if (x != hipSuccess) throw std::runtime_error(hipGetErrorString(x)); }
void complete() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("partition deadline");
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
struct Buffer {
    unsigned* base = nullptr; size_t size;
    explicit Buffer(const std::vector<unsigned>& v):size(v.size()) {
        check(hipMalloc(reinterpret_cast<void**>(&base), size * sizeof(unsigned)));
        check(hipMemcpy(base, v.data(), size * sizeof(unsigned), hipMemcpyHostToDevice));
    }
    ~Buffer() { if (base) (void)hipFree(base); }
    unsigned* data() { return base + guard; }
    std::vector<unsigned> read() {
        std::vector<unsigned> v(size); check(hipMemcpy(v.data(), base, size * sizeof(unsigned), hipMemcpyDeviceToHost)); return v;
    }
};
__global__ void capacity_kernel(const unsigned* indices, unsigned* output,
    unsigned* counts, const unsigned* weights, const unsigned* inputs,
    unsigned rows, unsigned count, unsigned capacity) {
    const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = slot < count;
    const unsigned cell = valid ? indices[slot] : 0u;
    const bool floating = valid && (weights[cell % rows] & inputs[cell / rows] & 1u);
    qrt_sm121_replay_partition::append_block(cell, valid, floating, output, counts, capacity);
}
void run(unsigned count, unsigned mode, unsigned gap) {
    constexpr unsigned rows = 37u;
    const unsigned tokens = (count + rows - 1u) / rows + 1u, cells = rows * tokens;
    const unsigned capacity = count + gap;
    std::vector<unsigned> input(count + 2u * guard, sentinel), output(capacity + 2u * guard, sentinel);
    std::vector<unsigned> weights(rows + 2u * guard, sentinel), inputs(tokens + 2u * guard, sentinel);
    std::vector<unsigned> counts(2u + 2u * guard, sentinel);
    counts[guard] = counts[guard + 1u] = 0u;
    for (unsigned row = 0; row < rows; ++row) weights[guard + row] = mode == 0u ? 0u : mode == 1u ? 3u : unsigned(row % 7u != 1u);
    for (unsigned token = 0; token < tokens; ++token) inputs[guard + token] = mode < 2u ? 3u : unsigned(token % 5u != 1u);
    unsigned floating = 0u;
    for (unsigned i = 0; i < count; ++i) {
        const unsigned cell = cells - 1u - i;
        input[guard + i] = cell;
        floating += (weights[guard + cell % rows] & inputs[guard + cell / rows] & 1u) != 0u;
    }
    Buffer di(input), out(output), dc(counts), dw(weights), dx(inputs);
    hipLaunchKernelGGL(capacity_kernel,
        dim3(std::max(1u, (count + 255u) / 256u)), dim3(256u), 0u, nullptr,
        di.data(), out.data(), dc.data(), dw.data(), dx.data(), rows, count, capacity);
    check(hipGetLastError()); complete();
    auto actual = out.read(), counter = dc.read();
    if (counter[guard] != floating || counter[guard + 1u] != count - floating) throw std::runtime_error("partition count differs");
    std::vector<unsigned> sorted;
    for (unsigned i = 0u; i < count; ++i) {
        const unsigned physical = i < floating ? i : capacity - 1u - (i - floating);
        const unsigned cell = actual[guard + physical];
        sorted.push_back(cell);
        if (cell >= cells) throw std::runtime_error("partition index out of range");
        const bool is_float = (weights[guard + cell % rows] & inputs[guard + cell / rows] & 1u) != 0u;
        if (is_float != (i < floating)) throw std::runtime_error("partition class differs");
    }
    for (unsigned i = floating; i < floating + gap; ++i)
        if (actual[guard + i] != sentinel) throw std::runtime_error("unused partition gap changed");
    auto original = std::vector<unsigned>(input.begin() + guard, input.end() - guard);
    std::sort(sorted.begin(), sorted.end()); std::sort(original.begin(), original.end());
    if (sorted != original) throw std::runtime_error("partition is not an exact permutation");
    for (unsigned i = 0; i < guard; ++i)
        if (actual[i] != sentinel || actual[guard + capacity + i] != sentinel ||
            counter[i] != sentinel || counter[guard + 2u + i] != sentinel) throw std::runtime_error("partition redzone");
    if (di.read() != input || dw.read() != weights || dx.read() != inputs) throw std::runtime_error("partition input changed");
    std::printf("{\"kind\":\"exact_replay_partition\",\"count\":%u,\"capacity\":%u,\"mode\":%u,\"floating\":%u,\"integer\":%u,\"permutation_mismatches\":0,\"class_mismatches\":0,\"unused_gap_unchanged\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",count,capacity,mode,floating,count-floating);
}
int main() try {
    hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0));
    if (std::string(device.gcnArchName).find("gfx1151") != 0u) throw std::runtime_error("requires gfx1151");
    for (unsigned count : {0u,1u,17u,255u,256u,257u,4099u,65539u})
        for (unsigned mode = 0u; mode < 3u; ++mode)
            for (unsigned gap : {0u,19u}) run(count,mode,gap);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr,"replay_partition_error=%s\n",e.what()); return 2; }
