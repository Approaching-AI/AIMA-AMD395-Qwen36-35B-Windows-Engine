#include <hip/hip_runtime.h>
#include "../../native/providers/triton_moe/class_expert_candidate_order.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace order = qrt_moe_class_expert_order;
constexpr unsigned guard = 65u, sentinel = 0x5a5a5a5au, routes = 65536u;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
template<class T> struct Device {
    T* pointer = nullptr;
    explicit Device(const std::vector<T>& values) {
        check(hipMalloc(reinterpret_cast<void**>(&pointer), values.size() * sizeof(T)));
        check(hipMemcpy(pointer, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    ~Device() { if (pointer) (void)hipFree(pointer); }
    T* data() { return pointer + guard; }
    std::vector<T> read(size_t count) {
        std::vector<T> result(count);
        check(hipMemcpy(result.data(), pointer, count * sizeof(T), hipMemcpyDeviceToHost));
        return result;
    }
};
void finish(hipStream_t stream) {
    hipEvent_t event = nullptr; check(hipEventCreate(&event)); check(hipEventRecord(event, stream));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    for (;;) {
        const auto status = hipEventQuery(event); if (status == hipSuccess) break;
        require(status == hipErrorNotReady && std::chrono::steady_clock::now() < deadline, "completion deadline");
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
// Visit the same three disjoint class ranges used by the actual replay owner.
__global__ void visit(const uint32_t* indices, const uint32_t* range,
    unsigned first_cell, uint32_t* visits) {
    const unsigned begin = range[0], end = range[256u];
    for (unsigned slot = begin + blockIdx.x * blockDim.x + threadIdx.x; slot < end;
         slot += gridDim.x * blockDim.x) atomicAdd(visits + indices[slot] - first_cell, 1u);
}
void run(order::Projection projection, unsigned capacity, unsigned selected, unsigned mode) {
    const bool down = projection == order::Projection::Down;
    const unsigned columns = down ? 2048u : 512u, first = capacity * 3u;
    std::vector<uint32_t> input(capacity + 2u * guard, sentinel), count(1u + 2u * guard, sentinel);
    std::vector<int32_t> ids(routes + 2u * guard, int32_t(sentinel));
    std::vector<uint32_t> inputs(routes + 2u * guard, sentinel), weights(524288u + 2u * guard, sentinel);
    std::vector<uint32_t> storage(capacity + order::metadata_words + 2u * guard, sentinel);
    std::vector<uint32_t> visited(capacity + 2u * guard, sentinel);
    std::fill(visited.begin() + guard, visited.end() - guard, 0u);
    std::array<unsigned, order::buckets> expected_counts{};
    std::vector<unsigned char> present(capacity, 0u);
    for (unsigned row = 0u; row < routes; ++row) {
        ids[guard + row] = mode == 0u ? 17 : int32_t((row * 73u + (row / 8u) * 17u) & 255u);
        inputs[guard + row] = mode == 0u ? 3u : mode == 1u ? 1u : row % 11u == 0u ? 0u : row % 5u == 0u ? 1u : 3u;
    }
    for (unsigned row = 0u; row < 524288u; ++row)
        weights[guard + row] = mode == 0u ? 3u : mode == 1u ? 0u : row % 13u == 0u ? 0u : row % 7u == 0u ? 1u : 3u;
    // Independent host indexing covers gate/up's distinct expert halves and
    // down's per-route activation rows. Do not call the device bucket helper.
    auto bucket = [&](unsigned cell) {
        const unsigned route = cell / columns, expert = unsigned(ids[guard + route]);
        const unsigned input_row = down ? route : route / 8u;
        const unsigned weight_row = down ? expert * 2048u + cell % 2048u :
            expert * 1024u + (projection == order::Projection::Up ? 512u : 0u) + cell % 512u;
        const unsigned code = inputs[guard + input_row] & weights[guard + weight_row];
        return (code == 3u ? 2u : code == 1u ? 1u : 0u) * 256u + expert;
    };
    for (unsigned slot = 0u; slot < selected; ++slot) {
        const unsigned offset = (slot * 2654435761u) & (capacity - 1u), cell = first + offset;
        require(cell / columns < routes, "generated route span");
        input[guard + slot] = cell; present[offset] = 1u; ++expected_counts[bucket(cell)];
    }
    count[guard] = selected;
    Device<uint32_t> di(input), dc(count), da(inputs), db(weights), dw(storage), dv(visited); Device<int32_t> dt(ids);
    hipStream_t stream = nullptr; check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    check(order::launch(di.data(), dc.data(), dt.data(), {da.data(), db.data(), projection}, {dw.data(), capacity}, stream));
    const auto view = order::views({dw.data(), capacity});
    for (unsigned classification = 0u; classification < 3u; ++classification) {
        hipLaunchKernelGGL(visit, dim3(17u), dim3(256u), 0u, stream, view.indices,
            view.offsets + classification * 256u, first, dv.data()); check(hipGetLastError());
    }
    finish(stream);
    const auto actual = dw.read(storage.size()), visits = dv.read(visited.size()); unsigned previous = 0u;
    for (unsigned slot = 0u; slot < selected; ++slot) {
        const unsigned cell = actual[guard + slot]; require(cell >= first && cell < first + capacity, "cell domain");
        require(present[cell - first] == 1u, "duplicate or foreign candidate"); present[cell - first] = 2u;
        const unsigned current = bucket(cell); require(!slot || current >= previous, "class/expert bucket order"); previous = current;
    }
    for (unsigned i = 0u; i < capacity; ++i) {
        require(present[i] != 1u, "missing candidate");
        require(visits[guard + i] == unsigned(present[i] == 2u), "range consumption exactly once");
    }
    const size_t counts = guard + capacity, offsets = counts + order::buckets, cursors = offsets + order::buckets + 1u;
    unsigned total = 0u;
    for (unsigned index = 0u; index < order::buckets; ++index) {
        require(actual[counts + index] == expected_counts[index] && actual[offsets + index] == total, "histogram/prefix");
        total += expected_counts[index]; require(actual[cursors + index] == total, "cursor endpoint");
    }
    require(total == selected && actual[offsets + order::buckets] == selected, "complete count");
    for (size_t i = 0u; i < actual.size(); ++i)
        if (i < guard || (i >= guard + selected && i < counts) || i >= cursors + order::buckets)
            require(actual[i] == sentinel, "workspace guard or unused indices");
    for (unsigned i = 0u; i < guard; ++i)
        require(visits[i] == sentinel && visits[guard + capacity + i] == sentinel, "visit guard");
    require(di.read(input.size()) == input && dc.read(count.size()) == count && dt.read(ids.size()) == ids &&
        da.read(inputs.size()) == inputs && db.read(weights.size()) == weights, "immutable input or classification");
    check(hipStreamDestroy(stream));
    std::printf("{\"kind\":\"moe_class_expert_order_safety\",\"projection\":%u,\"capacity\":%u,\"selected\":%u,\"mode\":%u,\"buckets\":768,\"workspace_bytes\":%zu,\"exact_permutation\":true,\"ordered_classes_and_experts\":true,\"range_consumed_once\":true,\"all_metadata_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
        unsigned(projection), capacity, selected, mode, order::bytes(capacity)); std::fflush(stdout);
}
int main() try {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    require(std::strncmp(properties.gcnArchName, "gfx1151", 7u) == 0, "requires gfx1151");
    for (auto projection : {order::Projection::Gate, order::Projection::Up, order::Projection::Down})
        for (unsigned capacity : {256u, 4096u, 4194304u})
            for (unsigned selected : {0u, 1u, capacity / 2u + 17u, capacity})
                for (unsigned mode : {0u, 1u, 2u}) run(projection, capacity, selected, mode);
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
