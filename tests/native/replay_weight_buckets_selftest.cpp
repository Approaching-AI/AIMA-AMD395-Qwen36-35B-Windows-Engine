#include "../../native/providers/moe_accumulator/replay_weight_buckets.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
namespace b = qrt_replay_weight_buckets;
namespace {
constexpr unsigned guard = 64u, marker = 0xa5a5a5a5u;
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
void require(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event); if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        require(std::chrono::steady_clock::now() < deadline, "weight bucket completion deadline"); std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
struct Buffer {
    unsigned* base = nullptr; size_t size;
    explicit Buffer(size_t n) : size(n) { check(hipMalloc(reinterpret_cast<void**>(&base), (n + 2u * guard) * 4u)); reset(); }
    ~Buffer() { if (base && hipFree(base) != hipSuccess) std::abort(); }
    unsigned* data() { return base + guard; }
    void reset() { check(hipMemset(base, marker & 255u, (size + 2u * guard) * 4u)); }
    void upload(const std::vector<unsigned>& data) { require(data.size() == size, "bucket upload size"); if (size) check(hipMemcpy(this->data(), data.data(), size * 4u, hipMemcpyHostToDevice)); }
    std::vector<unsigned> read() {
        std::vector<unsigned> all(size + 2u * guard); check(hipMemcpy(all.data(), base, all.size() * 4u, hipMemcpyDeviceToHost));
        for (size_t i = 0u; i < guard; ++i) require(all[i] == marker && all[guard + size + i] == marker, "bucket buffer guard changed");
        return {all.begin() + guard, all.end() - guard};
    }
};
void run(b::Plan plan, unsigned mode) {
    const unsigned cells = plan.rows * plan.tokens, bins = b::bucket_count(plan);
    std::vector<unsigned> selected;
    if (mode) for (unsigned cell = 0u; cell < cells; ++cell) {
        const unsigned stride = cells > 1048576u ? 113u : 3u;
        if (mode == 1u || (cell * 37u + cell / plan.rows * 11u) % stride == 0u) selected.push_back(cell);
    }
    // A bijective source permutation removes reliance on source sorting.
    if (mode == 3u) std::reverse(selected.begin(), selected.end());
    Buffer input(selected.size()), output(selected.size()), workspace(b::workspace_words(plan) + 17u);
    input.upload(selected);
    check(b::launch(input.data(), selected.size(), unsigned(selected.size()), plan, output.data(), selected.size(), workspace.data(), workspace.size, nullptr)); finish();
    const auto actual = output.read(), work = workspace.read(); require(input.read() == selected, "bucket source changed");
    const unsigned* histogram = work.data(); const unsigned* starts = histogram + bins; const unsigned* cursors = starts + bins + 1u;
    require(cursors[bins] == 0u && starts[0] == 0u && starts[bins] == selected.size(), "bucket total/status");
    std::vector<unsigned> expected_histogram(bins);
    // Independent wide arithmetic for every bucket identity.
    const uint64_t columns = (uint64_t(plan.rows) + plan.weight_rows - 1u) / plan.weight_rows;
    auto independent_bin = [&](unsigned cell) { return unsigned(uint64_t(cell / plan.rows / plan.token_rows) * columns + cell % plan.rows / plan.weight_rows); };
    for (unsigned cell : selected) ++expected_histogram[independent_bin(cell)];
    for (unsigned bin = 0u; bin < bins; ++bin) {
        require(histogram[bin] == expected_histogram[bin] && starts[bin + 1u] - starts[bin] == histogram[bin] && cursors[bin] == starts[bin + 1u], "bucket histogram/prefix/cursor");
        for (unsigned slot = starts[bin]; slot < starts[bin + 1u]; ++slot) require(actual[slot] < cells && independent_bin(actual[slot]) == bin, "bucket membership");
    }
    auto permutation = actual; std::sort(permutation.begin(), permutation.end()); std::sort(selected.begin(), selected.end());
    require(permutation == selected, "bucket full permutation");
    for (size_t i = b::workspace_words(plan); i < work.size(); ++i) require(work[i] == marker, "bucket unused workspace tail");
    std::printf("{\"kind\":\"weight_bucket_native_permutation\",\"rows\":%u,\"tokens\":%u,\"weight_rows\":%u,\"token_rows\":%u,\"mode\":%u,\"candidates\":%zu,\"buckets\":%u,\"permutation_mismatches\":0,\"histogram_and_cursor_pass\":true,\"redzones_pass\":true,\"unused_workspace_tail_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n", plan.rows,plan.tokens,plan.weight_rows,plan.token_rows,mode,selected.size(),bins);
}
void invalid() {
    const b::Plan plan{17u,65u,16u,256u};
    const std::vector<unsigned> indices{0u,1104u,1105u,UINT32_MAX};
    Buffer input(indices.size()), output(indices.size()), workspace(b::workspace_words(plan));input.upload(indices);
    unsigned rejections = 0u;
    auto reject = [&](hipError_t status) { require(status == hipErrorInvalidValue, "bucket host validation accepted invalid view"); ++rejections; };
    reject(b::launch(nullptr,4u,4u,plan,output.data(),4u,workspace.data(),workspace.size,nullptr));
    reject(b::launch(input.data(),3u,4u,plan,output.data(),4u,workspace.data(),workspace.size,nullptr));
    reject(b::launch(input.data(),4u,4u,plan,output.data(),3u,workspace.data(),workspace.size,nullptr));
    reject(b::launch(input.data(),4u,4u,plan,output.data(),4u,workspace.data(),workspace.size-1u,nullptr));
    reject(b::launch(input.data(),4u,4u,plan,input.data(),4u,workspace.data(),workspace.size,nullptr));
    reject(b::launch(input.data(),4u,4u,plan,workspace.data(),4u,workspace.data(),workspace.size,nullptr));
    reject(b::launch(input.data(),4u,4u,{17u,65u,0u,256u},output.data(),4u,workspace.data(),workspace.size,nullptr));
    require(output.read() == std::vector<unsigned>(4u,marker) && workspace.read() == std::vector<unsigned>(workspace.size,marker), "invalid view mutated storage");
    check(b::launch(input.data(),4u,4u,plan,output.data(),4u,workspace.data(),workspace.size,nullptr));finish();
    const auto work = workspace.read();require((work.back() & 3u) == 3u, "invalid index status missing");
    require(output.read() == std::vector<unsigned>(4u,marker) && input.read() == indices, "invalid indices wrote output or input");
    std::printf("{\"kind\":\"weight_bucket_native_invalid\",\"rejected_host_views\":%u,\"rejected_device_indices\":2,\"scatter_suppressed\":true,\"redzones_pass\":true,\"immutable_inputs\":true}\n",rejections);
}
}
int main() try {
    hipDeviceProp_t device{};check(hipGetDeviceProperties(&device,0));require(!std::strncmp(device.gcnArchName,"gfx1151",7u),"requires gfx1151");
    const unsigned shapes[][2]={{1u,1u},{17u,65u},{64u,256u},{129u,257u},{2048u,8192u},{8192u,8192u},{16384u,8192u}};
    for (const auto& shape : shapes) for (unsigned policy=0u;policy<3u;++policy) {
        const b::Plan plan{shape[0],shape[1],policy==0u?1u:policy==1u?16u:64u,policy==0u?8192u:policy==1u?256u:64u};
        for (unsigned mode : {0u,2u,3u}) run(plan,mode);
        if (shape[0]<=129u) run(plan,1u);
    }
    invalid();return 0;
} catch (const std::exception& error) { std::fprintf(stderr,"%s\n",error.what());return 1; }
