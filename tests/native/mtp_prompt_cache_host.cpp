// Host orchestration only: queued mocks do not validate any GPU arithmetic.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>
using hipError_t = int;
using hipStream_t = void*;
constexpr int hipSuccess = 0, hipErrorInvalidValue = 1, hipErrorOutOfMemory = 2;
constexpr int hipMemcpyDeviceToHost = 3, injected = 99;
static std::map<void*, bool> allocations;
static std::vector<std::function<void()>> queued;
static std::vector<std::string> stages;
static unsigned int allocation_call = 0, fail_allocation = 0, sync_call = 0, fail_sync = 0;
static std::string fail_stage;
static bool invalid_token = false;
static hipStream_t expected_stream = reinterpret_cast<void*>(0x1234);
static hipError_t allocate(void** p, size_t bytes, bool host) {
    if (++allocation_call == fail_allocation) return hipErrorOutOfMemory;
    *p = std::malloc(bytes); assert(*p); allocations.emplace(*p, host);
    std::memset(*p, 0xa5, bytes); return hipSuccess;
}
static hipError_t hipMalloc(void** p, size_t n) { return allocate(p, n, false); }
static hipError_t hipHostMalloc(void** p, size_t n) { return allocate(p, n, true); }
static hipError_t release(void* p, bool host) {
    assert(allocations.count(p) && allocations.at(p) == host);
    allocations.erase(p); std::free(p); return hipSuccess;
}
static hipError_t hipFree(void* p) { return release(p, false); }
static hipError_t hipHostFree(void* p) { return release(p, true); }
static hipError_t enqueue(const char* stage, hipStream_t stream, std::function<void()> fn) {
    assert(stream == expected_stream); stages.emplace_back(stage); queued.push_back(fn);
    return fail_stage == stage ? injected : hipSuccess;
}
static hipError_t hipStreamSynchronize(hipStream_t stream) {
    assert(stream == expected_stream);
    if (++sync_call == fail_sync) return injected;
    auto work = std::move(queued); queued.clear(); for (auto& fn : work) fn(); return hipSuccess;
}
static hipError_t hipMemsetAsync(void* p, int value, size_t n, hipStream_t stream) {
    return enqueue("clear", stream, [=] { std::memset(p, value, n); });
}
static hipError_t hipMemcpyAsync(void* p, const void* q, size_t n, int kind, hipStream_t stream) {
    assert(kind == hipMemcpyDeviceToHost && allocations.at(p));
    return enqueue("copy", stream, [=] { std::memcpy(p, q, n); });
}
namespace qrt_sm121_mtp {
static bool observed_split1024 = false;
static hipError_t launch_fusion_inputs(const uint16_t*, const uint16_t*, const uint32_t*,
    const uint16_t*, const uint16_t*, const unsigned char*, unsigned rows, uint16_t* out,
    uint32_t* invalid, hipStream_t stream, bool split1024) {
    observed_split1024 = split1024;
    return enqueue("gather", stream, [=] {
        *invalid = invalid_token ? 1u : 0u; std::fill_n(out, rows * 4096u, 11u);
    });
}
static hipError_t launch_normalize(const uint16_t* in, const uint16_t*, const unsigned char*,
    unsigned rows, uint16_t* out, hipStream_t stream) {
    return enqueue("norm", stream, [=] {
        assert(in[0] == 22u); std::fill_n(out, rows * 2048u, 33u);
    });
}
static hipError_t launch_key_values(const uint16_t* in, const uint16_t*, const unsigned char*,
    const uint16_t*, unsigned, unsigned first, unsigned rows, unsigned capacity,
    uint16_t* out, uint16_t*, hipStream_t stream) {
    assert(first + rows <= capacity);
    return enqueue("cache", stream, [=] {
        assert(in[0] == 44u); std::fill_n(out + first * 1024u, rows * 1024u, 55u);
    });
}
}
#include "sm121_mtp_prompt_cache_under_test.h"

static hipError_t project(void*, const uint16_t*, const uint16_t* in, uint16_t* out,
    unsigned output, unsigned input, unsigned rows, hipStream_t stream) {
    const bool fc = output == 2048u;
    assert((fc && input == 4096u) || (output == 1024u && input == 2048u));
    return enqueue(fc ? "fc" : "kv", stream, [=] {
        assert(in[0] == (fc ? 11u : 33u)); std::fill_n(out, rows * output, fc ? 22u : 44u);
    });
}
static void reset_faults() {
    assert(queued.empty()); allocation_call = sync_call = fail_allocation = fail_sync = 0;
    fail_stage.clear(); invalid_token = false; stages.clear();
}
static qrt_sm121_mtp::PromptStep append(qrt_sm121_mtp::PromptCache& cache, unsigned first,
                                      unsigned rows, bool split1024 = false) {
    uint16_t value = 0; uint32_t id = 1; unsigned char table = 0;
    qrt_sm121_mtp::PromptWeights weights{&value,&value,&value,&value,&value,&value,&value};
    weights.split1024_pre_fc_norm = split1024;
    return cache.append(weights, &value, &id, &table, &value, 8u, first, rows,
                        project, nullptr, expected_stream);
}
int main() {
    using qrt_sm121_mtp::PromptCache;
    // Every allocation failure preserves existing ownership and frees candidates.
    for (unsigned fail = 1; fail <= 7u; ++fail) {
        reset_faults();
        { PromptCache c; assert(c.reserve(4u,1u) == hipSuccess);
          const auto* old = c.data(); reset_faults(); fail_allocation = fail;
          assert(c.reserve(8u,2u) == hipErrorOutOfMemory);
          assert(c.data() == old && c.capacity() == 4u && allocations.size() == 7u);
        } assert(allocations.empty());
    }
    reset_faults();
    { PromptCache c; assert(c.reserve(8u,2u) == hipSuccess);
      assert(c.allocated_bytes() == 8u*2048u + 2u*18432u + 8u);
      assert(!c.tail(0u,1u).fusion);
      assert(append(c,0u,2u).status == hipSuccess && c.retained_tokens() == 2u);
      const auto first_tail = c.tail(0u,2u), last_tail = c.tail(1u,1u);
      assert(c.valid_tail(first_tail) && c.valid_tail(last_tail));
      assert(first_tail.fusion[0] == 22u && first_tail.normalized[0] == 33u);
      assert(last_tail.fusion == first_tail.fusion + 2048u);
      assert(last_tail.normalized == first_tail.normalized + 2048u);
      assert(!c.tail(0u,0u).fusion && !c.tail(0u,3u).fusion && !c.tail(2u,1u).fusion);
      assert(!c.tail(~0u,1u).fusion && !c.tail(1u,~0u).fusion);
      assert(!qrt_sm121_mtp::observed_split1024);
      assert((stages == std::vector<std::string>{"clear","gather","copy","fc","norm","kv","cache"}));
      const auto* pointer = c.data(); const auto calls = stages.size();
      assert(append(c,1u,2u).status == hipErrorInvalidValue && stages.size() == calls);
      assert(c.valid_tail(first_tail)); // Rejected before modifying scratch.
      assert(c.reserve(9u,2u) == hipErrorInvalidValue && c.data() == pointer);
      assert(!c.truncate(3u) && c.truncate(1u));
      assert(!c.valid_tail(first_tail) && !c.valid_tail(last_tail));
      assert(append(c,1u,2u,true).status == hipSuccess && c.retained_tokens() == 3u);
      const auto next_tail = c.tail(1u,2u);
      assert(c.valid_tail(next_tail) && !c.valid_tail(first_tail));
      assert(next_tail.generation != first_tail.generation);
      assert(c.truncate(3u) && c.valid_tail(next_tail));
      assert(qrt_sm121_mtp::observed_split1024);
      assert(c.data()[3u*1024u] == 0xa5a5u); // Unpublished tail untouched.
    } assert(allocations.empty());
    // Recoverable launch failures drain partially queued work and preserve prefix.
    for (const char* stage : {"clear","gather","copy","fc","norm","kv","cache"}) {
        reset_faults();
        { PromptCache c; assert(c.reserve(8u,2u) == hipSuccess); assert(append(c,0u,1u).status == hipSuccess);
          const auto view = c.tail(0u,1u); assert(c.valid_tail(view));
          reset_faults(); fail_stage = stage; auto result = append(c,1u,2u);
          assert(result.status == injected && !result.completion_unknown && !c.quarantined());
          assert(c.retained_tokens() == 1u && c.data()[0] == 55u && queued.empty());
          assert(!c.valid_tail(view) && !c.tail(0u,1u).fusion);
          reset_faults(); assert(append(c,1u,2u).status == hipSuccess && c.retained_tokens() == 3u);
        } assert(allocations.empty());
    }
    reset_faults();
    { PromptCache c; assert(c.reserve(8u,2u) == hipSuccess); invalid_token = true;
      auto result = append(c,0u,2u); assert(result.status == hipErrorInvalidValue);
      assert(std::string(result.stage) == "invalid_token" && c.retained_tokens() == 0u);
      assert(stages.size() == 3u && queued.empty() && c.data()[0] == 0xa5a5u);
    } assert(allocations.empty());
    // Both normal fences and failure-drain fences quarantine all ownership.
    for (unsigned fence = 0; fence < 3u; ++fence) {
        reset_faults();
        { PromptCache c; assert(c.reserve(8u,2u) == hipSuccess);
          fail_sync = fence == 1u ? 2u : 1u; if (fence == 2u) fail_stage = "copy";
          auto result = append(c,0u,2u);
          assert(result.status == injected && result.completion_unknown && c.quarantined());
          assert(!c.data() && !c.truncate(0u) && c.reserve(8u,2u) == injected);
          const auto calls = stages.size(); assert(append(c,0u,1u).completion_unknown);
          assert(stages.size() == calls);
        }
        assert(allocations.size() == 7u && !queued.empty());
        // Simulate late device completion after owner destruction. In particular
        // the async host destination must remain valid (checked by ASan).
        fail_sync = 0; assert(hipStreamSynchronize(expected_stream) == hipSuccess);
        while (!allocations.empty()) release(allocations.begin()->first, allocations.begin()->second);
    }
}
