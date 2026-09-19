"""Compare actual optional allocation requests with the exported accounting helper."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class MoeScratchAccountingTests(unittest.TestCase):
    def test_optional_owners_and_partial_allocations_match_requests(self):
        provider = (ROOT / 'native/providers/triton_moe/qrt_triton_moe_q8192_provider.cpp').read_text()
        row = (ROOT / 'native/providers/moe_accumulator/sm121_scaled_half_products.h').read_text()
        declarations = provider[provider.index('enum class MoeL2 :'):provider.index('struct MoeWeightMetadata {')]
        declarations += provider[provider.index('constexpr size_t kMoePreparedWeightElements'):provider.index('enum class MoeCorrectionPhase')]
        order = ''
        for namespace, file in (
            ('qrt_moe_expert_order', 'expert_candidate_order.h'),
            ('qrt_moe_class_expert_order', 'class_expert_candidate_order.h'),
        ):
            text = (ROOT / 'native/providers/triton_moe' / file).read_text()
            first = text.index('constexpr unsigned experts')
            last = text.index('struct Workspace')
            order += 'namespace ' + namespace + '{\n' + text[first:last]
            order += function(text, 'inline size_t bytes(') + '\n}\n'
        # These source spans also declare constants used by kernels outside
        # the host fixture. Keep their original expressions without requiring
        # those unrelated kernels to make every declaration used here.
        declarations = declarations.replace('constexpr ', '[[maybe_unused]] constexpr ')
        order = order.replace('constexpr ', '[[maybe_unused]] constexpr ')
        actual = '\n'.join(function(provider, name) for name in (
            'bool allocate_optional_moe_l2()',
            'bool allocate_optional_moe_compaction()',
            'bool allocate_optional_moe_prepared_replay()',
            'bool allocate_optional_moe_shared_replay()',
            'bool allocate_optional_shared_projection_hawkeye()',
            'bool allocate_optional_hipblaslt_router_logits()',
            'uint64_t optional_moe_scratch_bytes()',
        ))
        source = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
constexpr unsigned kTokens=TOKEN_COUNT,kExperts=256,kHidden=2048,kIntermediate=512;
constexpr unsigned kRoutes=kTokens*8u,kNativeThreads=256;
constexpr size_t kSharedProjectionElements=size_t(kTokens)*kIntermediate;
constexpr size_t kOutputElements=size_t(kTokens)*kHidden;
namespace qrt_sm121_staged_half_projection {
''' + function(row, 'struct Row {') + r''';
static_assert(sizeof(Row)==36u);
}
''' + declarations + order + r'''
struct State {
    unsigned sm121_moe_absolute_error_ppb=0,moe_compaction_blocks=1024;
    unsigned shared_projection_hawkeye_midpoint_radius=0,router_hawkeye_midpoint_radius=0;
    bool compact_routed_hawkeye=false,partition_replay=false;
    bool moe_expert_order=false,moe_class_expert_order=false;
    bool prepared_replay=false,prevalidated_float=false,staged_half_replay=false;
    bool shared_prevalidated_float=false,shared_staged_half=false;
    uint16_t *router_logits_bf16=nullptr,*prepared_replay_weights=nullptr,*prepared_replay_inputs=nullptr;
    float *router_logits_f32=nullptr,*shared_gate_projection_f32=nullptr;
    float *shared_up_projection_f32=nullptr,*shared_down_projection_f32=nullptr;
    uint32_t *moe_compacted_indices=nullptr,*moe_compacted_count=nullptr,*moe_expert_order_storage=nullptr;
    uint32_t *prepared_replay_weight_rows=nullptr,*prepared_replay_input_rows=nullptr;
    uint32_t *moe_half_weight_classes=nullptr,*moe_half_input_classes=nullptr;
    std::array<float*,size_t(MoeL2::Count)> moe_l2{};
    std::array<uint32_t*,size_t(MoeL2::Count)> shared_replay_rows{};
    std::array<uint16_t*,size_t(MoeL2::Count)> shared_staged_operands{};
} g_state;
bool router_requested=false;
bool q8192_hipblaslt_bf16_router_requested(){return router_requested;}
std::map<void*,size_t> allocated;
unsigned calls=0,fail_at=0;
uint64_t requested_bytes=0,observations=0;
uint64_t optional_moe_scratch_bytes();
template<class T> bool allocate(T** output,size_t bytes,const char*) {
    assert(output && !*output && bytes);
    if(++calls==fail_at){assert(optional_moe_scratch_bytes()==requested_bytes);return false;}
    // Reserve a host identity only. The tested byte count is the actual
    // allocation function's request; no multi-gigabyte mock storage is needed.
    *output=static_cast<T*>(std::malloc(1));assert(*output);
    assert(allocated.emplace(*output,bytes).second);
    requested_bytes+=bytes;
    assert(optional_moe_scratch_bytes()==requested_bytes);++observations;
    return true;
}
''' + actual + r'''
void reset(unsigned mask,unsigned blocks,unsigned failure=0) {
    assert(optional_moe_scratch_bytes()==requested_bytes);
    for(auto entry:allocated)std::free(entry.first);
    allocated.clear();g_state=State{};calls=0;fail_at=failure;requested_bytes=0;
    assert(optional_moe_scratch_bytes()==0);
    g_state.moe_compaction_blocks=blocks;
    g_state.sm121_moe_absolute_error_ppb=(mask&1u)?1000u:0u;
    router_requested=mask&2u;
    g_state.router_hawkeye_midpoint_radius=(mask&4u)?512u:0u;
    g_state.compact_routed_hawkeye=mask&8u;
    g_state.partition_replay=mask&16u;
    g_state.moe_expert_order=mask&32u;
    g_state.moe_class_expert_order=mask&64u;
    g_state.prepared_replay=mask&128u;
    g_state.prevalidated_float=mask&256u;
    g_state.staged_half_replay=mask&512u;
    g_state.shared_prevalidated_float=mask&1024u;
    g_state.shared_staged_half=mask&2048u;
    g_state.shared_projection_hawkeye_midpoint_radius=(mask&4096u)?128u:0u;
    assert(optional_moe_scratch_bytes()==0); // Flags alone own no memory.
}
bool prepare() {
    return allocate_optional_hipblaslt_router_logits() && allocate_optional_moe_l2() &&
        allocate_optional_moe_compaction() && allocate_optional_moe_prepared_replay() &&
        allocate_optional_moe_shared_replay() && allocate_optional_shared_projection_hawkeye();
}
int main() {
    uint64_t cases=0,failures=0;
    for(unsigned blocks:{1u,1024u,16384u})for(unsigned mask=0;mask<8192u;++mask) {
        reset(mask,blocks);assert(prepare());
        assert(optional_moe_scratch_bytes()==requested_bytes);++cases;
        uint64_t independent=0;for(auto entry:allocated)independent+=entry.second;
        assert(independent==requested_bytes);
        // Each failure in every dense option family leaves the successful
        // prefix allocated. The counter must track that prefix immediately.
        if((mask&255u)==255u) {
            const unsigned allocations=calls;
            for(unsigned failure=1u;failure<=allocations;++failure) {
                reset(mask,blocks,failure);assert(!prepare() && calls==failure);
                assert(allocated.size()==failure-1u && optional_moe_scratch_bytes()==requested_bytes);
                ++failures;
            }
        }
    }
    // Current-style staged routed replay plus expert order and shared
    // prevalidation. This generated configuration is not a model measurement.
    constexpr unsigned illustrative=1u|2u|4u|8u|32u|256u|512u|1024u;
    reset(illustrative,16384u);assert(prepare());
    const auto illustrative_bytes=optional_moe_scratch_bytes();
    reset(0u,1024u);assert(allocated.empty());
    std::printf("{\"tokens\":%u,\"option_configurations\":%llu,\"injected_allocation_failures\":%llu,"
        "\"allocation_observations\":%llu,\"illustrative_optional_bytes\":%llu,"
        "\"actual_allocation_functions\":true,\"every_allocation_checked\":true,"
        "\"native_execution\":false}\n",kTokens,(unsigned long long)cases,
        (unsigned long long)failures,(unsigned long long)observations,(unsigned long long)illustrative_bytes);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-moe-scratch-') as tmp:
            for tokens in (1024, 8192):
                executable = str(Path(tmp) / f'check-{tokens}')
                compiled = subprocess.run([
                    os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-x', 'c++', '-', '-o', executable,
                ], input=source.replace('TOKEN_COUNT', str(tokens)), text=True, capture_output=True, timeout=30)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                result = subprocess.run([executable], text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('"every_allocation_checked":true', result.stdout)
                print(result.stdout.strip())


if __name__ == '__main__':
    unittest.main()
