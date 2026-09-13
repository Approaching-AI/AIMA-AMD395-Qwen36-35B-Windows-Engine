"""Exercise actual prepared allocation, scan routing, and cleanup boundaries."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class MoePreparedWorkspaceTests(unittest.TestCase):
    def test_real_workspace_faults_and_scan_routing(self):
        s = (ROOT / 'native/providers/triton_moe/qrt_triton_moe_q8192_provider.cpp').read_text()
        allocations = function(s, 'bool allocate_optional_moe_prepared_replay()')
        scans = function(s, 'bool launch_moe_l2(')
        # Extract the real ordered release prefix: execution-state drain must
        # succeed before any prepared storage can be freed.
        release = s.split('bool release_state() {', 1)[1].split('    release_matrix_plan(', 1)[0]
        source = r'''
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
using hipStream_t=void*;
struct dim3 { unsigned x; explicit dim3(unsigned v):x(v){} };
constexpr unsigned kNativeThreads=256;
enum class MoeL2 { Input, Router, SharedInput, SharedGate, SharedUp, SharedActivated,
                  SharedDown, RoutedGateUp, RoutedActivated, RoutedDown, Count };
constexpr size_t kMoeL2Rows[]={8192,256,8192,512,512,8192,2048,262144,65536,524288};
constexpr size_t kMoePreparedWeightElements=536870912, kMoePreparedInputElements=33554432;
constexpr size_t kMoePreparedWeightRows=524288, kMoePreparedInputRows=65536;
struct State {
    bool prepared_replay=false,prepared_replay_active=false,scaled_l2=false;
    bool prevalidated_float=false,prevalidated_float_active=false;
    unsigned sm121_moe_absolute_error_ppb=1000;
    uint16_t *prepared_replay_weights=nullptr,*prepared_replay_inputs=nullptr;
    uint32_t *prepared_replay_weight_rows=nullptr,*prepared_replay_input_rows=nullptr;
    std::array<float*,10> moe_l2{};
} g_state;
std::vector<size_t> sizes;
unsigned fail_allocation=0,free_calls=0,launches=0,fail_launch=0;
bool drain_ok=true;
template<class T> bool allocate(T** p,size_t bytes,const char*) {
    sizes.push_back(bytes);
    if(sizes.size()==fail_allocation)return false;
    *p=reinterpret_cast<T*>(uintptr_t(sizes.size()*4096));return true;
}
int hipFree(void* p) { assert(p);++free_calls;return 0; }
bool release_full_v3_execution_state() { return drain_ok; }
void set_error_text(const char*) {}
void set_error(const char*,hipError_t) {}
constexpr int moe_bf16_row_l2_kernel=0,moe_bf16_scaled_row_l2_kernel=1;
template<bool ValidateOnly> constexpr int moe_bf16_row_l2_prepared_kernel=ValidateOnly?3:2;
#define HIP_KERNEL_NAME(...) __VA_ARGS__
int expected_kernel=0;
unsigned expected_rows=0,expected_columns=0,covered=0;
uint16_t value;
hipStream_t stream=reinterpret_cast<void*>(uintptr_t(101));
template<class... Args> void launch(int kernel,dim3 grid,dim3 block,int shared,hipStream_t q,Args... args) {
    assert(kernel==expected_kernel&&grid.x&&grid.x<=4096&&block.x==256&&shared==0&&q==stream);
    auto a=std::make_tuple(args...);constexpr size_t n=sizeof...(args);
    assert(std::get<0>(a)==&value&&std::get<n-3>(a)==expected_rows&&std::get<n-2>(a)==expected_columns);
    assert(std::get<n-1>(a)==covered);covered+=grid.x;++launches;
    if constexpr(n==7) {
        if(kernel==3)assert(std::get<2>(a)==nullptr);
        assert(std::get<2>(a)==g_state.prepared_replay_inputs||std::get<2>(a)==g_state.prepared_replay_weights);
        assert(std::get<3>(a)==g_state.prepared_replay_input_rows||std::get<3>(a)==g_state.prepared_replay_weight_rows);
    }
}
#define hipLaunchKernelGGL(...) launch(__VA_ARGS__)
hipError_t hipGetLastError() { return launches==fail_launch?hipErrorUnknown:hipSuccess; }
''' + allocations + scans + '\nbool release_prefix() {' + release + r'''
    g_state=State{};return true;
}
int main() {
    assert(allocate_optional_moe_prepared_replay()&&sizes.empty());
    const std::vector<size_t> wanted{1073741824,67108864,2097152,262144};
    for(unsigned fault=1;fault<=5;++fault) {
        g_state=State{};g_state.prepared_replay=true;sizes.clear();free_calls=0;fail_allocation=fault;
        const bool success=allocate_optional_moe_prepared_replay();assert(success==(fault==5));
        assert(sizes==std::vector<size_t>(wanted.begin(),wanted.begin()+(fault==5?4:fault)));
        drain_ok=false;assert(!release_prefix()&&free_calls==0);
        drain_ok=true;assert(release_prefix()&&free_calls==(fault==5?4:fault-1));
    }
    g_state.prepared_replay=true;fail_allocation=0;sizes.clear();assert(allocate_optional_moe_prepared_replay());
    auto run=[&](MoeL2 surface,unsigned rows,unsigned columns) {
        launches=covered=0;expected_rows=rows;expected_columns=columns;
        return launch_moe_l2(&value,surface,rows,columns,stream);
    };
    expected_kernel=0;assert(run(MoeL2::Input,8192,2048)&&covered==8192&&launches==2);
    g_state.prepared_replay_active=true;expected_kernel=2;
    assert(run(MoeL2::Input,8192,2048)&&covered==8192);
    assert(run(MoeL2::RoutedGateUp,262144,2048)&&covered==262144&&launches==64);
    assert(run(MoeL2::RoutedActivated,65536,512)&&covered==65536);
    assert(run(MoeL2::RoutedDown,524288,512)&&covered==524288&&launches==128);
    expected_kernel=0;assert(run(MoeL2::SharedInput,8192,2048)&&launches==2);
    g_state.scaled_l2=true;assert(!run(MoeL2::Input,8192,2048)&&launches==0);g_state.scaled_l2=false;
    auto* saved=g_state.prepared_replay_input_rows;g_state.prepared_replay_input_rows=nullptr;
    assert(!run(MoeL2::Input,8192,2048)&&launches==0);g_state.prepared_replay_input_rows=saved;
    assert(!run(MoeL2::Input,8192,8192)&&launches==0);
    expected_kernel=2;fail_launch=3;assert(!run(MoeL2::RoutedGateUp,262144,2048)&&launches==3);
    fail_launch=0;assert(release_prefix());
    const std::vector<size_t> flag_bytes{2097152,262144};
    for(unsigned fault=1;fault<=3;++fault) {
        g_state=State{};g_state.prevalidated_float=true;sizes.clear();free_calls=0;fail_allocation=fault;
        assert(allocate_optional_moe_prepared_replay()==(fault==3));
        assert(sizes==std::vector<size_t>(flag_bytes.begin(),flag_bytes.begin()+(fault==3?2:fault)));
        assert(!g_state.prepared_replay_inputs&&!g_state.prepared_replay_weights);
        drain_ok=false;assert(!release_prefix()&&free_calls==0);
        drain_ok=true;assert(release_prefix()&&free_calls==(fault==3?2:fault-1));
    }
    g_state.prevalidated_float=true;g_state.prevalidated_float_active=true;
    fail_allocation=0;sizes.clear();assert(allocate_optional_moe_prepared_replay());
    expected_kernel=3;
    assert(run(MoeL2::Input,8192,2048)&&covered==8192&&launches==2);
    assert(run(MoeL2::RoutedGateUp,262144,2048)&&covered==262144&&launches==64);
    assert(run(MoeL2::RoutedActivated,65536,512)&&covered==65536);
    assert(run(MoeL2::RoutedDown,524288,512)&&covered==524288&&launches==128);
    expected_kernel=0;assert(run(MoeL2::SharedInput,8192,2048)&&launches==2);
    g_state.scaled_l2=true;assert(!run(MoeL2::Input,8192,2048)&&launches==0);g_state.scaled_l2=false;
    saved=g_state.prepared_replay_input_rows;g_state.prepared_replay_input_rows=nullptr;
    assert(!run(MoeL2::Input,8192,2048)&&launches==0);g_state.prepared_replay_input_rows=saved;
    assert(!run(MoeL2::Input,8192,8192)&&launches==0);
    expected_kernel=3;fail_launch=3;assert(!run(MoeL2::RoutedGateUp,262144,2048)&&launches==3);
    fail_launch=0;assert(release_prefix());
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-moe-prepared-') as tmp:
            exe = str(Path(tmp) / 'workspace')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-x', 'c++', '-', '-o', exe], input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=5)


if __name__ == '__main__':
    unittest.main()
