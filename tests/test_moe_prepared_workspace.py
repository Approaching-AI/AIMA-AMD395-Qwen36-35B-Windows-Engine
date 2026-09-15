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
        allocations += '\n' + function(s, 'constexpr bool shared_replay_surface(')
        allocations += '\n' + function(s, 'constexpr uint32_t shared_staged_columns(')
        allocations += '\n' + function(s, 'bool allocate_optional_moe_shared_replay()')
        scans = function(s, 'bool prepare_moe_staged_half(')
        scans += '\n' + function(s, 'bool launch_moe_l2(')
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
constexpr unsigned kNativeThreads=256,kHidden=2048,kIntermediate=512;
enum class MoeL2 { Input, Router, SharedInput, SharedGate, SharedUp, SharedActivated,
                  SharedDown, RoutedGateUp, RoutedActivated, RoutedDown, Count };
constexpr size_t kMoeL2Rows[]={8192,256,8192,512,512,8192,2048,262144,65536,524288};
constexpr size_t kMoePreparedWeightElements=536870912, kMoePreparedInputElements=33554432;
constexpr size_t kMoePreparedWeightRows=524288, kMoePreparedInputRows=65536;
namespace qrt_sm121_staged_half_projection { struct Row { uint32_t pairs[8],control; }; }
namespace qrt_sm121_scaled_half_projection { constexpr int prepare_rows=5; }
struct State {
    bool prepared_replay=false,prepared_replay_active=false,scaled_l2=false;
    bool prevalidated_float=false,prevalidated_float_active=false;
    bool staged_half_replay=false,staged_half_replay_active=false;
    bool scaled_significand_fallback=false;
    bool shared_prevalidated_float=false,shared_prevalidated_float_active=false;
    bool shared_staged_half=false,shared_staged_half_active=false;
    std::array<uint16_t*,10> shared_staged_operands{};
    std::array<uint32_t*,10> shared_replay_rows{};
    unsigned sm121_moe_absolute_error_ppb=1000;
    uint16_t *prepared_replay_weights=nullptr,*prepared_replay_inputs=nullptr;
    uint32_t *prepared_replay_weight_rows=nullptr,*prepared_replay_input_rows=nullptr;
    std::array<float*,10> moe_l2{};
} g_state;
std::vector<size_t> sizes;
unsigned fail_allocation=0,free_calls=0,launches=0,fail_launch=0;
unsigned staged_launches=0,cache_calls=0;
bool fail_staged=false,last_staged=false;
int cache_result=0;
bool drain_ok=true;
template<class T> bool allocate(T** p,size_t bytes,const char*) {
    sizes.push_back(bytes);
    if(sizes.size()==fail_allocation)return false;
    *p=reinterpret_cast<T*>(uintptr_t(sizes.size()*4096));return true;
}
int hipFree(void* p) { assert(p);++free_calls;return 0; }
bool release_full_v3_execution_state() { return drain_ok; }
// The separate registration test owns immutable weight cache transitions.
bool release_moe_weight_metadata() { return true; }
int copy_registered_moe_weight_metadata(const uint16_t*,MoeL2,unsigned,unsigned,hipStream_t) {
    ++cache_calls;return cache_result;
}
void set_error_text(const char*) {}
void set_error(const char*,hipError_t) {}
constexpr int moe_bf16_row_l2_kernel=0,moe_bf16_scaled_row_l2_kernel=1;
template<bool ValidateOnly,bool ScaledFallback=false> constexpr int moe_bf16_row_l2_prepared_kernel=ScaledFallback?4:ValidateOnly?3:2;
#define HIP_KERNEL_NAME(...) __VA_ARGS__
int expected_kernel=0;
unsigned expected_rows=0,expected_columns=0,covered=0;
uint32_t* expected_flags=nullptr;
uint16_t value;
hipStream_t stream=reinterpret_cast<void*>(uintptr_t(101));
qrt_sm121_staged_half_projection::Row* expected_staged=nullptr;
void launch(int kernel,dim3 grid,dim3 block,int shared,hipStream_t q,
            const uint16_t* input,qrt_sm121_staged_half_projection::Row* output,unsigned rows,unsigned columns) {
    assert(kernel==5&&input==&value&&output==expected_staged);
    assert(rows==expected_rows&&columns==expected_columns);
    assert(grid.x==(size_t(rows)*(columns/16)+255)/256&&block.x==256&&shared==0&&q==stream);
    ++staged_launches;last_staged=true;
}
template<class... Args> void launch(int kernel,dim3 grid,dim3 block,int shared,hipStream_t q,Args... args) {
    assert(kernel==expected_kernel&&grid.x&&grid.x<=4096&&block.x==256&&shared==0&&q==stream);
    auto a=std::make_tuple(args...);constexpr size_t n=sizeof...(args);
    assert(std::get<0>(a)==&value&&std::get<n-3>(a)==expected_rows&&std::get<n-2>(a)==expected_columns);
    assert(std::get<n-1>(a)==covered);covered+=grid.x;++launches;last_staged=false;
    if constexpr(n==7) {
        if(kernel==3||kernel==4)assert(std::get<2>(a)==nullptr);
        else assert(std::get<2>(a)==g_state.prepared_replay_inputs||std::get<2>(a)==g_state.prepared_replay_weights);
        assert(std::get<3>(a)==expected_flags);
    }
}
#define hipLaunchKernelGGL(...) launch(__VA_ARGS__)
hipError_t hipGetLastError() {
    return (last_staged?fail_staged:(launches==fail_launch))?hipErrorUnknown:hipSuccess;
}
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
        launches=covered=staged_launches=cache_calls=0;last_staged=false;
        expected_rows=rows;expected_columns=columns;
        expected_staged=reinterpret_cast<qrt_sm121_staged_half_projection::Row*>(
            g_state.shared_staged_half_active&&shared_replay_surface(surface)
                ? g_state.shared_staged_operands[size_t(surface)]
                : surface==MoeL2::Input||surface==MoeL2::RoutedActivated
                    ?g_state.prepared_replay_inputs:g_state.prepared_replay_weights);
        expected_flags=g_state.shared_prevalidated_float_active&&shared_replay_surface(surface)
            ? g_state.shared_replay_rows[size_t(surface)]
            : (surface==MoeL2::Input||surface==MoeL2::RoutedActivated)
                ? g_state.prepared_replay_input_rows : g_state.prepared_replay_weight_rows;
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
    const std::vector<size_t> shared_bytes{32768,2048,2048,32768,8192};
    for(unsigned fault=1;fault<=6;++fault) {
        g_state=State{};g_state.shared_prevalidated_float=true;sizes.clear();free_calls=0;fail_allocation=fault;
        assert(allocate_optional_moe_shared_replay()==(fault==6));
        assert(sizes==std::vector<size_t>(shared_bytes.begin(),shared_bytes.begin()+(fault==6?5:fault)));
        assert(!g_state.prepared_replay_input_rows&&!g_state.prepared_replay_weight_rows);
        drain_ok=false;assert(!release_prefix()&&free_calls==0);
        drain_ok=true;assert(release_prefix()&&free_calls==(fault==6?5:fault-1));
    }
    g_state.prevalidated_float=true;g_state.prevalidated_float_active=true;
    g_state.shared_prevalidated_float=true;g_state.shared_prevalidated_float_active=true;
    sizes.clear();fail_allocation=0;assert(allocate_optional_moe_prepared_replay()&&allocate_optional_moe_shared_replay());
    expected_kernel=3;
    for(auto surface : {MoeL2::SharedInput,MoeL2::SharedGate,MoeL2::SharedUp,MoeL2::SharedActivated,MoeL2::SharedDown}) {
        const unsigned rows=unsigned(kMoeL2Rows[size_t(surface)]);
        const unsigned columns=surface==MoeL2::SharedActivated||surface==MoeL2::SharedDown?512:2048;
        auto* flags=g_state.shared_replay_rows[size_t(surface)];
        assert(flags&&flags!=g_state.prepared_replay_input_rows&&flags!=g_state.prepared_replay_weight_rows);
        assert(run(surface,rows,columns)&&covered==rows);
        g_state.shared_replay_rows[size_t(surface)]=nullptr;
        assert(!run(surface,rows,columns)&&launches==0);
        g_state.shared_replay_rows[size_t(surface)]=flags;
        assert(!run(surface,rows+1,columns)&&launches==0);
    }
    assert(run(MoeL2::Input,8192,2048)&&covered==8192);
    assert(run(MoeL2::RoutedDown,524288,512)&&covered==524288);
    g_state.scaled_significand_fallback=true;expected_kernel=4;
    assert(run(MoeL2::Input,8192,2048)&&covered==8192);
    assert(run(MoeL2::RoutedDown,524288,512)&&covered==524288);
    assert(run(MoeL2::SharedInput,8192,2048)&&covered==8192);
    fail_launch=2;assert(!run(MoeL2::SharedInput,8192,2048)&&launches==2);
    fail_launch=0;g_state.scaled_significand_fallback=false;expected_kernel=3;
    g_state.scaled_l2=true;assert(!run(MoeL2::SharedInput,8192,2048)&&launches==0);g_state.scaled_l2=false;
    fail_launch=2;assert(!run(MoeL2::SharedInput,8192,2048)&&launches==2);
    fail_launch=0;free_calls=0;assert(release_prefix()&&free_calls==7);
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
    const std::vector<size_t> staged_bytes{1207959552,75497472,2097152,262144};
    for(unsigned fault=1;fault<=5;++fault) {
        g_state=State{};g_state.prevalidated_float=true;g_state.staged_half_replay=true;
        sizes.clear();free_calls=0;fail_allocation=fault;
        assert(allocate_optional_moe_prepared_replay()==(fault==5));
        assert(sizes==std::vector<size_t>(staged_bytes.begin(),staged_bytes.begin()+(fault==5?4:fault)));
        drain_ok=false;assert(!release_prefix()&&free_calls==0);
        drain_ok=true;assert(release_prefix()&&free_calls==(fault==5?4:fault-1));
    }
    g_state.prevalidated_float=g_state.prevalidated_float_active=true;
    g_state.staged_half_replay=g_state.staged_half_replay_active=true;
    fail_allocation=0;sizes.clear();assert(allocate_optional_moe_prepared_replay());
    expected_kernel=3;
    for(auto surface:{MoeL2::Input,MoeL2::RoutedGateUp,MoeL2::RoutedActivated,MoeL2::RoutedDown}) {
        const unsigned rows=unsigned(kMoeL2Rows[size_t(surface)]);
        const unsigned columns=surface==MoeL2::RoutedActivated||surface==MoeL2::RoutedDown?512:2048;
        assert(run(surface,rows,columns)&&staged_launches==1&&covered==rows&&cache_calls==1);
        // Cached norms never suppress a fresh operand view. Also refresh when
        // the caller revisits the same surface on a later layer.
        for(unsigned repeat=0;repeat<2;++repeat) {
            cache_result=1;assert(run(surface,rows,columns)&&staged_launches==1&&launches==0&&cache_calls==1);
        }
        cache_result=-1;assert(!run(surface,rows,columns)&&staged_launches==1&&launches==0);
        cache_result=0;fail_staged=true;
        assert(!run(surface,rows,columns)&&staged_launches==1&&launches==0&&cache_calls==0);
        fail_staged=false;
        assert(!run(surface,rows,columns+1)&&staged_launches==0&&launches==0&&cache_calls==0);
        assert(!run(surface,rows+1,columns)&&staged_launches==0&&launches==0);
    }
    expected_kernel=0;
    for(auto surface:{MoeL2::Router,MoeL2::SharedInput,MoeL2::SharedGate,MoeL2::SharedUp,
                      MoeL2::SharedActivated,MoeL2::SharedDown}) {
        const unsigned rows=unsigned(kMoeL2Rows[size_t(surface)]);
        assert(run(surface,rows,2048)&&covered==rows&&staged_launches==0);
    }
    assert(!run(MoeL2::Input,0,2048)&&staged_launches==0);
    assert(!run(MoeL2::Input,8192,0)&&staged_launches==0);
    assert(!run(MoeL2::Input,8192,8192)&&staged_launches==0);
    assert(!launch_moe_l2(nullptr,MoeL2::Input,8192,2048,stream));
    auto* input_storage=g_state.prepared_replay_inputs;g_state.prepared_replay_inputs=nullptr;
    assert(!run(MoeL2::Input,8192,2048)&&staged_launches==0&&cache_calls==0);
    g_state.prepared_replay_inputs=input_storage;
    auto* weight_storage=g_state.prepared_replay_weights;g_state.prepared_replay_weights=nullptr;
    assert(!run(MoeL2::RoutedDown,524288,512)&&staged_launches==0&&cache_calls==0);
    g_state.prepared_replay_weights=weight_storage;
    g_state.staged_half_replay_active=false;expected_kernel=3;
    assert(run(MoeL2::Input,8192,2048)&&covered==8192&&staged_launches==0);
    free_calls=0;drain_ok=false;assert(!release_prefix()&&free_calls==0);
    drain_ok=true;assert(release_prefix()&&free_calls==4);

    const std::vector<size_t> shared_staged_bytes{32768,37748736,2048,2359296,2048,2359296,32768,9437184,8192,2359296};
    for(unsigned fault=1;fault<=11;++fault){
        g_state=State{};g_state.shared_prevalidated_float=true;g_state.shared_staged_half=true;
        sizes.clear();free_calls=0;fail_allocation=fault;
        assert(allocate_optional_moe_shared_replay()==(fault==11));
        assert(sizes==std::vector<size_t>(shared_staged_bytes.begin(),shared_staged_bytes.begin()+(fault==11?10:fault)));
        drain_ok=false;assert(!release_prefix()&&free_calls==0);
        drain_ok=true;assert(release_prefix()&&free_calls==(fault==11?10:fault-1));
    }
    g_state.shared_prevalidated_float=g_state.shared_prevalidated_float_active=true;
    g_state.shared_staged_half=g_state.shared_staged_half_active=true;
    sizes.clear();fail_allocation=0;assert(allocate_optional_moe_shared_replay());expected_kernel=3;
    assert(!g_state.prepared_replay_inputs&&!g_state.prepared_replay_weights);
    for(auto surface:{MoeL2::SharedInput,MoeL2::SharedGate,MoeL2::SharedUp,MoeL2::SharedActivated,MoeL2::SharedDown}){
        const unsigned rows=unsigned(kMoeL2Rows[size_t(surface)]),columns=shared_staged_columns(surface);
        for(unsigned repeat=0;repeat<2;++repeat)assert(run(surface,rows,columns)&&staged_launches==1&&covered==rows);
        assert(!run(surface,rows,columns==512?2048:512)&&staged_launches==0&&launches==0);
        assert(!run(surface,rows+1,columns)&&staged_launches==0&&launches==0);
        assert(!run(surface,0,columns)&&staged_launches==0&&launches==0);
        auto* saved_operand=g_state.shared_staged_operands[size_t(surface)];g_state.shared_staged_operands[size_t(surface)]=nullptr;
        assert(!run(surface,rows,columns)&&staged_launches==0&&launches==0);
        g_state.shared_staged_operands[size_t(surface)]=saved_operand;
        fail_staged=true;assert(!run(surface,rows,columns)&&staged_launches==1&&launches==0&&cache_calls==0);fail_staged=false;
        g_state.shared_prevalidated_float_active=false;assert(!run(surface,rows,columns)&&staged_launches==0);g_state.shared_prevalidated_float_active=true;
        g_state.scaled_l2=true;assert(!run(surface,rows,columns)&&staged_launches==0);g_state.scaled_l2=false;
    }
    g_state.shared_staged_half_active=false;
    assert(run(MoeL2::SharedInput,8192,2048)&&staged_launches==0&&covered==8192);
    free_calls=0;drain_ok=false;assert(!release_prefix()&&free_calls==0);
    drain_ok=true;assert(release_prefix()&&free_calls==10);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-moe-prepared-') as tmp:
            exe = str(Path(tmp) / 'workspace')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-x', 'c++', '-', '-o', exe], input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=5)


if __name__ == '__main__':
    unittest.main()
