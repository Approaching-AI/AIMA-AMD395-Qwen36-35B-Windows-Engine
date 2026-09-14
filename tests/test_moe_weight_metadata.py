"""Exercise actual immutable weight registration, invalidation and copy faults."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class MoeWeightMetadataTests(unittest.TestCase):
    def test_registration_is_atomic_and_addresses_require_explicit_reregistration(self):
        provider = (ROOT / 'native/providers/triton_moe/qrt_triton_moe_q8192_provider.cpp').read_text()
        declarations = '\n'.join(function(provider, name) + ';' for name in (
            'enum class MoeL2 :', 'struct MoeWeightMetadata {'))
        actual = '\n'.join(function(provider, name) for name in (
            'bool release_moe_weight_metadata()',
            'int copy_registered_moe_weight_metadata(',
            'int qrt_triton_moe_q8192_register_weight_metadata('))
        source = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
using hipStream_t=void*;
enum hipError_t { hipSuccess,hipErrorUnknown };
constexpr unsigned kTokens=8192,kExperts=2,kHidden=32,kIntermediate=16,kNativeThreads=256;
constexpr unsigned kMoePreparedWeightRows=64,kMoeWeightMetadataEntries=80;
constexpr int hipMemcpyDeviceToDevice=3;
struct dim3 { unsigned x; explicit dim3(unsigned n):x(n){} };
''' + declarations + r'''
struct State {
    bool prepared=true,full_v3_poisoned=false,scaled_l2=false;
    bool prepared_replay=false,prepared_replay_active=false,prevalidated_float_active=false;
    bool weight_metadata_ready=false;
    bool scaled_significand_fallback=false;
    unsigned sm121_moe_absolute_error_ppb=1000;
    std::array<MoeWeightMetadata,kMoeWeightMetadataEntries> weight_metadata{};
    std::array<float*,static_cast<size_t>(MoeL2::Count)> moe_l2{};
    uint32_t* prepared_replay_weight_rows=nullptr;
} g_state;
struct FullV3InFlightGuard { explicit FullV3InFlightGuard(bool wait) { assert(wait); } };
unsigned allocations=0,frees=0,scans=0,copies=0,syncs=0;
unsigned fail_allocation=0,fail_launch=0,fail_copy=0,fail_free=0;
bool fail_device_sync=false,fail_stream_sync=false,pending=false;
std::map<void*,size_t> live;
void set_error_text(const char*) {}
void set_error(const char*,hipError_t) {}
template<class T> bool allocate(T** output,size_t bytes,const char*) {
    if(++allocations==fail_allocation) return false;
    *output=static_cast<T*>(std::malloc(bytes));assert(*output);
    assert(live.emplace(*output,bytes).second);return true;
}
hipError_t hipFree(void* pointer) {
    assert(!pending && live.count(pointer));
    if(++frees==fail_free) return hipErrorUnknown;
    std::free(pointer);live.erase(pointer);return hipSuccess;
}
hipError_t hipDeviceSynchronize() {
    ++syncs;if(fail_device_sync)return hipErrorUnknown;
    pending=false;return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t stream) {
    assert(!stream);++syncs;if(fail_stream_sync)return hipErrorUnknown;
    pending=false;return hipSuccess;
}
hipError_t hipGetLastError() { return scans==fail_launch ? hipErrorUnknown : hipSuccess; }
template<bool ValidateOnly,bool ScaledFallback=false> void moe_bf16_row_l2_prepared_kernel(
    const uint16_t* input,float* output,std::nullptr_t,uint32_t* flags,
    unsigned rows,unsigned columns,unsigned first) {
    static_assert(ValidateOnly);
    assert(live.count(output) && live[output]==rows*8u && flags==reinterpret_cast<uint32_t*>(output+rows));
    assert(first<rows && (columns==kHidden || columns==kIntermediate));
    for(unsigned r=first;r<std::min(rows,first+4096u);++r) {
        // Arithmetic is a stand-in; native norm tests own FP64 equivalence.
        output[r]=float(input[0]+r);flags[r]=unsigned((input[0]+r)%3u!=0u)|(ScaledFallback?2u:0u);
    }
    ++scans;pending=true;
}
template<class Kernel,class... A> void launch(Kernel kernel,dim3 grid,dim3 block,
    int shared,std::nullptr_t,A... arguments) {
    assert(grid.x && grid.x<=4096 && block.x==kNativeThreads && !shared);
    kernel(arguments...);
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(...) launch(__VA_ARGS__)
hipError_t hipMemcpyAsync(void* to,const void* from,size_t bytes,int kind,hipStream_t stream) {
    assert(to && from && bytes==64u*4u && kind==hipMemcpyDeviceToDevice && stream==reinterpret_cast<void*>(17));
    bool owned=false;
    for(auto [base,size]:live) {
        const auto first=reinterpret_cast<uintptr_t>(base),p=reinterpret_cast<uintptr_t>(from);
        owned|=p>=first && p-first+bytes<=size;
    }
    assert(owned);++copies;
    if(copies==fail_copy)return hipErrorUnknown;
    std::memcpy(to,from,bytes);pending=true;return hipSuccess;
}
''' + actual + r'''
void reset() {
    fail_device_sync=fail_stream_sync=false;fail_free=0;
    assert(hipDeviceSynchronize()==hipSuccess && release_moe_weight_metadata());
    assert(live.empty());g_state=State{};
    allocations=frees=scans=copies=syncs=fail_allocation=fail_launch=fail_copy=0;
}
int main() {
    uint16_t a=3,b=7,c=13,d=19;
    const uint16_t* gate[]={&a,&c};const uint16_t* down[]={&b,&d};
    auto prepare=[&] { return qrt_triton_moe_q8192_register_weight_metadata(gate,down,2u); };
    std::array<float,66> norms;std::array<unsigned,66> flags;
    auto bind=[&] {
        norms.fill(-99);flags.fill(0xa5a5a5a5u);
        g_state.moe_l2[static_cast<size_t>(MoeL2::RoutedGateUp)]=norms.data()+1;
        g_state.moe_l2[static_cast<size_t>(MoeL2::RoutedDown)]=norms.data()+1;
        g_state.prepared_replay_weight_rows=flags.data()+1;
    };
    auto copy=[&](const uint16_t* src,MoeL2 surface=MoeL2::RoutedGateUp,unsigned rows=64,unsigned cols=32) {
        copies=0;return copy_registered_moe_weight_metadata(src,surface,rows,cols,reinterpret_cast<void*>(17));
    };
    for(unsigned fault=0;fault<7;++fault) {
        reset();
        if(fault==0)g_state.prepared=false;
        if(fault==1)g_state.scaled_l2=true;
        if(fault==2)g_state.prepared_replay=true;
        if(fault==3)g_state.sm121_moe_absolute_error_ppb=0;
        if(fault==4)g_state.full_v3_poisoned=true;
        if(fault==5)gate[1]=gate[0];
        if(fault==6)down[1]=nullptr;
        assert(!prepare() && !allocations && !scans && !syncs && !g_state.weight_metadata_ready);
        gate[1]=&c;down[1]=&d;
    }
    reset();
    assert(!qrt_triton_moe_q8192_register_weight_metadata(gate,down,41));
    assert(!qrt_triton_moe_q8192_register_weight_metadata(nullptr,down,2));
    assert(!qrt_triton_moe_q8192_register_weight_metadata(gate,nullptr,2));
    assert(!qrt_triton_moe_q8192_register_weight_metadata(gate,down,0));
    assert(!allocations && !syncs);
    for(unsigned fault=1;fault<=4;++fault) {
        reset();fail_allocation=fault;
        assert(!prepare() && !g_state.weight_metadata_ready && live.size()==fault-1u && copy(&a)==0);
    }
    for(unsigned fault=1;fault<=4;++fault) {
        reset();fail_launch=fault;
        assert(!prepare() && !g_state.weight_metadata_ready && g_state.full_v3_poisoned && pending);
        const auto owned=live.size();assert(qrt_triton_moe_q8192_register_weight_metadata(nullptr,nullptr,0)==1);
        assert(live.size()==owned && !frees && copy(&a)==0);
    }
    reset();fail_device_sync=true;
    assert(!prepare() && !allocations && g_state.full_v3_poisoned);
    reset();fail_stream_sync=true;
    assert(!prepare() && scans==1 && g_state.full_v3_poisoned && pending && !g_state.weight_metadata_ready);
    reset();assert(prepare()==1 && g_state.weight_metadata_ready && allocations==4 && scans==4);
    bind();g_state.prevalidated_float_active=true;
    assert(copy(&a)==1 && copies==2);
    for(unsigned r=0;r<64;++r)assert(norms[r+1]==a+r && flags[r+1]==unsigned((a+r)%3u!=0));
    assert(norms.front()==-99 && norms.back()==-99 && flags.front()==0xa5a5a5a5u && flags.back()==0xa5a5a5a5u);
    assert(copy(&d,MoeL2::RoutedDown,64,16)==1 && copies==2 && norms[1]==d);
    assert(copy(&a,MoeL2::Input)==0 && !copies);
    assert(copy(&a,MoeL2::RoutedDown)==0 && !copies);
    assert(copy(&a,MoeL2::RoutedGateUp,63)==0 && !copies);
    assert(copy(&a,MoeL2::RoutedGateUp,64,16)==0 && !copies);
    uint16_t unregistered=3;assert(copy(&unregistered)==0 && !copies);
    g_state.prepared_replay_active=true;assert(copy(&a)==0 && !copies);g_state.prepared_replay_active=false;
    g_state.prevalidated_float_active=false;assert(copy(&a)==1 && copies==1);
    g_state.prevalidated_float_active=true;
    g_state.prepared_replay_weight_rows=nullptr;assert(copy(&a)==-1 && !copies);bind();
    for(unsigned fault=1;fault<=2;++fault) {
        fail_copy=fault;assert(copy(&a)==-1 && copies==fault);
    }
    fail_copy=0;
    // Invalidating with queued copies keeps their allocation alive but makes
    // every source address miss immediately. Reusing an address requires a
    // complete explicit scan, including the changed content.
    assert(pending);const auto old_live=live.size();
    assert(qrt_triton_moe_q8192_register_weight_metadata(nullptr,nullptr,0)==1);
    assert(!g_state.weight_metadata_ready && live.size()==old_live && !frees && copy(&a)==0);
    a=61;assert(prepare()==1 && frees==4 && live.size()==4);bind();
    assert(copy(&a)==1 && norms[1]==61);
    assert(hipDeviceSynchronize()==hipSuccess);
    fail_free=frees+1u;assert(!release_moe_weight_metadata() && live.size()==4 && !g_state.weight_metadata_ready);
    fail_free=0;assert(release_moe_weight_metadata() && live.empty());
    reset();g_state.scaled_significand_fallback=true;
    assert(prepare()==1 && allocations==4 && scans==4);bind();g_state.prevalidated_float_active=true;
    assert(copy(&a)==1 && copies==2);
    for(unsigned r=0;r<64;++r)assert(norms[r+1]==a+r && flags[r+1]==(unsigned((a+r)%3u!=0)|2u));
    assert(flags.front()==0xa5a5a5a5u && flags.back()==0xa5a5a5a5u);
    reset();
    std::puts("registered_metadata_faults_and_address_reuse=pass");
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-moe-metadata-') as tmp:
            executable = str(Path(tmp) / 'check')
            result = subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra',
                '-Werror', '-fsanitize=address,undefined', '-x', 'c++', '-', '-o', executable],
                input=source, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([executable], text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('registered_metadata_faults_and_address_reuse=pass', result.stdout)


if __name__ == '__main__':
    unittest.main()
