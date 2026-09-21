#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include "../native/providers/gdn/sm121_q1_gdn.h"

namespace {
void check(hipError_t code) {
    if (code != hipSuccess) throw std::runtime_error(hipGetErrorString(code));
}
template<class T> std::vector<T> read(const char* path, size_t count) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if (!file || file.tellg()!=static_cast<std::streamoff>(count*sizeof(T))) throw std::runtime_error(path);
    std::vector<T> values(count);file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()),count*sizeof(T));
    if (!file) throw std::runtime_error("short read");
    return values;
}
struct Scratch {
    std::vector<void*> pointers;
    ~Scratch() { for (auto p:pointers) (void)hipFree(p); }
    template<class T> T* upload(const std::vector<T>& values) {
        T* p=nullptr;check(hipMalloc(reinterpret_cast<void**>(&p),values.size()*sizeof(T)));
        pointers.push_back(p);check(hipMemcpy(p,values.data(),values.size()*sizeof(T),hipMemcpyHostToDevice));
        return p;
    }
};
template<class T> std::vector<T> download(const T* p,size_t count) {
    std::vector<T> result(count);check(hipMemcpy(result.data(),p,count*sizeof(T),hipMemcpyDeviceToHost));
    return result;
}
template<class T> void immutable(const T* p,const std::vector<T>& expected) {
    const auto actual=download(p,expected.size());
    if (std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(T)))
        throw std::runtime_error("immutable input changed");
}
void finish() {
    hipEvent_t done=nullptr;check(hipEventCreateWithFlags(&done,hipEventDisableTiming));
    check(hipEventRecord(done,nullptr));
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    for (;;) {
        const auto status=hipEventQuery(done);
        if (status==hipSuccess) break;
        if (status!=hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now()>=deadline) throw std::runtime_error("convolution completion timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(hipEventDestroy(done));
}
template<class T> T element(uint16_t bits) {
    if constexpr(std::is_same<T,uint16_t>::value) return bits;
    else return qrt_sm121_q1::widen(bits);
}
template<class T> uint16_t encoded(T value) {
    if constexpr(std::is_same<T,uint16_t>::value) return value;
    else return qrt_sm121_q1::bf16(value);
}
template<class T> bool replay(const std::vector<uint16_t>& before,const std::vector<uint16_t>& after,
    const std::vector<uint16_t>& expected,const std::vector<float>& current,
    const float* dc,const uint16_t* dw,const unsigned char* ds,size_t position) {
    constexpr size_t guard=16;
    const T sentinel=element<T>(0x4a35u);
    const float marker=qrt_sm121_q1::widen(0x4a51u);
    std::vector<T> ring(32768+2*guard,sentinel);
    std::vector<float> output(8192+2*guard,marker);
    for (size_t feature=0;feature<8192;++feature)
        for (size_t tap=0;tap<3;++tap)
            ring[guard+((position-3+tap)%4)*8192+feature]=element<T>(before[feature*4+tap]);
    const auto original_ring=ring;
    Scratch scratch;auto dr=scratch.upload(ring);auto dout=scratch.upload(output);
    // One full out-of-range block exercises the kernel's feature bound.
    hipLaunchKernelGGL((qrt_sm121_q1::convolution<T>),dim3(33),dim3(256),0,nullptr,
        dc,dr+guard,dw,dout+guard,position,ds);
    check(hipGetLastError());finish();
    ring=download(dr,ring.size());output=download(dout,output.size());
    size_t output_bad=0,carrier_bad=0,history_bad=0,stored_bad=0,unowned_bad=0,guard_bad=0;
    for (size_t feature=0;feature<8192;++feature) {
        const float value=output[guard+feature];
        output_bad += qrt_sm121_q1::bf16(value)!=expected[feature];
        carrier_bad += qrt_sm121_exp2::bits(value)!=(uint32_t(expected[feature])<<16u);
        for (size_t tap=0;tap<3;++tap)
            history_bad += encoded(ring[guard+((position-2+tap)%4)*8192+feature])!=after[feature*4+tap];
        for (size_t slot=0;slot<4;++slot) {
            const auto index=guard+slot*8192+feature;
            if (slot!=position%4) unowned_bad += ring[index]!=original_ring[index];
            else if constexpr(std::is_same<T,float>::value)
                stored_bad += qrt_sm121_exp2::bits(ring[index])!=qrt_sm121_exp2::bits(current[feature]);
            else stored_bad += ring[index]!=qrt_sm121_q1::bf16(current[feature]);
        }
        if (before[feature*4+3]!=after[feature*4+3]) throw std::runtime_error("reference padding changed");
    }
    for (size_t i=0;i<guard;++i) {
        guard_bad += ring[i]!=sentinel;guard_bad += ring[guard+32768+i]!=sentinel;
        guard_bad += qrt_sm121_exp2::bits(output[i])!=qrt_sm121_exp2::bits(marker);
        guard_bad += qrt_sm121_exp2::bits(output[guard+8192+i])!=qrt_sm121_exp2::bits(marker);
    }
    immutable(dc,current);
    const bool passed=!(output_bad || carrier_bad || history_bad || stored_bad || unowned_bad || guard_bad);
    std::cout<<"{\"kind\":\"original_q1_convolution_native_replay\",\"ring_storage\":\""
        <<(std::is_same<T,uint16_t>::value?"bf16":"f32")<<"\",\"position\":"<<position
        <<",\"output_cells\":8192,\"active_history_cells\":24576,\"output_bf16_mismatches\":"<<output_bad
        <<",\"output_f32_carrier_bit_mismatches\":"<<carrier_bad<<",\"active_history_bf16_mismatches\":"<<history_bad
        <<",\"stored_current_mismatches\":"<<stored_bad<<",\"unowned_ring_mismatches\":"<<unowned_bad
        <<",\"guard_mismatches\":"<<guard_bad<<",\"reference_padding_unchanged\":true,\"current_input_immutable\":true"
        <<",\"out_of_range_block_checked\":true,\"passed\":"<<(passed?"true":"false")<<",\"inference_acceptance\":false}\n";
    return passed;
}
} // namespace

int main(int argc,char** argv) try {
    if (argc!=8) throw std::runtime_error("qkv before after weights silu expected-output position");
    const auto qkv=read<uint16_t>(argv[1],8192),before=read<uint16_t>(argv[2],32768),after=read<uint16_t>(argv[3],32768);
    const auto weights=read<uint16_t>(argv[4],32768),expected=read<uint16_t>(argv[6],8192);
    const auto silu=read<unsigned char>(argv[5],qrt_sm121_silu::table_bytes);
    const size_t position=std::stoull(argv[7]);
    if (position<3 || position>1048576 || !qrt_sm121_silu::valid_layout(silu.data(),silu.size()))
        throw std::runtime_error("position/table layout");
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0));
    if (std::strncmp(properties.gcnArchName,"gfx1151",7)) throw std::runtime_error("requires gfx1151");
    std::vector<float> current(qkv.size());
    for (size_t i=0;i<qkv.size();++i) current[i]=qrt_sm121_q1::widen(qkv[i]);
    Scratch scratch;const auto dc=scratch.upload(current);const auto dw=scratch.upload(weights);const auto ds=scratch.upload(silu);
    // Expected output and history-after remain exclusively on the host.
    const bool bf16=replay<uint16_t>(before,after,expected,current,dc,dw,ds,position);
    // Check each invocation before another kernel could mask a transient
    // mutation by restoring the same input bytes.
    immutable(dc,current);immutable(dw,weights);immutable(ds,silu);
    const bool f32=replay<float>(before,after,expected,current,dc,dw,ds,position);
    immutable(dc,current);immutable(dw,weights);immutable(ds,silu);
    std::cout<<"{\"kind\":\"original_q1_convolution_native_inputs\",\"immutable_inputs\":true,\"reference_output_is_compute_input\":false,\"inference_acceptance\":false}\n";
    return bf16 && f32 ? 0 : 1;
} catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 2; }
