// Standalone diagnosis of an already completed failed output stage. Captured
// native values are comparison data here, never model inference inputs.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <hip/hip_runtime.h>
#include "blackwell_cooperative.h"
#include "blackwell_state.h"
#include "blackwell_wu_output.h"
#include "completion_guard.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t guard_bytes = 256u;
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void setting(const char* name, const char* value) {
#ifdef _WIN32
    require(_putenv_s(name, value) == 0, "environment setup failed");
#else
    require(setenv(name, value, 1) == 0, "environment setup failed");
#endif
}
std::vector<unsigned char> load(const std::string& root, const char* name, size_t bytes) {
    std::ifstream file(root + "/" + name, std::ios::binary | std::ios::ate);
    require(file && file.tellg() == std::streamoff(bytes), "capture size mismatch");
    std::vector<unsigned char> data(bytes);
    file.seekg(0);file.read(reinterpret_cast<char*>(data.data()), std::streamsize(bytes));
    require(bool(file), "capture read failed");return data;
}
struct Buffer {
    void* device = nullptr;
    std::vector<unsigned char> original;
    explicit Buffer(const std::vector<unsigned char>& data) : original(data.size() + 2u * guard_bytes, 0xa5u) {
        std::copy(data.begin(), data.end(), original.begin() + guard_bytes);
        check(hipMalloc(&device, original.size()));
        try { check(hipMemcpy(device, original.data(), original.size(), hipMemcpyHostToDevice)); }
        catch (...) { (void)hipFree(device);device=nullptr;throw; }
    }
    Buffer(const Buffer&) = delete;
    ~Buffer() { if (device) (void)hipFree(device); }
    template<class T> T* at(size_t elements = 0) {
        return reinterpret_cast<T*>(static_cast<unsigned char*>(device) + guard_bytes) + elements;
    }
    std::vector<unsigned char> read(bool immutable) {
        std::vector<unsigned char> observed(original.size());
        check(hipMemcpy(observed.data(), device, observed.size(), hipMemcpyDeviceToHost));
        require(std::equal(observed.begin(), observed.begin()+guard_bytes, original.begin()) &&
                std::equal(observed.end()-guard_bytes, observed.end(), original.end()-guard_bytes), "redzone changed");
        if (immutable) require(observed == original, "captured input mutated");
        return {observed.begin()+guard_bytes, observed.end()-guard_bytes};
    }
};
struct Table {
    Table() { check(qrt_fla_blackwell_state::prepare_exp2_table()); }
    ~Table() { qrt_fla_blackwell_state::release_exp2_table(); }
};
struct Stream {
    hipStream_t value = nullptr;
    Stream() { check(hipStreamCreate(&value)); }
    ~Stream() { if (value) { (void)hipStreamSynchronize(value);(void)hipStreamDestroy(value); } }
};
struct Event {
    hipEvent_t value = nullptr;
    Event() { check(hipEventCreate(&value)); }
    ~Event() { if (value) (void)hipEventDestroy(value); }
};
void number(double value) { if (std::isfinite(value)) std::cout << value;else std::cout << "null"; }
template<class Operation>
bool timed(const char* stage, unsigned first, unsigned count, hipStream_t stream, Operation operation) {
    Event begin, end;
    const qrt_fla_completion::Timer<> timer;
    check(hipEventRecord(begin.value, stream));
    try {
        check(operation());check(hipEventRecord(end.value, stream));check(hipEventSynchronize(end.value));
    } catch (...) { (void)hipStreamSynchronize(stream);throw; }
    const double host = timer.elapsed_ms();
    float gpu = 0;check(hipEventElapsedTime(&gpu, begin.value, end.value));
    const auto decision = qrt_fla_completion::evaluate(gpu, host);
    std::cout << "{\"kind\":\"fla_output_replay_interval\",\"stage\":\"" << stage
              << "\",\"first_token\":" << first << ",\"tokens\":" << count << ",\"gpu_ms\":";
    number(gpu);std::cout << ",\"host_ms\":";number(host);
    std::cout << ",\"guard_ms\":100,\"completed\":true,\"accepted\":" << (decision.accepted()?"true":"false")
              << ",\"gpu_clock_finite_nonnegative\":" << (std::isfinite(gpu) && gpu>=0?"true":"false")
              << ",\"selected_clock\":\"" << (decision.source==qrt_fla_completion::ClockSource::gpu?"gpu":
                  decision.source==qrt_fla_completion::ClockSource::host?"host":"unavailable") << "\"}\n" << std::flush;
    return decision.accepted();
}
size_t differences(const std::vector<unsigned char>& actual, const std::vector<unsigned char>& reference, size_t cells, size_t width) {
    require(cells <= actual.size()/width && cells <= reference.size()/width, "comparison span");
    size_t result = 0;
    for (size_t i=0;i<cells;++i) result += std::memcmp(actual.data()+i*width,reference.data()+i*width,width)!=0;
    return result;
}
} // namespace

int main(int argc, char** argv) try {
    require(argc==4, "usage: output-failure-replay <capture-dir> <tokens1..1024> <combined|scores|output|split|tiles>");
    unsigned count=0;
    require(*argv[2]!=0, "empty token count");
    for (const char* p=argv[2];*p;++p) {
        require(*p>='0' && *p<='9' && count<=102u, "invalid token count");count=count*10u+unsigned(*p-'0');
    }
    require(count>0 && count<=1024, "token count outside captured segment");
    const std::string root=argv[1],mode=argv[3];
    require(mode=="combined" || mode=="scores" || mode=="output" || mode=="split" || mode=="tiles", "invalid replay mode");
    const auto q=load(root,"q-bf16.bin",size_t(count)*2048u*2u);
    const auto k=load(root,"k-bf16.bin",q.size());
    const auto v=load(root,"v-new-bf16.bin",size_t(count)*4096u*2u);
    const auto h=load(root,"chunk-state-bf16.bin",size_t((count+63u)/64u)*524288u*2u);
    const auto g=load(root,"g-cumsum-f32.bin",size_t(count)*32u*4u);
    const auto scores=load(root,"scores-bf16.bin",q.size());
    const auto output=load(root,"output-f32.bin",size_t(count)*4096u*4u);
    const size_t captured_bytes=q.size()+k.size()+v.size()+h.size()+g.size()+scores.size()+output.size();
    require(captured_bytes <= (64u<<20), "capture exceeds diagnostic budget");
    setting("QRT_FLA_GDN_COOPERATIVE_EXACT","1");setting("QRT_FLA_GDN_SCALAR_FLOAT_MATRICES","1");
    setting("QRT_FLA_GDN_COARSE_INTERVAL","0");setting("QRT_FLA_GDN_PAIRED_SCORE_ARENAS","1");
    check(hipSetDevice(0));hipDeviceProp_t device{};check(hipGetDeviceProperties(&device,0));
    require(std::string(device.gcnArchName).find("gfx1151")==0, "requires gfx1151");
    size_t free_bytes=0,total_bytes=0;check(hipMemGetInfo(&free_bytes,&total_bytes));
    require(free_bytes >= captured_bytes+(512u<<20), "device memory reserve");
    Table table;
    require(qrt_fla_blackwell_state::exp2_table_device()!=nullptr, "requires validated original exp2 table");
    Stream stream;
    Buffer dq(q),dk(k),dv(v),dh(h),dg(g),ds(mode=="output"?scores:std::vector<unsigned char>(scores.size(),0xa5u)),
        dout(std::vector<unsigned char>(output.size(),0xa5u));
    const auto* exp=qrt_fla_blackwell_state::exp2_table_device();
    std::cout << std::setprecision(17);
    bool accepted=true;unsigned score_rows=0,output_rows=0;
    if (mode=="combined" || mode=="tiles") {
        const unsigned step=mode=="tiles"?64u:count;
        for (unsigned first=0;first<count;first+=step) {
            const unsigned n=std::min(step,count-first);
            accepted=timed("scores_and_output",first,n,stream.value,[&] {
                return qrt_fla_blackwell_aux::output_segment(dq.at<uint16_t>(size_t(first)*2048u),
                    dk.at<uint16_t>(size_t(first)*2048u),dv.at<uint16_t>(size_t(first)*4096u),
                    dh.at<uint16_t>(size_t(first/64u)*524288u),dg.at<float>(size_t(first)*32u),
                    ds.at<uint16_t>(size_t(first)*2048u),dout.at<float>(size_t(first)*4096u),n,stream.value);
            });
            score_rows=output_rows=first+n;
            if (!accepted) break;
        }
    } else {
        if (mode=="scores" || mode=="split") {
            accepted=timed("scores",0,count,stream.value,[&] {
                return qrt_fla_blackwell_cooperative::scores(dq.at<uint16_t>(),dk.at<uint16_t>(),dg.at<float>(),ds.at<uint16_t>(),count,exp,stream.value);
            });score_rows=count;
        }
        if (mode=="output" || (mode=="split" && accepted)) {
            accepted=timed("output",0,count,stream.value,[&] {
                return qrt_fla_blackwell_cooperative::output(dq.at<uint16_t>(),dv.at<uint16_t>(),dh.at<uint16_t>(),dg.at<float>(),ds.at<uint16_t>(),dout.at<float>(),count,exp,stream.value);
            });output_rows=count;
        }
    }
    // Every preceding interval completed. Only diagnostic reads follow a
    // rejected bound; no additional kernel is submitted after rejection.
    dq.read(true);dk.read(true);dv.read(true);dh.read(true);dg.read(true);
    const auto actual_scores=ds.read(mode=="output"),actual_output=dout.read(false);
    if (mode!="output") require(std::all_of(actual_scores.begin()+size_t(score_rows)*4096u,
        actual_scores.end(),[](unsigned char x){return x==0xa5u;}), "unsubmitted score rows changed");
    require(std::all_of(actual_output.begin()+size_t(output_rows)*16384u,
        actual_output.end(),[](unsigned char x){return x==0xa5u;}), "unsubmitted output rows changed");
    const size_t score_differences=differences(actual_scores,scores,size_t(score_rows)*2048u,2u);
    const size_t output_differences=differences(actual_output,output,size_t(output_rows)*4096u,4u);
    std::cout << "{\"kind\":\"fla_output_failure_replay\",\"mode\":\"" << mode << "\",\"tokens\":" << count
        << ",\"score_rows_recomputed\":" << score_rows << ",\"output_rows_recomputed\":" << output_rows
        << ",\"score_bit_mismatches\":" << score_differences << ",\"output_bit_mismatches\":" << output_differences
        << ",\"immutable_inputs\":true,\"redzones_pass\":true,\"completed_bound_pass\":" << (accepted?"true":"false")
        << ",\"model_loaded\":false,\"gb10_acceptance\":false,\"performance_acceptance\":false}\n";
    if (!accepted) return 3;
    return score_differences || output_differences ? 4 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr,"output failure replay: %s\n",error.what());return 1;
}
