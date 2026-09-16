#include "../../native/providers/ck_fmha/native_delta_probability.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace qrt_blackwell_attention;
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
struct Device {
    void* pointer = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&pointer, bytes)); }
    ~Device() { if (pointer) (void)hipFree(pointer); }
    template<class T> T* as() { return static_cast<T*>(pointer); }
};
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("probability completion deadline");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(hipEventDestroy(event));
}
template<class T> void upload(Device& d, const std::vector<T>& v) {
    check(hipMemcpy(d.pointer, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
}
template<class T> std::vector<T> download(Device& d, size_t count) {
    std::vector<T> v(count); check(hipMemcpy(v.data(), d.pointer, count * sizeof(T), hipMemcpyDeviceToHost)); return v;
}
float score_value(unsigned row, unsigned key, unsigned mode) {
    uint32_t random = (row + 3u) * 747796405u + key * 2891336453u;
    random = (random ^ (random >> 16u)) * 2246822519u;
    if (mode == 0u) return float(int(random % 10241u) - 5120) / 256.0f;
    if (mode == 1u) return key % 7u == 0u ? 8.0f : -float(random % 4096u) / 128.0f;
    if (mode == 2u) return float(key / 32u) * 256.0f - float(key % 32u);
    if (mode == 3u) return key % 2u ? -0.0f : 0.0f;
    if (mode == 4u) return std::ldexp(float(int(random % 33u) - 16), -129);
    return (key / 32u) % 3u == 0u ? -1000.0f : float(int(random % 65u) - 32) / 8.0f;
}
void run(unsigned start, unsigned queries, unsigned mode, bool vllm, const unsigned char* table, const unsigned char* packed) {
    constexpr size_t guard = 64u;
    if(start+queries>8192u) throw std::runtime_error("staged capacity exceeded");
    const unsigned stride = start + queries, rows = queries * kQueryHeads;
    const unsigned tile_stride = (stride + 31u) / 32u;
    const size_t cells = size_t(rows) * stride, scale_cells = size_t(rows) * (tile_stride + 1u);
    std::vector<float> scores(cells + 2u * guard, 12345.0f), scales(scale_cells + 2u * guard, 12345.0f);
    std::vector<uint16_t> probability(cells + 2u * guard, 0xa5a5u);
    for (unsigned row = 0u; row < rows; ++row) {
        const unsigned tokens = start + row / kQueryHeads + 1u;
        for (unsigned key = 0u; key < tokens; ++key)
            scores[guard + size_t(row) * stride + key] = score_value(row, key, mode);
    }
    Device ds(scores.size() * sizeof(float)), dp(probability.size() * sizeof(uint16_t)), da(scales.size() * sizeof(float));
    upload(ds, scores); upload(dp, probability); upload(da, scales);
    hipLaunchKernelGGL(blackwell_online_probability_kernel, dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
        ds.as<float>() + guard, dp.as<uint16_t>() + guard, da.as<float>() + guard, start, stride, table, vllm);
    check(hipGetLastError()); finish();
    const auto control_p = download<uint16_t>(dp, probability.size());
    const auto control_s = download<float>(da, scales.size());
    upload(dp, probability); upload(da, scales);
    hipLaunchKernelGGL(qrt_native_delta_probability::probabilities, dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
        ds.as<float>() + guard, dp.as<uint16_t>() + guard, da.as<float>() + guard, start, stride, table, vllm, packed);
    check(hipGetLastError()); finish();
    const auto actual_p = download<uint16_t>(dp, probability.size());
    const auto actual_s = download<float>(da, scales.size());
    const auto actual_scores = download<float>(ds, scores.size());
    if (std::memcmp(actual_scores.data(), scores.data(), scores.size() * sizeof(float))) throw std::runtime_error("scores changed");
    size_t bad_p = 0u, bad_s = 0u, denominators = 0u, zero_alphas = 0u;
    for (size_t i = 0u; i < actual_p.size(); ++i) {
        bad_p += actual_p[i] != control_p[i];
        bool live = i >= guard && i < guard + cells;
        if (live) {
            const unsigned row = unsigned((i - guard) / stride), key = unsigned((i - guard) % stride);
            const unsigned tokens = start + row / kQueryHeads + 1u;
            live = key < ((tokens + 31u) / 32u) * 32u;
            if (live && mode == 3u && actual_p[i] != (key < tokens ? 0x3f80u : 0u))
                throw std::runtime_error("constant probability CPU check");
        }
        if (!live && actual_p[i] != probability[i]) throw std::runtime_error("probability tail or redzone changed");
    }
    for (size_t i = 0u; i < actual_s.size(); ++i) {
        const bool mismatch = std::memcmp(&actual_s[i], &control_s[i], sizeof(float)) != 0;
        bad_s += mismatch;
        bool live = i >= guard && i < guard + scale_cells;
        if (live) {
            const unsigned row = unsigned((i - guard) / (tile_stride + 1u)), tile = unsigned((i - guard) % (tile_stride + 1u));
            const unsigned tokens = start + row / kQueryHeads + 1u;
            live = tile < (tokens + 31u) / 32u || tile == tile_stride;
            if (live && !std::isfinite(actual_s[i])) throw std::runtime_error("nonfinite scale");
            if (tile == tile_stride) denominators += mismatch;
            else if (live && actual_s[i] == 0.0f) ++zero_alphas;
            if (live && mode == 3u && actual_s[i] != (tile == tile_stride ? float(tokens) : tile == 0u ? 0.0f : 1.0f))
                throw std::runtime_error("constant denominator CPU check");
        }
        if (!live && actual_s[i] != scales[i]) throw std::runtime_error("scale tail or redzone changed");
    }
    std::printf("{\"kind\":\"native_delta_probability\",\"query_start\":%u,\"queries\":%u,\"mode\":%u,\"vllm_sum\":%s,\"sm121_exp2_table\":%s,\"probability_cells\":%llu,\"scale_cells\":%llu,\"probability_bit_mismatches\":%llu,\"scale_bit_mismatches\":%llu,\"denominator_bit_mismatches\":%llu,\"zero_alphas\":%llu,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
        start, queries, mode, vllm ? "true" : "false", table ? "true" : "false", (unsigned long long)cells,
        (unsigned long long)scale_cells, (unsigned long long)bad_p, (unsigned long long)bad_s, (unsigned long long)denominators, (unsigned long long)zero_alphas);
    if (bad_p || bad_s) throw std::runtime_error("native-delta probability differs from original one-wave recurrence");
}
}
namespace {
namespace delta = qrt_sm121_exp2_native_delta;
struct Stats { unsigned counts[7]{}; };
__global__ void verify_table(const unsigned char* original, const unsigned char* packed, Stats* total) {
    Stats local;
    for (uint32_t relative=blockIdx.x*blockDim.x+threadIdx.x; relative<delta::cells;
         relative+=gridDim.x*blockDim.x) {
        const float argument=qrt_sm121_exp2::value(0x80000000u|(delta::source::begin+relative));
        const uint32_t expected=delta::source::decode(original,relative);
        const uint32_t native=qrt_sm121_exp2::bits(delta::native_exp(argument));
        const unsigned code=delta::code(packed,relative);
        ++local.counts[code];
        local.counts[4]+=qrt_sm121_exp2::bits(delta::evaluate(original,packed,argument))!=expected;
        local.counts[5]+=code!=delta::encode(native,expected);
        const unsigned difference=native>expected?native-expected:expected-native;
        local.counts[6]=max(local.counts[6],difference);
    }
    auto* values=local.counts;
    auto* destination=total->counts;
    static_assert(sizeof(Stats)==7u*sizeof(unsigned));
#pragma unroll
    for(unsigned item=0u;item<7u;++item){
        unsigned value=values[item];
        for(unsigned shift=16u;shift;shift>>=1u){
            const unsigned other=__shfl_down(value,shift,32u);
            value=item==6u?max(value,other):value+other;
        }
        if(!(threadIdx.x&31u)){
            if(item==6u)atomicMax(destination+item,value);
            else atomicAdd(destination+item,value);
        }
    }
}
__global__ void exterior(const uint32_t* inputs,uint32_t* outputs,unsigned count){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)outputs[i]=qrt_sm121_exp2::bits(delta::evaluate(nullptr,nullptr,qrt_sm121_exp2::value(inputs[i])));
}
template<class Function> double completed(Function fn){
    const auto start=std::chrono::steady_clock::now();fn();check(hipGetLastError());finish();
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
void validate_guard(Device& d,size_t bytes){
    unsigned char before[128],after[128];
    check(hipMemcpy(before,d.pointer,128u,hipMemcpyDeviceToHost));
    check(hipMemcpy(after,d.as<unsigned char>()+128u+bytes,128u,hipMemcpyDeviceToHost));
    for(unsigned i=0;i<128u;++i)if(before[i]!=0xa5u||after[i]!=0xa5u)throw std::runtime_error("delta workspace redzone changed");
}
void validate_exterior(){
    std::vector<uint32_t> input,expected;
    for(uint32_t sign:{0u,0x80000000u})for(uint32_t magnitude:{0u,1u,0x007fffffu,0x00800000u,
        delta::source::begin-1u,qrt_sm121_exp2::positive_one_end-1u,qrt_sm121_exp2::positive_one_end,
        delta::source::end,delta::source::end+1u,0x7f7fffffu,0x7f800000u,0x7f800001u,0x7fc00000u,0x7fffffffu}){
        if(sign&&magnitude>=delta::source::begin&&magnitude<delta::source::end)continue;
        input.push_back(sign|magnitude);
        expected.push_back(qrt_sm121_exp2::bits(delta::source::evaluate(nullptr,qrt_sm121_exp2::value(sign|magnitude))));
    }
    Device di(input.size()*4u),dout((expected.size()+64u)*4u);upload(di,input);
    check(hipMemset(dout.pointer,0xa5,(expected.size()+64u)*4u));
    hipLaunchKernelGGL(exterior,dim3(1u),dim3(256u),0u,nullptr,di.as<uint32_t>(),dout.as<uint32_t>()+32u,unsigned(input.size()));
    check(hipGetLastError());finish();
    const auto output=download<uint32_t>(dout,expected.size()+64u);
    for(size_t i=0;i<output.size();++i)if(output[i]!=(i>=32u&&i<32u+expected.size()?expected[i-32u]:0xa5a5a5a5u))throw std::runtime_error("exterior guard or result mismatch");
    if(download<uint32_t>(di,input.size())!=input)throw std::runtime_error("exterior inputs changed");
    std::printf("{\"kind\":\"native_exp2_delta_exterior\",\"cells\":%zu,\"bit_mismatches\":0,\"null_table_pointers\":true,\"redzones_pass\":true,\"immutable_inputs\":true}\n",input.size());
}
} // namespace
int main(int argc,char** argv)try{
    if(argc!=2)throw std::runtime_error("requires SHA-verified original interpolated table");
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0));
    if(std::strncmp(properties.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    std::ifstream file(argv[1],std::ios::binary|std::ios::ate);
    if(!file||file.tellg()!=std::streamoff(delta::source::table_bytes))throw std::runtime_error("table size mismatch");
    std::vector<unsigned char> table(delta::source::table_bytes);file.seekg(0);file.read(reinterpret_cast<char*>(table.data()),table.size());
    if(!file||!delta::source::valid_layout(table.data(),table.size()))throw std::runtime_error("table layout mismatch");
    Device dt(table.size()+256u),dd(delta::packed_bytes+256u),ds(sizeof(Stats));
    check(hipMemset(dt.pointer,0xa5,table.size()+256u));check(hipMemset(dd.pointer,0xa5,delta::packed_bytes+256u));
    auto* original=dt.as<unsigned char>()+128u;auto* packed=dd.as<unsigned char>()+128u;
    check(hipMemcpy(original,table.data(),table.size(),hipMemcpyHostToDevice));
    const double build_ms=completed([&]{hipLaunchKernelGGL(delta::build,dim3(4096u),dim3(256u),0u,nullptr,original,packed);});
    std::vector<unsigned char> saved(delta::packed_bytes);
    check(hipMemcpy(saved.data(),packed,saved.size(),hipMemcpyDeviceToHost));
    Stats previous;
    for(unsigned attempt=0;attempt<2u;++attempt){
        check(hipMemset(ds.pointer,0,sizeof(Stats)));
        const double verify_ms=completed([&]{hipLaunchKernelGGL(verify_table,dim3(4096u),dim3(256u),0u,nullptr,original,packed,ds.as<Stats>());});
        const auto stat=download<Stats>(ds,1u)[0];
        if(stat.counts[4]||stat.counts[5]||uint64_t(stat.counts[0])+stat.counts[1]+stat.counts[2]+stat.counts[3]!=delta::cells)throw std::runtime_error("full-domain delta verification failed");
        if(attempt&&std::memcmp(&previous,&stat,sizeof(stat)))throw std::runtime_error("native EXP repeat changed");
        previous=stat;validate_guard(dt,table.size());validate_guard(dd,delta::packed_bytes);
        std::printf("{\"kind\":\"native_exp2_delta_domain\",\"attempt\":%u,\"cells\":%u,\"packed_bytes\":%zu,\"minus_one\":%u,\"equal\":%u,\"plus_one\":%u,\"escaped\":%u,\"maximum_absolute_native_ulp_difference\":%u,\"bit_mismatches\":0,\"encoding_mismatches\":0,\"build_completed_host_ms\":%.9f,\"verify_completed_host_ms\":%.9f,\"redzones_pass\":true,\"inference_acceptance\":false}\n",attempt,delta::cells,delta::packed_bytes,stat.counts[0],stat.counts[1],stat.counts[2],stat.counts[3],stat.counts[6],build_ms,verify_ms);std::fflush(stdout);
    }
    validate_exterior();
    for(auto shape:{std::pair<unsigned,unsigned>{0,1},{0,32},{31,2},{17,32},{255,2},{7167,2},{8191,1},{8064,128},{0,128},{511,65}})
        for(unsigned mode=0;mode<6u;++mode)for(bool vllm:{false,true})run(shape.first,shape.second,mode,vllm,original,packed);
    std::vector<unsigned char> after_table(table.size()),after_packed(saved.size());
    check(hipMemcpy(after_table.data(),original,after_table.size(),hipMemcpyDeviceToHost));
    check(hipMemcpy(after_packed.data(),packed,after_packed.size(),hipMemcpyDeviceToHost));
    if(after_table!=table||after_packed!=saved)throw std::runtime_error("source or derived table changed");
    validate_guard(dt,table.size());validate_guard(dd,delta::packed_bytes);
    std::printf("{\"kind\":\"native_exp2_delta_ownership\",\"source_bytes\":%zu,\"derived_bytes\":%zu,\"immutable_inputs\":true,\"redzones_pass\":true,\"probability_cases\":120,\"inference_acceptance\":false}\n",table.size(),delta::packed_bytes);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"native_exp2_delta_selftest_error=%s\n",e.what());return 2;}
