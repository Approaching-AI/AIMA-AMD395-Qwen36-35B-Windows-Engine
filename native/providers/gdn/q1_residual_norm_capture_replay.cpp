// Execute the actual extracted provider kernels on original-operation replay
// operands. Reference buffers remain on the host and never feed a kernel.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <hip/hip_runtime.h>
#include "sm121_rsqrt_table.h"
#include "postnorm_live_extract.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t rows = 80, width = 2048, elements = rows * width;
bool completion_known = true;
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
template<class T> std::vector<T> read(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != std::streamoff(count * sizeof(T))) throw std::runtime_error("input size");
    std::vector<T> values(count);f.seekg(0);
    f.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    if (!f) throw std::runtime_error("input read");
    return values;
}
float widen(uint16_t value) {
    const uint32_t bits = uint32_t(value) << 16u;float result;
    std::memcpy(&result, &bits, sizeof(result));return result;
}
uint32_t bits(float value) { uint32_t result;std::memcpy(&result, &value, 4);return result; }
struct Device {
    unsigned char* allocation = nullptr;size_t bytes;
    explicit Device(size_t count):bytes(count) {
        check(hipMalloc(reinterpret_cast<void**>(&allocation), bytes + 512u));
        check(hipMemset(allocation, 0x5a, bytes + 512u));
    }
    ~Device() { if (allocation && completion_known) (void)hipFree(allocation); }
    Device(const Device&) = delete;
    template<class T> T* as() { return reinterpret_cast<T*>(allocation + 256u); }
    template<class T> void upload(const std::vector<T>& source) {
        if (source.size() * sizeof(T) != bytes) throw std::runtime_error("upload size");
        check(hipMemcpy(as<T>(), source.data(), bytes, hipMemcpyHostToDevice));
    }
    template<class T> std::vector<T> download() {
        std::vector<T> result(bytes / sizeof(T));
        check(hipMemcpy(result.data(), as<T>(), bytes, hipMemcpyDeviceToHost));return result;
    }
    unsigned guard_errors() {
        std::array<unsigned char,512> guard;
        check(hipMemcpy(guard.data(), allocation, 256u, hipMemcpyDeviceToHost));
        check(hipMemcpy(guard.data()+256u, allocation+256u+bytes, 256u, hipMemcpyDeviceToHost));
        unsigned bad = 0;for (auto value : guard) bad += value != 0x5au;return bad;
    }
};
__global__ void q1_residual_scalars(const float* sums, const unsigned char* correction,
                                  const unsigned char* table, float* output,
                                  bool reference_single_row) {
    __shared__ float partial[512];const unsigned lane = threadIdx.x;
    float values[8];for (unsigned i=0;i<8u;++i) values[i]=sums[size_t(blockIdx.x)*2048u+lane*8u+i];
    const float sum = reference_single_row
        ? vllm_triton_q1_reduce_sumsq(values,partial,lane)
        : vllm_triton_reduce_sumsq(vllm_triton_lane8_sumsq(values),partial,lane);
    if (!lane) {
        const float variance = sum/2048.0f, total = __fadd_rn(variance,1.0e-6f);
        output[blockIdx.x*3u] = variance;
        output[blockIdx.x*3u+1u] = device_sm121_rsqrt_from_gfx1151(total,correction);
        output[blockIdx.x*3u+2u] = qrt_sm121_rsqrt::evaluate(table,total);
    }
}
}

int main(int argc,char** argv) try {
    if (argc!=5 || (std::string(argv[4])!="1" && std::string(argv[4])!="2"))
        throw std::runtime_error("input_directory delta2_table original_rsqrt_table reference_compute_rows(1|2)");
    const bool reference_single_row = std::string(argv[4]) == "1";
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0));
    if (std::string(properties.gcnArchName).find("gfx1151")!=0) throw std::runtime_error("requires gfx1151");
    const std::string directory=argv[1];
    const auto operands=read<uint16_t>(directory+"/operands.bin",2u*elements);
    const auto weights=read<uint16_t>(directory+"/weights.bf16.bin",elements);
    const auto expected=read<uint16_t>(directory+"/original-output.bf16.bin",elements);
    const auto expected_carrier=read<uint16_t>(directory+"/original-residual.bf16.bin",elements);
    const auto expected_variance=read<float>(directory+"/variance.f32.bin",rows);
    const auto expected_inverse=read<float>(directory+"/inverse.f32.bin",rows);
    const auto correction=read<unsigned char>(argv[2],4194304u);
    const auto table=read<unsigned char>(argv[3],qrt_sm121_rsqrt::table_bytes);
    if (!qrt_sm121_rsqrt::valid_layout(table.data(),table.size())) throw std::runtime_error("table layout");
    std::vector<float> residual(elements),sums(elements),zero(elements,0.0f),sigmoid(65536u,0.0f);
    std::vector<uint16_t> updates(elements),gate(1u,0u);sigmoid[0]=1.0f;
    for (size_t row=0;row<rows;++row) for (size_t i=0;i<width;++i) {
        const size_t at=row*width+i;residual[at]=widen(operands[row*2u*width+i]);
        updates[at]=operands[row*2u*width+width+i];sums[at]=residual[at]+widen(updates[at]);
    }
    Device dr(elements*4u),du(elements*2u),dw(elements*2u),ds(elements*4u),dc(correction.size()),dt(table.size());
    Device dh(elements*4u),dn(elements*4u),db(elements*2u),df(elements*4u),dm(elements*4u),dmb(elements*2u),dmc(elements*4u);
    Device dz(elements*4u),dg(2u),dscale(sigmoid.size()*4u),dscalar(rows*3u*4u);
    dr.upload(residual);du.upload(updates);dw.upload(weights);ds.upload(sums);dc.upload(correction);dt.upload(table);
    dz.upload(zero);dg.upload(gate);dscale.upload(sigmoid);
    hipStream_t stream=nullptr;check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    // All uploads/guard initialization complete before the independent stream.
    check(hipDeviceSynchronize());completion_known=false;
    for (size_t row=0;row<rows;++row) {
        const size_t at=row*width;
        hipLaunchKernelGGL(output_bf16_residual_postnorm_vllm_kernel,dim3(1),dim3(256),0,stream,
            dr.as<float>()+at,du.as<uint16_t>()+at,dw.as<uint16_t>()+at,dh.as<float>()+at,dn.as<float>()+at,
            1u,dc.as<unsigned char>(),db.as<uint16_t>()+at,reference_single_row);check(hipGetLastError());
        hipLaunchKernelGGL(final_norm_unrounded_vllm_kernel,dim3(1),dim3(256),0,stream,
            ds.as<float>()+at,dw.as<uint16_t>()+at,df.as<float>()+at,1u,dc.as<unsigned char>(),reference_single_row);check(hipGetLastError());
        hipLaunchKernelGGL(q1_moe_sm121_tail_kernel,dim3(1),dim3(256),0,stream,
            du.as<uint16_t>()+at,dg.as<uint16_t>(),dz.as<float>()+at,dr.as<float>()+at,dmc.as<float>()+at,
            dw.as<uint16_t>()+at,dm.as<float>()+at,dmb.as<uint16_t>()+at,dscale.as<float>(),dt.as<unsigned char>(),true,reference_single_row);
        check(hipGetLastError());
    }
    hipLaunchKernelGGL(q1_residual_scalars,dim3(rows),dim3(256),0,stream,
        ds.as<float>(),dc.as<unsigned char>(),dt.as<unsigned char>(),dscalar.as<float>(),reference_single_row);check(hipGetLastError());
    check(hipStreamSynchronize(stream));completion_known=true;
    unsigned post=0,post_bf16=0,final=0,moe=0,moe_bf16=0,carrier=0,unrounded=0,var=0,inverse=0,table_difference=0,guards=0;
    const auto h=dh.download<float>(),n=dn.download<float>(),f=df.download<float>(),mt=dm.download<float>(),mc=dmc.download<float>();
    const auto packed=db.download<uint16_t>(),mp=dmb.download<uint16_t>();const auto scalar=dscalar.download<float>();
    for (size_t i=0;i<elements;++i) {
        const auto e=bits(widen(expected[i]));post+=bits(n[i])!=e;post_bf16+=packed[i]!=expected[i];
        final+=bits(f[i])!=e;moe+=bits(mt[i])!=e;moe_bf16+=mp[i]!=expected[i];
        carrier+=bits(h[i])!=bits(widen(expected_carrier[i]));unrounded+=bits(mc[i])!=bits(sums[i]);
    }
    for (size_t i=0;i<rows;++i) {
        var+=bits(scalar[3u*i])!=bits(expected_variance[i]);inverse+=bits(scalar[3u*i+1u])!=bits(expected_inverse[i]);
        table_difference+=bits(scalar[3u*i+1u])!=bits(scalar[3u*i+2u]);
    }
    for (auto* buffer:{&dr,&du,&dw,&ds,&dc,&dt,&dh,&dn,&db,&df,&dm,&dmb,&dmc,&dz,&dg,&dscale,&dscalar}) guards+=buffer->guard_errors();
    const bool unchanged=dr.download<float>()==residual && du.download<uint16_t>()==updates && dw.download<uint16_t>()==weights &&
        ds.download<float>()==sums && dc.download<unsigned char>()==correction && dt.download<unsigned char>()==table;
    check(hipStreamDestroy(stream));
    const bool passed=!(post+post_bf16+final+moe+moe_bf16+carrier+unrounded+table_difference+guards)&&unchanged;
    std::cout<<"{\"kind\":\"actual_q1_residual_norm_kernel_replay\",\"cases\":80,\"elements_per_surface\":163840,"
        <<"\"reference_compute_rows\":"<<(reference_single_row?1:2)<<","
        <<"\"postnorm_f32_mismatches\":"<<post<<",\"postnorm_bf16_mismatches\":"<<post_bf16
        <<",\"finalnorm_mismatches\":"<<final<<",\"moe_norm_f32_mismatches\":"<<moe<<",\"moe_norm_bf16_mismatches\":"<<moe_bf16
        <<",\"rounded_carrier_mismatches\":"<<carrier<<",\"unrounded_carrier_mismatches\":"<<unrounded
        <<",\"diagnostic_variance_mismatches\":"<<var<<",\"diagnostic_inverse_mismatches\":"<<inverse
        <<",\"rsqrt_table_disagreements\":"<<table_difference<<",\"guard_errors\":"<<guards
        <<",\"inputs_unchanged\":"<<(unchanged?"true":"false")<<",\"passed\":"<<(passed?"true":"false")
        <<",\"reference_is_compute_input\":false,\"native_execution\":true,\"inference_acceptance\":false}\n";
    return passed?0:1;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 2; }
