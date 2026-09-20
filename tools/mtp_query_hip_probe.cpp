#include <hip/hip_runtime.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "native/providers/gdn/sm121_mtp_query.h"
#include "native/providers/gdn/sm121_mtp_gate.h"

void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
template<class T> std::vector<T> read_file(const char* path) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() % sizeof(T)) throw std::runtime_error("input file");
    const size_t bytes=static_cast<size_t>(file.tellg());
    if (bytes>128u*1024u*1024u) throw std::runtime_error("file bound");
    std::vector<T> values(bytes/sizeof(T));file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()),bytes);
    if (!file) throw std::runtime_error("short input");
    return values;
}
struct Scratch {
    std::vector<void*> pointers;
    ~Scratch() {
        if (hipDeviceSynchronize()!=hipSuccess) return;
        for(void* pointer:pointers) (void)hipFree(pointer);
    }
    template<class T> T* upload(const std::vector<T>& values) {
        T* pointer=nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&pointer),values.size()*sizeof(T)));
        pointers.push_back(pointer);
        check(hipMemcpy(pointer,values.data(),values.size()*sizeof(T),hipMemcpyHostToDevice));
        return pointer;
    }
};
int main(int argc,char** argv) try {
    if(argc!=8 && argc!=11) throw std::runtime_error("q_projection weights expected_norm expected_rope positions rsqrt rope [context expected_gated sigmoid]");
    const bool check_gated=argc==11;
    const auto input=read_file<uint16_t>(argv[1]),weights=read_file<uint16_t>(argv[2]);
    const auto expected_norm=read_file<uint16_t>(argv[3]),expected_rope=read_file<uint16_t>(argv[4]);
    const auto positions=read_file<uint32_t>(argv[5]);const auto table=read_file<unsigned char>(argv[6]);
    const auto rope=read_file<uint16_t>(argv[7]);const size_t rows=positions.size();
    const auto context=check_gated?read_file<uint16_t>(argv[8]):std::vector<uint16_t>{};
    const auto expected_gated=check_gated?read_file<uint16_t>(argv[9]):std::vector<uint16_t>{};
    const auto sigmoid=check_gated?read_file<uint16_t>(argv[10]):std::vector<uint16_t>{};
    if(!rows || rows>8192u || input.size()!=rows*8192u || weights.size()!=256u ||
       expected_norm.size()!=rows*4096u || expected_rope.size()!=expected_norm.size() ||
       rope.size()%64u || !qrt_sm121_rsqrt::valid_layout(table.data(),table.size()))
        throw std::runtime_error("shape or table");
    if(check_gated && (context.size()!=rows*4096u || expected_gated.size()!=context.size() || sigmoid.size()!=65536u))
        throw std::runtime_error("gated context shape or table");
    for(size_t i=0;i<rows;++i)
        if(positions[i]>=262144u || positions[i]>=rope.size()/64u || (i && positions[i]<=positions[i-1u]))
            throw std::runtime_error("positions must increase within the MTP limit");
    constexpr size_t guard=64u;constexpr uint16_t sentinel=0xa5a5u;
    std::vector<uint16_t> guarded_input(input.size()+guard*2u,sentinel);
    std::copy(input.begin(),input.end(),guarded_input.begin()+guard);
    std::vector<uint16_t> norm(expected_norm.size()+guard*2u,sentinel),queries=norm,gates=norm;
    Scratch scratch;
    auto device_input=scratch.upload(guarded_input),device_weights=scratch.upload(weights);
    auto device_rope=scratch.upload(rope);auto device_table=scratch.upload(table);
    auto device_norm=scratch.upload(norm),device_queries=scratch.upload(queries),device_gates=scratch.upload(gates);
    const auto launch=[&](size_t first,unsigned position,unsigned count) {
        return qrt_sm121_mtp::launch_queries(device_input+guard+first*8192u,device_weights,
            device_table,device_rope,static_cast<unsigned>(rope.size()/64u),position,count,
            device_queries+guard+first*4096u,device_gates+guard+first*4096u,
            device_norm+guard+first*4096u);
    };
    if(launch(0u,0u,0u)!=hipErrorInvalidValue || launch(0u,262144u,1u)!=hipErrorInvalidValue ||
       launch(0u,262143u,2u)!=hipErrorInvalidValue || launch(0u,0u,8193u)!=hipErrorInvalidValue)
        throw std::runtime_error("launch bound accepted invalid shape");
    unsigned launches=0;
    for(size_t first=0;first<rows;) {
        size_t last=first+1u;while(last<rows && positions[last]==positions[last-1u]+1u)++last;
        check(launch(first,positions[first],static_cast<unsigned>(last-first)));++launches;first=last;
    }
    check(hipDeviceSynchronize());
    check(hipMemcpy(norm.data(),device_norm,norm.size()*sizeof(uint16_t),hipMemcpyDeviceToHost));
    check(hipMemcpy(queries.data(),device_queries,queries.size()*sizeof(uint16_t),hipMemcpyDeviceToHost));
    check(hipMemcpy(gates.data(),device_gates,gates.size()*sizeof(uint16_t),hipMemcpyDeviceToHost));
    size_t norm_bad=0,rope_bad=0,gate_bad=0,guard_bad=0,input_bad=0;
    size_t gated_bad=0,context_input_bad=0,sigmoid_bad=0;
    if(check_gated) {
        std::vector<uint16_t> guarded_context(context.size()+2u*guard,sentinel),gated(guarded_context.size(),sentinel);
        std::copy(context.begin(),context.end(),guarded_context.begin()+guard);
        auto* device_context=scratch.upload(guarded_context);
        auto* device_gated=scratch.upload(gated);auto* device_sigmoid=scratch.upload(sigmoid);
        for(unsigned invalid_rows:{0u,8193u})
            if(qrt_sm121_mtp::launch_gate(device_context+guard,device_gates+guard,device_sigmoid,
                device_gated+guard,invalid_rows)!=hipErrorInvalidValue)
                throw std::runtime_error("gate accepted invalid shape");
        for(auto* alias:{device_gates+guard,device_sigmoid})
            if(qrt_sm121_mtp::launch_gate(device_context+guard,device_gates+guard,device_sigmoid,
                alias,static_cast<unsigned>(rows))!=hipErrorInvalidValue)
                throw std::runtime_error("gate accepted immutable output alias");
        for(unsigned attempt=0;attempt<2u;++attempt) {
            uint16_t* destination=attempt?device_context:device_gated;
            check(qrt_sm121_mtp::launch_gate(device_context+guard,device_gates+guard,device_sigmoid,
                destination+guard,static_cast<unsigned>(rows)));
            check(hipDeviceSynchronize());
            check(hipMemcpy(gated.data(),destination,gated.size()*sizeof(uint16_t),hipMemcpyDeviceToHost));
            for(size_t i=0;i<context.size();++i)gated_bad+=gated[guard+i]!=expected_gated[i];
            for(size_t i=0;i<guard;++i) {
                guard_bad+=gated[i]!=sentinel;guard_bad+=gated[gated.size()-1u-i]!=sentinel;
            }
            if(!attempt) {
                std::vector<uint16_t> untouched(guarded_context.size());
                check(hipMemcpy(untouched.data(),device_context,untouched.size()*sizeof(uint16_t),hipMemcpyDeviceToHost));
                for(size_t i=0;i<untouched.size();++i)context_input_bad+=untouched[i]!=guarded_context[i];
            }
        }
        check(hipMemcpy(gates.data(),device_gates,gates.size()*sizeof(uint16_t),hipMemcpyDeviceToHost));
        std::vector<uint16_t> sigmoid_after(sigmoid.size());
        check(hipMemcpy(sigmoid_after.data(),device_sigmoid,sigmoid.size()*sizeof(uint16_t),hipMemcpyDeviceToHost));
        for(size_t i=0;i<sigmoid.size();++i)sigmoid_bad+=sigmoid_after[i]!=sigmoid[i];
    }
    for(size_t row=0;row<rows;++row)for(unsigned head=0;head<16u;++head)for(unsigned channel=0;channel<256u;++channel) {
        const size_t index=row*4096u+head*256u+channel;
        norm_bad+=norm[guard+index]!=expected_norm[index];
        rope_bad+=queries[guard+index]!=expected_rope[index];
        gate_bad+=gates[guard+index]!=input[row*8192u+head*512u+256u+channel];
    }
    for(const auto* values:{&norm,&queries,&gates})for(size_t i=0;i<guard;++i) {
        guard_bad+=(*values)[i]!=sentinel;guard_bad+=(*values)[values->size()-1u-i]!=sentinel;
    }
    std::vector<uint16_t> readback(guarded_input.size());
    check(hipMemcpy(readback.data(),device_input,readback.size()*sizeof(uint16_t),hipMemcpyDeviceToHost));
    for(size_t i=0;i<readback.size();++i)input_bad+=readback[i]!=guarded_input[i];
    const bool passed=!norm_bad&&!rope_bad&&!gate_bad&&!guard_bad&&!input_bad&&!gated_bad&&!context_input_bad&&!sigmoid_bad;
    std::cout<<"{\"kind\":\"original_mtp_query_native_replay\",\"rows\":"<<rows
        <<",\"launches\":"<<launches<<",\"elements_per_boundary\":"<<expected_norm.size()
        <<",\"norm_bf16_mismatches\":"<<norm_bad<<",\"rope_bf16_mismatches\":"<<rope_bad
        <<",\"gate_passthrough_mismatches\":"<<gate_bad<<",\"guard_mismatches\":"<<guard_bad
        <<",\"gated_context_checked\":"<<(check_gated?"true":"false")
        <<",\"gated_context_elements\":"<<context.size()*2u<<",\"gated_context_bf16_mismatches\":"<<gated_bad
        <<",\"context_input_mismatches\":"<<context_input_bad<<",\"sigmoid_table_mismatches\":"<<sigmoid_bad
        <<",\"invalid_gate_launch_cases\":"<<(check_gated?4u:0u)
        <<",\"input_mismatches\":"<<input_bad<<",\"invalid_launch_cases\":4,\"passed\":"
        <<(passed?"true":"false")<<",\"inference_acceptance\":false}\n";
    return passed?0:1;
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 2; }
