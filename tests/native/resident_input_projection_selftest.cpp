#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_resident_input_projection.h"
#include "../../native/providers/moe_accumulator/replay_weight_buckets.h"
#include "strong_float_replay_cases.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
namespace resident=qrt_sm121_resident_input_projection;
namespace buckets=qrt_replay_weight_buckets;
namespace staged=qrt_sm121_staged_half_projection;
namespace cases=qrt_strong_replay_cases;
constexpr unsigned guard=65u,marker=0xa5a5a5a5u;
void check(hipError_t s){if(s!=hipSuccess)throw std::runtime_error(hipGetErrorString(s));}
void require(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
struct Device {
    void* pointer=nullptr;
    explicit Device(size_t bytes){check(hipMalloc(&pointer,bytes));check(hipMemset(pointer,0xa5,bytes));}
    ~Device(){if(pointer)(void)hipFree(pointer);}
    template<class T>T* data(){return static_cast<T*>(pointer)+guard;}
};
template<class T>std::vector<T> read(Device& d,size_t count){std::vector<T> v(count+2u*guard);check(hipMemcpy(v.data(),d.pointer,v.size()*sizeof(T),hipMemcpyDeviceToHost));return v;}
template<class T>void guards(const std::vector<T>& v){const auto* b=reinterpret_cast<const unsigned char*>(v.data());for(size_t i=0u;i<guard*sizeof(T);++i)require(b[i]==0xa5u&&b[(v.size()-guard)*sizeof(T)+i]==0xa5u,"redzone changed");}
template<class T>void put(Device& d,const std::vector<T>& v){check(hipMemcpy(d.data<T>(),v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice));}
void finish(){hipEvent_t e;check(hipEventCreate(&e));check(hipEventRecord(e));const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);for(;;){auto s=hipEventQuery(e);if(s==hipSuccess)break;if(s!=hipErrorNotReady)check(s);require(std::chrono::steady_clock::now()<deadline,"completion deadline");std::this_thread::yield();}check(hipEventDestroy(e));}
void run(unsigned width){
    constexpr unsigned rows=257u,tokens=19u,cells=rows*tokens;const unsigned groups=width/16u;
    const size_t wg=size_t(rows)*groups,ig=size_t(tokens)*groups,trace_words=size_t(cells)*groups*3u;
    std::vector<uint16_t> weights(size_t(rows)*width),inputs(size_t(tokens)*width);
    for(unsigned r=0u;r<rows;++r)for(unsigned k=0u;k<width;++k)weights[size_t(r)*width+k]=cases::input(r+7u,k/16u,k%16u).y;
    for(unsigned t=0u;t<tokens;++t)for(unsigned k=0u;k<width;++k)inputs[size_t(t)*width+k]=cases::input(t,k/16u,k%16u).x;
    std::vector<staged::Row> packed_w(wg),packed_i(ig);
    for(size_t g=0u;g<wg;++g)packed_w[g]=staged::half::prepare(weights.data()+g*16u);
    for(size_t g=0u;g<ig;++g)packed_i[g]=staged::half::prepare(inputs.data()+g*16u);
    const unsigned lengths[]={0u,1u,63u,64u,65u,127u,128u,129u,257u,31u};
    std::vector<unsigned> selected,histogram(tokens,0u);std::vector<uint32_t> expected_trace(trace_words,marker),expected_output(cells,marker);
    std::vector<staged::Stats> expected_stats(cells);uint64_t transformed=0u,original=0u;
    for(unsigned t=0u;t<tokens;++t)for(unsigned s=0u;s<lengths[t%10u];++s){
        const unsigned r=(s*37u+t*13u)%rows,cell=t*rows+r;selected.push_back(cell);++histogram[t];
        const auto* a=inputs.data()+size_t(t)*width;const auto* b=weights.data()+size_t(r)*width;
        staged::Value carry{0u,-133,false};
        for(unsigned g=0u;g<groups;++g){qrt_q1_moe_hawkeye::Value terms[17];terms[0]=carry;
            for(unsigned i=0u;i<16u;++i)terms[i+1u]=qrt_q1_moe_hawkeye::multiply_bf16(a[g*16u+i],b[g*16u+i],-133);
            carry=qrt_q1_moe_hawkeye::group_sum<26,-133>(terms,17u);
            const size_t at=(size_t(cell)*groups+g)*3u;expected_trace[at]=carry.significand;expected_trace[at+1u]=uint32_t(int32_t(carry.exponent));expected_trace[at+2u]=unsigned(carry.negative);
            const bool valid=staged::half::unit(packed_i[size_t(t)*groups+g])!=-32768&&staged::half::unit(packed_w[size_t(r)*groups+g])!=-32768;
            expected_stats[cell].transformed+=valid;expected_stats[cell].original+=!valid;transformed+=valid;original+=!valid;
        }
        expected_output[cell]=cases::output_bits(carry);
    }
    std::mt19937 random(0x3958192u);std::shuffle(selected.begin(),selected.end(),random);
    const buckets::Plan plan{rows,tokens,rows,1u};const size_t workspace_words=buckets::workspace_words(plan);
    Device dw((weights.size()+2u*guard)*2u),di((inputs.size()+2u*guard)*2u),pw((wg+2u*guard)*sizeof(staged::Row)),pi((ig+2u*guard)*sizeof(staged::Row));
    Device ids((selected.size()+2u*guard)*4u),ordered((selected.size()+2u*guard)*4u),workspace((workspace_words+2u*guard)*4u);
    Device output((cells+2u*guard)*4u),trace((trace_words+2u*guard)*4u),stats((cells+2u*guard)*sizeof(staged::Stats));
    put(dw,weights);put(di,inputs);put(ids,selected);
    hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3(unsigned((wg+255u)/256u)),dim3(256u),0u,nullptr,dw.data<uint16_t>(),pw.data<staged::Row>(),rows,width);check(hipGetLastError());
    hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3(unsigned((ig+255u)/256u)),dim3(256u),0u,nullptr,di.data<uint16_t>(),pi.data<staged::Row>(),tokens,width);check(hipGetLastError());
    check(buckets::launch(ids.data<unsigned>(),selected.size(),unsigned(selected.size()),plan,ordered.data<unsigned>(),selected.size(),workspace.data<unsigned>(),workspace_words,nullptr));finish();
    const auto ws=read<unsigned>(workspace,workspace_words),permutation=read<unsigned>(ordered,selected.size());guards(ws);guards(permutation);
    auto* starts=workspace.data<unsigned>()+tokens;auto* status=workspace.data<unsigned>()+workspace_words-1u;
    require(ws[guard+workspace_words-1u]==0u,"bucketer status");auto sorted=selected,actual=std::vector<unsigned>(permutation.begin()+guard,permutation.end()-guard);std::sort(sorted.begin(),sorted.end());std::sort(actual.begin(),actual.end());require(actual==sorted,"candidate permutation");
    for(unsigned t=0u;t<tokens;++t){require(ws[guard+t]==histogram[t],"token histogram");const unsigned begin=ws[guard+tokens+t],end=ws[guard+tokens+t+1u];require(end-begin==histogram[t],"token span");for(unsigned at=begin;at<end;++at)require(permutation[guard+at]/rows==t,"token membership");}
    for(unsigned parts:{1u,2u}){
        auto launch=[&](bool audit,size_t weight_count=SIZE_MAX,size_t index_count=SIZE_MAX,size_t span_count=SIZE_MAX,unsigned partitions=0u){return resident::launch(pw.data<staged::Row>(),weight_count==SIZE_MAX?wg:weight_count,pi.data<staged::Row>(),ig,ordered.data<unsigned>(),index_count==SIZE_MAX?selected.size():index_count,unsigned(selected.size()),starts,span_count==SIZE_MAX?tokens+1u:span_count,status,output.data<float>(),cells,rows,tokens,width,partitions?partitions:parts,nullptr,audit?trace.data<uint32_t>():nullptr,audit?trace_words:0u,audit?stats.data<staged::Stats>():nullptr,audit?cells:0u);};
        require(launch(false,wg-1u)==hipErrorInvalidValue,"short weight span accepted");require(launch(false,SIZE_MAX,selected.size()-1u)==hipErrorInvalidValue,"short index span accepted");require(launch(false,SIZE_MAX,SIZE_MAX,tokens)==hipErrorInvalidValue,"short token span accepted");require(launch(false,SIZE_MAX,SIZE_MAX,SIZE_MAX,3u)==hipErrorInvalidValue,"bad partitions accepted");
        check(hipMemset(output.pointer,0xa5,(cells+2u*guard)*4u));check(hipMemset(trace.pointer,0xa5,(trace_words+2u*guard)*4u));check(hipMemset(stats.pointer,0xa5,(cells+2u*guard)*sizeof(staged::Stats)));
        check(launch(true));finish();const auto got=read<uint32_t>(output,cells),raw=read<uint32_t>(trace,trace_words);const auto counters=read<staged::Stats>(stats,cells);guards(got);guards(raw);guards(counters);
        require(!std::memcmp(got.data()+guard,expected_output.data(),cells*4u),"selected endpoint or unselected output changed");require(!std::memcmp(raw.data()+guard,expected_trace.data(),trace_words*4u),"original CPU raw K16 state or unselected trace differs");
        for(unsigned cell:selected)require(!std::memcmp(&counters[guard+cell],&expected_stats[cell],sizeof(staged::Stats)),"arithmetic counts differ");
        for(unsigned cell=0u;cell<cells;++cell)if(!std::binary_search(sorted.begin(),sorted.end(),cell))require(counters[guard+cell].transformed==marker&&counters[guard+cell].original==marker,"unselected statistics changed");
        check(hipMemset(output.pointer,0xa5,(cells+2u*guard)*4u));check(launch(false));finish();require(read<uint32_t>(output,cells)==got,"production and audit output differ");require(read<uint32_t>(trace,trace_words)==raw,"production changed trace");const auto after_stats=read<staged::Stats>(stats,cells);require(!std::memcmp(after_stats.data(),counters.data(),counters.size()*sizeof(staged::Stats)),"production changed statistics");
        require(read<unsigned>(workspace,workspace_words)==ws&&read<unsigned>(ordered,selected.size())==permutation,"replay changed bins or candidates");
        // Corrupt the first nonempty span, then a selected identity. Both are
        // device faults: no invalid index or weight access is permitted.
        const unsigned bad_end=unsigned(selected.size()+1u),saved_end=ws[guard+tokens+2u];check(hipMemcpy(starts+2u,&bad_end,4u,hipMemcpyHostToDevice));check(launch(false));finish();require(read<unsigned>(workspace,workspace_words)[guard+workspace_words-1u]&16u,"malformed span not reported");check(hipMemcpy(starts+2u,&saved_end,4u,hipMemcpyHostToDevice));check(hipMemset(status,0,4u));
        const unsigned bad_cell=UINT32_MAX,saved_cell=permutation[guard];check(hipMemcpy(ordered.data<unsigned>(),&bad_cell,4u,hipMemcpyHostToDevice));check(launch(false));finish();require(read<unsigned>(workspace,workspace_words)[guard+workspace_words-1u]&32u,"wrong-token candidate not reported");check(hipMemcpy(ordered.data<unsigned>(),&saved_cell,4u,hipMemcpyHostToDevice));check(hipMemset(status,0,4u));
        guards(read<uint32_t>(output,cells));require(transformed&&original,"missing arithmetic path coverage");
        std::printf("{\"kind\":\"resident_input_projection_safety\",\"parts\":%u,\"rows\":%u,\"tokens\":%u,\"width\":%u,\"candidates\":%zu,\"ordered_raw_carry_states\":%zu,\"transformed_groups\":%llu,\"original_groups\":%llu,\"raw_bit_mismatches\":0,\"production_diagnostic_parity\":true,\"unselected_storage_unchanged\":true,\"empty_and_partial_batches\":true,\"device_fault_guards\":true,\"redzones_pass\":true,\"inference_acceptance\":false}\n",parts,rows,tokens,width,selected.size(),selected.size()*groups,(unsigned long long)transformed,(unsigned long long)original);std::fflush(stdout);
    }
    const auto aw=read<uint16_t>(dw,weights.size()),ai=read<uint16_t>(di,inputs.size());const auto before_ids=read<unsigned>(ids,selected.size());guards(aw);guards(ai);guards(before_ids);require(!std::memcmp(aw.data()+guard,weights.data(),weights.size()*2u)&&!std::memcmp(ai.data()+guard,inputs.data(),inputs.size()*2u)&&!std::memcmp(before_ids.data()+guard,selected.data(),selected.size()*4u),"source changed");
    const auto ew=read<staged::Row>(pw,wg),ei=read<staged::Row>(pi,ig);guards(ew);guards(ei);require(!std::memcmp(ew.data()+guard,packed_w.data(),wg*sizeof(staged::Row))&&!std::memcmp(ei.data()+guard,packed_i.data(),ig*sizeof(staged::Row)),"prepared encoding or immutable operands differ");
}
int main()try{hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));require(!std::strncmp(p.gcnArchName,"gfx1151",7u),"requires gfx1151");for(unsigned width:{16u,272u,2048u,4096u,4112u,8192u})run(width);return 0;}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
