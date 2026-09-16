// Reuse the validated guarded Q/K loader, original CPU oracle, preparation,
// completion deadline and exact byte observers. Its original main is unused.
#define main qrt_prior_fused_probability_main
#include "fused_qk_probability_selftest.cpp"
#undef main
#include "../../native/providers/ck_fmha/native_delta_probability.h"
#include <array>
#define QRT_CPU_QK_NO_MAIN
#include "cpu_exact_qk_host.cpp"
#undef QRT_CPU_QK_NO_MAIN

namespace {
namespace delta = qrt_sm121_exp2_native_delta;
constexpr unsigned query_batch=128u;
struct Guarded {
    size_t bytes;
    Device storage;
    explicit Guarded(size_t n):bytes(n),storage(n+256u){reset();}
    unsigned char* data(){return storage.as<unsigned char>()+128u;}
    template<class T>T* as(){return reinterpret_cast<T*>(data());}
    void reset(){check(hipMemset(storage.pointer,0xa5,bytes+256u));}
    void guards(){
        unsigned char result[256];
        check(hipMemcpy(result,storage.pointer,128u,hipMemcpyDeviceToHost));
        check(hipMemcpy(result+128u,data()+bytes,128u,hipMemcpyDeviceToHost));
        for(auto b:result)if(b!=0xa5u)throw std::runtime_error("attention component guard changed");
    }
    template<class T>void put(const std::vector<T>& values){
        if(values.size()*sizeof(T)!=bytes)throw std::runtime_error("upload extent mismatch");
        check(hipMemcpy(data(),values.data(),bytes,hipMemcpyHostToDevice));
    }
    template<class T>void immutable(const std::vector<T>& values){
        if(values.size()*sizeof(T)!=bytes)throw std::runtime_error("immutable extent mismatch");
        std::vector<T> actual(values.size());check(hipMemcpy(actual.data(),data(),bytes,hipMemcpyDeviceToHost));
        if(std::memcmp(values.data(),actual.data(),bytes))throw std::runtime_error("attention source changed");
        guards();
    }
};
struct AttentionOutputs {
    Outputs tensor;
    Guarded output,accumulator,denominator,error,indices,count;
    AttentionOutputs(unsigned tokens):tensor(query_batch,tokens),
        output(size_t(query_batch)*4096u*4u),accumulator(output.bytes),
        denominator(size_t(query_batch)*16u*4u),error(output.bytes),indices(output.bytes),count(4u){}
    void reset(){tensor.reset();for(auto* p:{&output,&accumulator,&denominator,&error,&indices,&count})p->reset();}
    void guards(){for(auto* p:{&output,&accumulator,&denominator,&error,&indices,&count})p->guards();}
};
struct Events {
    hipEvent_t before{},after{};
    Events(){check(hipEventCreate(&before));check(hipEventCreate(&after));}
    ~Events(){(void)hipEventDestroy(after);(void)hipEventDestroy(before);}
};
void attention(const uint16_t* q,const uint16_t* transposed_key,const uint16_t* v,
    const uint16_t* transposed_value,Prepared& prepared,AttentionOutputs& out,
    unsigned start,unsigned queries,unsigned tokens,const unsigned char* exp2,
    const unsigned char* packed,const unsigned char* rcp,bool original_scores,Events* probability_events,const float* cpu_scores=nullptr,double* copy_ms=nullptr){
    const unsigned stride=start+queries;
    auto* scores=out.tensor.scores.as<float>()+guard;
    auto* probabilities=out.tensor.probability.as<uint16_t>()+guard;
    auto* scales=out.tensor.scales.as<float>()+guard;
    if(cpu_scores){
        const auto begin=std::chrono::steady_clock::now();
        check(hipMemcpy(scores,cpu_scores,size_t(queries)*kQueryHeads*stride*4u,hipMemcpyHostToDevice));
        if(copy_ms)*copy_ms=elapsed(begin);
    }else if(original_scores){
        hipLaunchKernelGGL(blackwell_tiled_exact_scores_kernel,
            dim3((stride+31u)/32u,kQueryHeads,(queries+7u)/8u),dim3(kThreads),0u,nullptr,
            q,transposed_key,scores,start,queries,stride,tokens);
    }else{
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::scores<128u,true>),
            dim3((stride+15u)/16u,kQueryHeads,(queries+15u)/16u),dim3(kThreads),0u,nullptr,
            q,transposed_key,prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,
            prepared.qf.as<unsigned>()+guard,prepared.kf.as<unsigned>()+guard,scores,start,queries,stride,tokens);
    }
    check(hipGetLastError());
    if(probability_events)check(hipEventRecord(probability_events->before));
    if(packed){
        hipLaunchKernelGGL(qrt_native_delta_probability::probabilities,dim3(kQueryHeads,queries),dim3(32u),0u,nullptr,
            scores,probabilities,scales,start,stride,exp2,true,packed);
    }else{
        hipLaunchKernelGGL(blackwell_online_probability_kernel,dim3(kQueryHeads,queries),dim3(32u),0u,nullptr,
            scores,probabilities,scales,start,stride,exp2,true);
    }
    check(hipGetLastError());
    if(probability_events)check(hipEventRecord(probability_events->after));
    hipLaunchKernelGGL((blackwell_mantissa_value_kernel<true,false,true,true,true>),
        dim3(kHeadDim/kIntegerMatrixColumns,kQueryHeads,(queries+15u)/16u),dim3(kThreads),0u,nullptr,
        v,probabilities,scales,out.output.as<float>(),start,queries,0u,stride,rcp,
        out.accumulator.as<float>(),out.denominator.as<float>(),nullptr,nullptr,out.error.as<float>());
    check(hipGetLastError());
    check(hipError_t(launch_compacted_pv_replay(v,probabilities,scales,out.output.as<float>(),start,queries,0u,stride,
        rcp,out.accumulator.as<float>(),out.denominator.as<float>(),out.error.as<float>(),
        out.indices.as<unsigned>(),out.count.as<unsigned>(),nullptr,nullptr,transposed_value,tokens)));
}
__global__ void external_context(const float* output,const uint16_t* reference,unsigned start,unsigned queries,unsigned* bad){
    const unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<queries*4096u){
        const float value=output[index];
        if(!isfinite(value)||(size_t(start)*4096u+index<7169u*4096u&&f32_to_bf16(value)!=reference[size_t(start)*4096u+index]))atomicAdd(bad,1u);
    }
}
__global__ void tensor_tails(const uint32_t* scores,const uint16_t* probability,const uint32_t* scales,
    size_t capacity,size_t scale_capacity,unsigned start,unsigned queries,unsigned stride,unsigned* bad){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t live=size_t(queries)*16u*stride;
    const unsigned tiles=(stride+31u)/32u;
    if(i<capacity+2u*guard){
        if(i<guard||i>=guard+live){if(scores[i]!=0xa5a5a5a5u||probability[i]!=0xa5a5u)atomicAdd(bad,1u);}
        else{
            const unsigned row=unsigned((i-guard)/stride),key=unsigned((i-guard)%stride),causal=start+row/16u+1u;
            if(key>=causal&&scores[i]!=0xff800000u)atomicAdd(bad,1u);
            if(key>=((causal+31u)/32u)*32u&&probability[i]!=0xa5a5u)atomicAdd(bad,1u);
        }
    }
    if(i<scale_capacity+2u*guard){
        if(i<guard||i>=guard+size_t(queries)*16u*(tiles+1u)){if(scales[i]!=0xa5a5a5a5u)atomicAdd(bad,1u);}
        else{
            const unsigned row=unsigned((i-guard)/(tiles+1u)),tile=unsigned((i-guard)%(tiles+1u));
            if(tile<tiles&&tile>=(start+row/16u+32u)/32u&&scales[i]!=0xa5a5a5a5u)atomicAdd(bad,1u);
        }
    }
}
void compare(AttentionOutputs& expected,AttentionOutputs& actual,Device& bad){
    for(unsigned surface=0;surface<3u;++surface){
        auto& a=surface==0u?expected.tensor.scores:surface==1u?expected.tensor.probability:expected.tensor.scales;
        auto& b=surface==0u?actual.tensor.scores:surface==1u?actual.tensor.probability:actual.tensor.scales;
        const size_t bytes=surface==2u?(expected.tensor.scale_cells+2u*guard)*4u:(expected.tensor.cells+2u*guard)*(surface==1u?2u:4u);
        hipLaunchKernelGGL(compare_words,dim3((bytes+255u)/256u),dim3(256u),0u,nullptr,a.as<unsigned char>(),b.as<unsigned char>(),bytes,bad.as<unsigned>());
        check(hipGetLastError());
    }
    for(auto pair:{std::make_pair(&expected.output,&actual.output),std::make_pair(&expected.accumulator,&actual.accumulator),
        std::make_pair(&expected.denominator,&actual.denominator),std::make_pair(&expected.error,&actual.error),std::make_pair(&expected.count,&actual.count)}){
        const size_t bytes=pair.first->bytes+256u;
        hipLaunchKernelGGL(compare_words,dim3((bytes+255u)/256u),dim3(256u),0u,nullptr,
            pair.first->storage.as<unsigned char>(),pair.second->storage.as<unsigned char>(),bytes,bad.as<unsigned>());
        check(hipGetLastError());
    }
    finish();
    if(download<unsigned>(bad,1u)[0])throw std::runtime_error("CPU/GPU complete attention differs");
    actual.guards();
}
std::vector<unsigned char> read_table(const char* path,size_t bytes){
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streamoff(bytes))throw std::runtime_error("table size mismatch");
    std::vector<unsigned char> out(bytes);f.seekg(0);
    if(!f.read(reinterpret_cast<char*>(out.data()),bytes))throw std::runtime_error("table read failed");return out;
}
} // namespace

#if !defined(__HIP_DEVICE_COMPILE__)
namespace {
namespace cpu=qrt_cpu_exact_qk;

void cpu_capture(unsigned tokens,const char* qfile,const char* kfile,const char* vfile,const char* reference_file,
    const char* exp_file,const char* rcp_file){
    if(!cpu::features().supported())throw std::runtime_error("AVX512F/CD and OS ZMM support required");
    auto q=read_words(qfile,7169u*4096u),k=read_words(kfile,7169u*512u),v=read_words(vfile,7169u*512u);
    const auto reference=read_words(reference_file,7169u*4096u);
    for(auto pair:{std::make_pair(&q,4096u),std::make_pair(&k,512u),std::make_pair(&v,512u)}){
        const auto old=*pair.first;
        pair.first->insert(pair.first->end(),old.begin(),old.begin()+size_t(tokens-7169u)*pair.second);
    }
    const auto exp=read_table(exp_file,delta::source::table_bytes),rcp=read_table(rcp_file,qrt_sm121_attention_rcp::table_bytes);
    if(!delta::source::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("table layout mismatch");
    Guarded dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),dr(reference.size()*2u),de(exp.size()),dc(rcp.size());
    dq.put(q);dk.put(k);dv.put(v);dr.put(reference);de.put(exp);dc.put(rcp);
    Guarded dt(k.size()*2u),vt(v.size()*2u);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
    Prepared prepared(dq.as<uint16_t>(),dk.as<uint16_t>(),dt.as<uint16_t>(),q.data(),k.data(),tokens);
    const auto transpose_begin=std::chrono::steady_clock::now();
    check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),tokens,nullptr)));finish();
    const double transpose_ms=elapsed(transpose_begin);
    std::vector<uint16_t> transposed(k.size()),transposed_v(v.size());
    for(unsigned token=0;token<tokens;++token)for(unsigned feature=0;feature<512u;++feature){
        transposed[size_t(feature)*tokens+token]=k[size_t(token)*512u+feature];
        transposed_v[size_t(feature)*tokens+token]=v[size_t(token)*512u+feature];
    }
    // Read CPU operands from this run's device tensors. Neither the golden
    // context nor baseline scores enter the CPU producer or its scheduling.
    std::vector<uint16_t> host_q(q.size()),host_k(k.size());
    const auto input_begin=std::chrono::steady_clock::now();
    check(hipMemcpy(host_q.data(),dq.data(),dq.bytes,hipMemcpyDeviceToHost));
    check(hipMemcpy(host_k.data(),dk.data(),dk.bytes,hipMemcpyDeviceToHost));
    const double readback_ms=elapsed(input_begin);
    const auto prepare_begin=std::chrono::steady_clock::now();
    cpu::Prepared host_prepared(host_q.data(),host_k.data(),tokens);
    const double cpu_prepare_ms=elapsed(prepare_begin);
    const auto worker16_begin=std::chrono::steady_clock::now();cpu::Workers workers16(16u);
    workers16.run(0u,[](size_t){});const double workers16_ms=elapsed(worker16_begin);
    const auto worker32_begin=std::chrono::steady_clock::now();cpu::Workers workers32(32u);
    workers32.run(0u,[](size_t){});const double workers32_ms=elapsed(worker32_begin);
    AttentionOutputs expected(tokens),actual(tokens);
    std::vector<float> host_scores(expected.tensor.cells+128u,qrt_sm121_float_alignment::from_bits(0xa5a5a5a5u));
    double samples[3][3]{},cpu_ms[3][3]{},transfer_ms[3][3]{};
    uint64_t score_slots=0u,scale_slots=0u,pv_candidates[3]{},fallbacks[3]{};unsigned cpu_dots=0u;
    for(unsigned start=0;start<tokens;start+=query_batch){
        const unsigned queries=std::min(query_batch,tokens-start),stride=start+queries;
        expected.reset();
        attention(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,expected,start,queries,tokens,de.data(),nullptr,dc.data(),true,nullptr);finish();
        expected.guards();
        hipLaunchKernelGGL(tensor_tails,dim3((expected.tensor.cells+2u*guard+255u)/256u),dim3(256u),0u,nullptr,
            expected.tensor.scores.as<uint32_t>(),expected.tensor.probability.as<uint16_t>(),expected.tensor.scales.as<uint32_t>(),
            expected.tensor.cells,expected.tensor.scale_cells,start,queries,stride,bad.as<unsigned>());
        check(hipGetLastError());
        hipLaunchKernelGGL(external_context,dim3((queries*4096u+255u)/256u),dim3(256u),0u,nullptr,expected.output.as<float>(),dr.as<uint16_t>(),start,queries,bad.as<unsigned>());
        check(hipGetLastError());finish();
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("original context differs from GB10");
        score_slots+=size_t(queries)*16u*stride;scale_slots+=size_t(queries)*16u*((stride+31u)/32u+1u);
        for(unsigned attempt=0;attempt<4u;++attempt)for(unsigned position=0;position<3u;++position){
            const unsigned variant=(position+start/query_batch+attempt)%3u;
            actual.reset();std::fill(host_scores.begin(),host_scores.end(),qrt_sm121_float_alignment::from_bits(0xa5a5a5a5u));finish();
            const auto begin=std::chrono::steady_clock::now();double score_ms=0.0,copy_ms=0.0;uint64_t fallback=0;
            if(variant){
                fallback=cpu::scores(host_prepared,variant==1u?workers16:workers32,start,queries,host_scores.data()+64u,expected.tensor.cells);
                score_ms=elapsed(begin);
            }
            attention(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,actual,start,queries,tokens,de.data(),nullptr,dc.data(),false,nullptr,
                variant?host_scores.data()+64u:nullptr,&copy_ms);finish();
            const double ms=elapsed(begin);
            if(attempt){samples[variant][attempt-1u]+=ms;cpu_ms[variant][attempt-1u]+=score_ms;transfer_ms[variant][attempt-1u]+=copy_ms;}
            compare(expected,actual,bad);
            if(variant)for(size_t i=0;i<host_scores.size();++i)if(i<64u||i>=64u+size_t(queries)*16u*stride)
                if(bits(host_scores[i])!=0xa5a5a5a5u)throw std::runtime_error("CPU output redzone or unused tail changed");
            if(!attempt){unsigned count=0;check(hipMemcpy(&count,actual.count.data(),4u,hipMemcpyDeviceToHost));pv_candidates[variant]+=count;fallbacks[variant]+=fallback;}
        }
        for(unsigned sample=0;sample<4u;++sample){
            const unsigned row=sample*(queries-1u)/3u,head=(start/query_batch+sample*5u)%16u,key=(start+row)*sample/3u;
            const float value=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
                q.data()+(size_t(start+row)*16u+head)*256u,k.data()+(size_t(key)*2u+head/8u)*256u,256u)*kExactScale;
            uint32_t gpu=0;check(hipMemcpy(&gpu,expected.tensor.scores.as<uint32_t>()+guard+(size_t(row)*16u+head)*stride+key,4u,hipMemcpyDeviceToHost));
            if(gpu!=bits(value))throw std::runtime_error("original QK differs from CPU");++cpu_dots;
        }
    }
    dq.immutable(q);dk.immutable(k);dv.immutable(v);dr.immutable(reference);de.immutable(exp);dc.immutable(rcp);
    dt.immutable(transposed);vt.immutable(transposed_v);prepared.verify();host_prepared.verify();
    if(host_q!=q||host_k!=k)throw std::runtime_error("CPU original input changed");
    if(pv_candidates[0]!=pv_candidates[1]||pv_candidates[0]!=pv_candidates[2]||fallbacks[1]!=fallbacks[2])throw std::runtime_error("CPU control counters differ");
    for(unsigned variant=0;variant<3u;++variant){
        auto sorted=std::vector<double>(samples[variant],samples[variant]+3u);std::sort(sorted.begin(),sorted.end());
        const unsigned workers=variant==1u?16u:variant==2u?32u:0u;
        const double startup=variant==1u?workers16_ms:variant==2u?workers32_ms:0.0;
        const double preparation=transpose_ms+(variant?readback_ms+cpu_prepare_ms+startup:prepared.ms);
        std::printf("{\"kind\":\"cpu_exact_qk_attention_capture\",\"tokens\":%u,\"source_capture_tokens\":7169,\"repeated_rows\":%u,\"cpu_workers\":%u,\"score_probability_slots\":%llu,\"scale_slots\":%llu,\"output_cells\":%u,\"gb10_context_cells\":29364224,\"independent_cpu_dots\":%u,\"pv_candidates\":%llu,\"cpu_fallback_scores\":%llu,\"completed_attention_samples_ms\":[%.9f,%.9f,%.9f],\"median_completed_attention_ms\":%.9f,\"cpu_score_samples_ms\":[%.9f,%.9f,%.9f],\"score_upload_samples_ms\":[%.9f,%.9f,%.9f],\"value_transpose_ms\":%.9f,\"gpu_qk_prepare_ms\":%.9f,\"cpu_input_readback_ms\":%.9f,\"cpu_encoding_ms\":%.9f,\"cpu_worker_startup_ms\":%.9f,\"candidate_preparation_ms\":%.9f,\"preparation_plus_median_ms\":%.9f,\"cpu_encoding_bytes\":%llu,\"raw_bit_mismatches\":0,\"gb10_context_mismatches\":0,\"all_attempts_checked\":true,\"warmups_per_slab\":1,\"timed_attempts_per_slab\":3,\"original_probability_and_pv\":true,\"cpu_gpu_overlap\":false,\"reference_is_compute_input\":false,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",tokens,tokens-7169u,workers,(unsigned long long)score_slots,(unsigned long long)scale_slots,tokens*4096u,cpu_dots,(unsigned long long)pv_candidates[variant],(unsigned long long)fallbacks[variant],samples[variant][0],samples[variant][1],samples[variant][2],sorted[1],cpu_ms[variant][0],cpu_ms[variant][1],cpu_ms[variant][2],transfer_ms[variant][0],transfer_ms[variant][1],transfer_ms[variant][2],transpose_ms,variant?0.0:prepared.ms,variant?readback_ms:0.0,variant?cpu_prepare_ms:0.0,startup,preparation,preparation+sorted[1],(unsigned long long)(variant?host_prepared.bytes():0u));std::fflush(stdout);
    }
}
} // namespace
int main(int argc,char** argv)try{
    if(argc==2&&!std::strcmp(argv[1],"--selftest")){cpu_qk_tests::portable();cpu_qk_tests::native();return 0;}
    if(argc!=8)throw std::runtime_error("usage: cpu-capture 7169|8192 Q K V GB10_CONTEXT EXP2 RCP | --selftest");
    const unsigned tokens=!std::strcmp(argv[1],"7169")?7169u:!std::strcmp(argv[1],"8192")?8192u:0u;
    if(!tokens)throw std::runtime_error("invalid capture shape");
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));
    if(std::strncmp(prop.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    cpu_capture(tokens,argv[2],argv[3],argv[4],argv[5],argv[6],argv[7]);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"cpu_exact_qk_capture_error=%s\n",e.what());return 2;}
#endif
