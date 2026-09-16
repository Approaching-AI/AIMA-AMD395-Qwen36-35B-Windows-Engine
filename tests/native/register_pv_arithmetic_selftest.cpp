#define main qrt_prior_coarse_pv_main
#include "coarse_pv_transfer_selftest.cpp"
#undef main
#include "../../native/providers/ck_fmha/register_pv_arithmetic.h"

namespace {
namespace arithmetic = qrt_register_pv;
struct ArithmeticBuffer {
    size_t bytes;Device device;
    explicit ArithmeticBuffer(size_t n):bytes(n),device(n+512u){reset();}
    void reset(){check(hipMemset(device.pointer,0xa5,bytes+512u));}
    template<class T>T* data(){return reinterpret_cast<T*>(device.as<unsigned char>()+256u);}
    void verify_guards(){
        unsigned char data[512];check(hipMemcpy(data,device.pointer,256u,hipMemcpyDeviceToHost));
        check(hipMemcpy(data+256u,device.as<unsigned char>()+256u+bytes,256u,hipMemcpyDeviceToHost));
        for(auto b:data)if(b!=0xa5u)throw std::runtime_error("register PV redzone");
    }
};

__global__ void rounding_compare(unsigned count,unsigned* bad) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    uint32_t r=(i+3u)*747796405u;r=(r^(r>>16u))*2246822519u;
    const uint32_t edges[]={0u,0x80000000u,1u,0x80000001u,0x007fffffu,0x00800000u,
        0x3f800000u,0xbf800000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0x7fc12345u};
    const float x=__uint_as_float(i%3u?r:edges[(i/3u)%12u]);
    const uint32_t abits=i%7u?r%0x3f800001u:((i/7u)%3u==0u?0u:(i/7u)%3u==1u?0x3f800000u:1u);
    const float alpha=__uint_as_float(abits);
    volatile float expected=x*alpha;
    const float actual=qrt_sm121_pv_final_bound::multiply(x,alpha);
    if(__float_as_uint(expected)!=__float_as_uint(actual))atomicAdd(bad,1u);
}
void rounding_safety(){
    constexpr unsigned count=1048576u;Device bad(4u);check(hipMemset(bad.pointer,0,4u));
    hipLaunchKernelGGL(rounding_compare,dim3(count/256u),dim3(256u),0u,nullptr,count,bad.as<unsigned>());check(hipGetLastError());finish();
    if(download<unsigned>(bad,1u)[0])throw std::runtime_error("register rounding instruction differs");
    std::printf("{\"kind\":\"register_pv_rounding\",\"comparisons\":1048576,\"raw_mismatches\":0}\n");std::fflush(stdout);
}
void arithmetic_run(unsigned variant,Stage& s,Device& ids,ArithmeticBuffer& reciprocals,
    const uint16_t* v,const uint16_t* tv,unsigned n,const unsigned char* rcp,bool only_produce=false){
    const auto& q=s.shape;auto* indices=ids.as<unsigned>()+guard;auto* count=indices+q.cells();
    if(!variant){
        hipLaunchKernelGGL((blackwell_mantissa_value_kernel<true,false,true,true,true>),
            dim3(2u,16u,(q.queries+15u)/16u),dim3(256u),0u,nullptr,v,s.p.as<uint16_t>()+guard,s.s.as<float>()+guard,
            s.o.as<float>()+guard,q.start,q.queries,q.output_start,q.stride,rcp,s.a.as<float>()+guard,s.d.as<float>()+guard,nullptr,nullptr,s.e.as<float>()+guard);
        check(hipGetLastError());
        if(!only_produce){check(hipMemsetAsync(count,0,4u,nullptr));queue_collect(s,ids);queue_exact(s,ids,v,tv,n,rcp);}return;
    }
    if(only_produce){
        if(variant==3u){hipLaunchKernelGGL(arithmetic::prepare_reciprocals,dim3((q.queries*16u+255u)/256u),dim3(256u),0u,nullptr,
            s.s.as<float>()+guard,reciprocals.data<float>(),q.queries,q.stride,rcp);check(hipGetLastError());}
#define ARITH_PRODUCE(R,S) hipLaunchKernelGGL((arithmetic::native_value<R,S>),dim3(2u,16u,(q.queries+15u)/16u),dim3(256u),0u,nullptr, \
 v,s.p.as<uint16_t>()+guard,s.s.as<float>()+guard,s.o.as<float>()+guard,q.start,q.queries,q.output_start,q.stride,rcp, \
 s.a.as<float>()+guard,s.d.as<float>()+guard,nullptr,nullptr,s.e.as<float>()+guard,reciprocals.data<float>())
        if(variant==1u){ARITH_PRODUCE(false,false);}else if(variant==2u){ARITH_PRODUCE(true,false);}else if(variant==3u){ARITH_PRODUCE(true,true);}else throw std::runtime_error("arithmetic variant");
#undef ARITH_PRODUCE
        check(hipGetLastError());return;
    }
#define ARITH_LAUNCH(R,S) check(hipError_t(arithmetic::launch<R,S>(v,s.p.as<uint16_t>()+guard,s.s.as<float>()+guard,s.o.as<float>()+guard, \
 s.e.as<float>()+guard,q.start,q.queries,q.output_start,q.stride,rcp,tv,n,indices,count,size_t(q.capacity)*4096u, \
 reciprocals.data<float>(),reciprocals.bytes/4u,nullptr,s.a.as<float>()+guard,s.d.as<float>()+guard)))
    if(variant==1u){ARITH_LAUNCH(false,false);}else if(variant==2u){ARITH_LAUNCH(true,false);}else if(variant==3u){ARITH_LAUNCH(true,true);}else throw std::runtime_error("arithmetic variant");
#undef ARITH_LAUNCH
}
void raw_surfaces(Stage& expected,Stage& actual,Device& bad){
    const auto& q=actual.shape;check(hipMemset(bad.pointer,0,4u));
    compare_device(expected.o,actual.o,q.owords(),bad);compare_device(expected.a,actual.a,q.owords(),bad);
    compare_device(expected.d,actual.d,q.dwords(),bad);compare_device(expected.e,actual.e,q.ewords(),bad);finish();
    if(download<unsigned>(bad,1u)[0])throw std::runtime_error("register PV raw surfaces differ");
    stage_guards(actual);
}
void reciprocal_observer(ArithmeticBuffer& buffer,const Shape& q,const std::vector<float>& scales,
    const unsigned char* table,bool used){
    std::vector<uint32_t> words(buffer.bytes/4u);check(hipMemcpy(words.data(),buffer.data<unsigned>(),buffer.bytes,hipMemcpyDeviceToHost));
    const unsigned tiles=(q.stride+31u)/32u;
    for(size_t i=0;i<words.size();++i){
        const uint32_t expected=used&&i<q.queries*16u?bits(qrt_sm121_attention_rcp::evaluate(table,scales[guard+i*(tiles+1u)+tiles])):0xa5a5a5a5u;
        if(words[i]!=expected)throw std::runtime_error("shared reciprocal or unused tail differs");
    }
    buffer.verify_guards();
}
__global__ void generated_result(const float* native,const float* native_raw,const float* errors,
    const float* exact,const float* exact_raw,const float* actual,const float* actual_raw,
    unsigned cells,unsigned output_start,unsigned* stats){
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=cells)return;
    const unsigned i=output_start*4096u+cell;
    const bool selected=!qrt_sm121_pv_bound::same_bf16(native[i],errors[cell]);
    if(__float_as_uint(actual[i])!=__float_as_uint(selected?exact[i]:native[i])||
        __float_as_uint(actual_raw[i])!=__float_as_uint(selected?exact_raw[i]:native_raw[i])||
        qrt_sm121_pv_bound::bf16(actual[i])!=qrt_sm121_pv_bound::bf16(exact[i]))atomicAdd(stats,1u);
    if(selected)atomicAdd(stats+1u,1u);
}
void arithmetic_generated(unsigned start,unsigned queries,unsigned mode,bool vllm,
    const unsigned char* exp,const unsigned char* rcp,const unsigned char* host_rcp){
    Shape q{start,queries,start+queries,queries+3u,start+queries+7u,3u};
    std::vector<float> scores(q.pwords(),12345.0f);
    std::vector<uint16_t> values(size_t(q.stride)*512u+2u*guard,0x5a5au),tv(values.size(),0x5a5au);
    for(unsigned row=0;row<queries*16u;++row)for(unsigned key=0;key<start+row/16u+1u;++key)
        scores[guard+size_t(row)*q.stride+key]=score_value(row,key,mode);
    for(unsigned key=0;key<q.stride;++key)for(unsigned feature=0;feature<512u;++feature){
        const size_t i=size_t(key)*512u+feature;
        uint16_t word=qrt_sm121_pv_bound::bf16(float(int((i*173u+i/17u)%63u)-31)/64.0f);
        if(mode==1u&&i%7u==0u)word=uint16_t(i%127u+1u);
        if(mode==2u&&i%3u==0u)word=uint16_t((i&1u?0x8000u:0u)|0x5f80u);
        if(mode==3u&&i%3u==0u)word=(i&1u)?0x8000u:0u;
        if(mode==4u)word=uint16_t((i&1u?0x8000u:0u)|(i%3u?0x0101u:0x0001u));
        if(mode==5u&&i%251u==0u)word=(i&1u)?0xffc1u:0x7fc1u;
        values[guard+i]=word;tv[guard+size_t(feature)*q.stride+key]=word;
    }
    Device ds(scores.size()*4u),dv(values.size()*2u),dvt(tv.size()*2u),bad(4u),stats(9u*8u);
    upload(ds,scores);upload(dv,values);upload(dvt,tv);
    Stage original(q),native(q),candidate(q);Parts unused(q);
    Device ids((size_t(q.capacity)*4096u+1u+2u*guard)*4u);
    hipLaunchKernelGGL(blackwell_online_probability_kernel,dim3(16u,q.queries),dim3(32u),0u,nullptr,
        ds.as<float>()+guard,original.p.as<uint16_t>()+guard,original.s.as<float>()+guard,
        q.start,q.stride,exp,vllm);check(hipGetLastError());
    copy_probability(native,original);coarse_produce(0u,native,unused,dv.as<uint16_t>()+guard,rcp);
    reset_indices(ids,q);hipLaunchKernelGGL(all_indices,dim3((q.cells()+255u)/256u),dim3(256u),0u,nullptr,
        ids.as<unsigned>()+guard,ids.as<unsigned>()+guard+q.cells(),q.cells());check(hipGetLastError());
    queue_exact(original,ids,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,q.stride,rcp);finish();
    verify_indices(ids,q);
    const auto raw=download<float>(original.a,q.owords());
    const auto hp=download<uint16_t>(original.p,q.pwords());const auto hs=download<float>(original.s,q.swords());
    unsigned cpu=0;
    for(unsigned sample=0;sample<8u;++sample){
        const unsigned cell=sample*(q.cells()-1u)/7u,row=cell/256u,head=row%16u;
        const float ref=cpu_pv(hp.data()+guard+size_t(row)*q.stride,values.data()+guard,
            hs.data()+guard+size_t(row)*((q.stride+31u)/32u+1u),q.start+row/16u+1u,(head/8u)*256u+cell%256u);
        if(bits(ref)!=bits(raw[guard+q.output_start*4096u+cell]))throw std::runtime_error("register PV CPU recurrence");++cpu;
    }
    ArithmeticBuffer reciprocals(size_t(q.capacity)*16u*4u);
    if(arithmetic::launch<true,true>(dv.as<uint16_t>()+guard,original.p.as<uint16_t>()+guard,
        original.s.as<float>()+guard,candidate.o.as<float>()+guard,candidate.e.as<float>()+guard,
        q.start,q.queries,q.output_start,q.stride,rcp,dvt.as<uint16_t>()+guard,q.stride,
        ids.as<unsigned>()+guard,ids.as<unsigned>()+guard+q.cells(),q.cells(),reciprocals.data<float>(),
        q.queries*16u-1u,nullptr)!=int(hipErrorInvalidValue))throw std::runtime_error("short reciprocal workspace accepted");
    if(arithmetic::launch<true,false>(dv.as<uint16_t>()+guard,original.p.as<uint16_t>()+guard,
        original.s.as<float>()+guard,candidate.o.as<float>()+guard,candidate.e.as<float>()+guard,
        q.start,q.queries,q.output_start,q.stride,rcp,dvt.as<uint16_t>()+guard,q.stride,
        ids.as<unsigned>()+guard,ids.as<unsigned>()+guard+q.cells(),q.cells()-1u,nullptr,
        0u,nullptr)!=int(hipErrorInvalidValue))throw std::runtime_error("short candidate workspace accepted");
    for(unsigned variant=1u;variant<=3u;++variant){
        candidate.reset();copy_probability(candidate,original);reciprocals.reset();reset_indices(ids,q);
        arithmetic_run(variant,candidate,ids,reciprocals,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,q.stride,rcp,true);finish();
        raw_surfaces(native,candidate,bad);reciprocal_observer(reciprocals,q,hs,host_rcp,variant==3u);
        candidate.reset();copy_probability(candidate,original);reciprocals.reset();reset_indices(ids,q);
        arithmetic_run(variant,candidate,ids,reciprocals,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,q.stride,rcp);finish();
        const auto selected=verify_indices(ids,q);check(hipMemset(stats.pointer,0,9u*8u));
        hipLaunchKernelGGL(generated_result,dim3((q.cells()+255u)/256u),dim3(256u),0u,nullptr,
            native.o.as<float>()+guard,native.a.as<float>()+guard,native.e.as<float>()+guard,
            original.o.as<float>()+guard,original.a.as<float>()+guard,candidate.o.as<float>()+guard,
            candidate.a.as<float>()+guard,q.cells(),q.output_start,stats.as<unsigned>());check(hipGetLastError());
        check(hipMemset(bad.pointer,0,4u));compare_device(native.d,candidate.d,q.dwords(),bad);compare_device(native.e,candidate.e,q.ewords(),bad);finish();
        const auto observed=download<unsigned>(stats,2u);
        if(observed[0]||observed[1]!=selected.size()||download<unsigned>(bad,1u)[0])throw std::runtime_error("register PV generated endpoint");
        stage_guards(candidate);reciprocal_observer(reciprocals,q,hs,host_rcp,variant==3u);
        unchanged(candidate.p,hp);unchanged(candidate.s,hs);
        std::printf("{\"kind\":\"register_pv_safety\",\"start\":%u,\"queries\":%u,\"mode\":%u,\"vllm_sum\":%s,\"variant\":%u,\"cells\":%u,\"selected\":%zu,\"cpu_dots\":%u,\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"native_pre_replay_raw_parity\":true,\"complete_original_membership\":true,\"cpu_reciprocal_parity\":true,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
            start,queries,mode,vllm?"true":"false",variant,q.cells(),selected.size(),cpu);std::fflush(stdout);
    }
    unchanged(ds,scores);unchanged(dv,values);unchanged(dvt,tv);stage_guards(native);
}
void arithmetic_extend(std::vector<uint16_t>& values,unsigned columns,unsigned tokens){
    if(tokens<7169u||tokens>8192u)throw std::runtime_error("register capture extent");
    const std::vector<uint16_t> tail(values.begin(),values.begin()+size_t(tokens-7169u)*columns);
    values.insert(values.end(),tail.begin(),tail.end());
    values.insert(values.begin(),guard,0x5a5au);values.insert(values.end(),guard,0x5a5au);
}
void arithmetic_capture(unsigned tokens,const char* qfile,const char* kfile,const char* vfile,
    const char* reference_file,const char* exp_file,const char* rcp_file){
    constexpr unsigned batch=128u;
    auto hq=read_values<uint16_t>(qfile,size_t(7169u)*4096u);
    auto hk=read_values<uint16_t>(kfile,size_t(7169u)*512u),hv=read_values<uint16_t>(vfile,size_t(7169u)*512u);
    const auto golden=read_values<uint16_t>(reference_file,size_t(7169u)*4096u);
    const auto exp=read_values<unsigned char>(exp_file,exp2_backend::table_bytes);
    const auto rcp=read_values<unsigned char>(rcp_file,qrt_sm121_attention_rcp::table_bytes);
    if(!exp2_backend::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("register capture table layout");
    arithmetic_extend(hq,4096u,tokens);arithmetic_extend(hk,512u,tokens);arithmetic_extend(hv,512u,tokens);
    Device dq(hq.size()*2u),dk(hk.size()*2u),dv(hv.size()*2u),dkt(hk.size()*2u),dvt(hv.size()*2u),dex(exp.size()),drcp(rcp.size());
    upload(dq,hq);upload(dk,hk);upload(dv,hv);upload(dex,exp);upload(drcp,rcp);
    check(hipMemset(dkt.pointer,0x5a,hk.size()*2u));check(hipMemset(dvt.pointer,0x5a,hv.size()*2u));finish();
    std::vector<uint16_t> tv(hv.size(),0x5a5au);
    for(unsigned key=0;key<tokens;++key)for(unsigned f=0;f<512u;++f)tv[guard+size_t(f)*tokens+key]=hv[guard+size_t(key)*512u+f];
    auto begin=std::chrono::steady_clock::now();
    check(hipError_t(transpose_keys(dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,size_t(tokens)*512u,tokens,nullptr)));finish();
    const double transpose_ms=elapsed(begin);unchanged(dvt,tv);
    Prepared prepared(dq.as<uint16_t>()+guard,dk.as<uint16_t>()+guard,dkt.as<uint16_t>()+guard,hq.data()+guard,hk.data()+guard,tokens);
    const auto kt_before=download<uint16_t>(dkt,hk.size());
    Shape q{0u,batch,batch,batch,tokens,0u};Stage original(q),native(q),candidate(q);Parts unused(q);
    ArithmeticBuffer reciprocals(size_t(batch)*16u*4u);
    Device scores(q.pwords()*4u),score_copy(q.pwords()*4u),bad(4u),stats(9u*8u);
    Device ids((size_t(batch)*4096u+1u+2u*guard)*4u);
    double samples[4][3]{},maximum[4]{},qk_ms=0.0,probability_ms=0.0;
    size_t selected_total[4]{};unsigned long long groups_total[4]{};size_t endpoints=0,score_cells=0,verified[4]{},external[4]{},cpu=0;
    for(unsigned start=0;start<tokens;start+=batch){
        q.start=start;q.queries=std::min(batch,tokens-start);q.stride=start+q.queries;
        original.shape=q;native.shape=q;candidate.shape=q;
        check(hipMemset(scores.pointer,0xa5,q.pwords()*4u));check(hipMemset(score_copy.pointer,0xa5,q.pwords()*4u));finish();
        begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::scores<128u,true,16u,16u>),
            dim3((q.stride+15u)/16u,16u,(q.queries+15u)/16u),dim3(256u),0u,nullptr,
            dq.as<uint16_t>()+guard,dkt.as<uint16_t>()+guard,prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,
            prepared.qf.as<unsigned>()+guard,prepared.kf.as<unsigned>()+guard,scores.as<float>()+guard,start,q.queries,q.stride,tokens);
        check(hipGetLastError());finish();qk_ms+=elapsed(begin);
        hipLaunchKernelGGL(blackwell_tiled_exact_scores_kernel,dim3((q.stride+31u)/32u,16u,(q.queries+7u)/8u),dim3(256u),0u,nullptr,
            dq.as<uint16_t>()+guard,dkt.as<uint16_t>()+guard,score_copy.as<float>()+guard,start,q.queries,q.stride,tokens);
        check(hipGetLastError());check(hipMemset(bad.pointer,0,4u));compare_device(score_copy,scores,q.pwords(),bad);finish();
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("register input QK raw parity");
        original.reset();native.reset();reset_indices(ids,q);finish();begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL(blackwell_online_probability_kernel,dim3(16u,q.queries),dim3(32u),0u,nullptr,
            scores.as<float>()+guard,original.p.as<uint16_t>()+guard,original.s.as<float>()+guard,q.start,q.stride,dex.as<unsigned char>(),true);
        check(hipGetLastError());finish();probability_ms+=elapsed(begin);
        copy_probability(native,original);coarse_produce(0u,native,unused,dv.as<uint16_t>()+guard,drcp.as<unsigned char>());
        coarse_produce(0u,original,unused,dv.as<uint16_t>()+guard,drcp.as<unsigned char>());
        queue_collect(original,ids);queue_exact(original,ids,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>());finish();
        const auto original_selected=verify_indices(ids,q);packed_expected(original);
        const auto expected=download<float>(original.o,q.owords()),raw=download<float>(original.a,q.owords());
        const auto hp=download<uint16_t>(original.p,q.pwords());const auto hs=download<float>(original.s,q.swords());
        unsigned long long original_groups=0;
        for(auto cell:original_selected)original_groups+=(start+cell/4096u+32u)/32u*2u;
        for(unsigned cell=0;cell<q.cells()&&size_t(start)*4096u+cell<golden.size();++cell)
            if(qrt_sm121_pv_bound::bf16(expected[guard+cell])!=golden[size_t(start)*4096u+cell])throw std::runtime_error("register original GB10 context");
        const unsigned dots=unsigned(std::min(size_t(4u),original_selected.size()));
        for(unsigned sample=0;sample<dots;++sample){
            const unsigned cell=original_selected[size_t(sample)*(original_selected.size()-1u)/std::max(1u,dots-1u)],row=cell/256u,head=row%16u;
            const float ref=cpu_pv(hp.data()+guard+size_t(row)*q.stride,hv.data()+guard,
                hs.data()+guard+size_t(row)*((q.stride+31u)/32u+1u),start+row/16u+1u,(head/8u)*256u+cell%256u);
            if(bits(ref)!=bits(raw[guard+cell]))throw std::runtime_error("register captured CPU PV");++cpu;
        }
        // Check every native pre-replay raw surface before measuring this slab.
        for(unsigned variant=1u;variant<=3u;++variant){
            candidate.reset();copy_probability(candidate,original);reciprocals.reset();
            arithmetic_run(variant,candidate,ids,reciprocals,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>(),true);finish();
            raw_surfaces(native,candidate,bad);reciprocal_observer(reciprocals,q,hs,rcp.data(),variant==3u);
        }
        for(unsigned attempt=0;attempt<4u;++attempt)for(unsigned position=0;position<4u;++position){
            const unsigned variant=(position+start/batch+attempt)%4u;
            candidate.reset();copy_probability(candidate,original);reciprocals.reset();reset_indices(ids,q);finish();
            begin=std::chrono::steady_clock::now();
            arithmetic_run(variant,candidate,ids,reciprocals,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>());finish();
            const double wall=elapsed(begin);
            if(attempt){samples[variant][attempt-1u]+=wall;maximum[variant]=std::max(maximum[variant],wall);}
            const auto selected=verify_indices(ids,q);if(selected!=original_selected)throw std::runtime_error("register original membership");
            raw_surfaces(original,candidate,bad);reciprocal_observer(reciprocals,q,hs,rcp.data(),variant==3u);
            if(!attempt){selected_total[variant]+=selected.size();groups_total[variant]+=original_groups;}
            unchanged(candidate.p,hp);unchanged(candidate.s,hs);
            const auto output=download<float>(candidate.o,q.owords());
            for(unsigned cell=0;cell<q.cells()&&size_t(start)*4096u+cell<golden.size();++cell){
                if(qrt_sm121_pv_bound::bf16(output[guard+cell])!=golden[size_t(start)*4096u+cell])throw std::runtime_error("register captured GB10 mismatch");
                ++external[variant];
            }
            verified[variant]+=q.cells();
        }
        check(hipMemset(bad.pointer,0,4u));compare_device(score_copy,scores,q.pwords(),bad);finish();
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("register immutable input scores");
        endpoints+=q.cells();score_cells+=size_t(q.queries)*16u*q.stride;
    }
    unchanged(dq,hq);unchanged(dk,hk);unchanged(dv,hv);unchanged(dkt,kt_before);unchanged(dvt,tv);
    unchanged(dex,exp);unchanged(drcp,rcp);prepared.verify();
    if(endpoints!=size_t(tokens)*4096u)throw std::runtime_error("register incomplete capture");
    const char* names[]={"original","copied_control","register_rescale","register_and_row_reciprocal"};
    for(unsigned variant=0;variant<4u;++variant){
        std::array<double,3> sorted{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(sorted.begin(),sorted.end());
        std::printf("{\"kind\":\"register_pv_capture\",\"tokens\":%u,\"variant\":%u,\"name\":\"%s\",\"query_batch\":128,\"unique_output_cells\":%zu,\"verified_output_cells\":%zu,\"external_gb10_cells_checked\":%zu,\"external_unique_gb10_cells\":29364224,\"repeated_first_capture_rows\":%u,\"selected\":%zu,\"replayed_groups\":%llu,\"cpu_pv_dots\":%zu,\"probability_cells\":%zu,\"qk_ms\":%.6f,\"qk_preparation_ms\":%.6f,\"probability_ms\":%.6f,\"value_transpose_ms\":%.6f,\"producer_collect_replay_median_ms\":%.6f,\"samples_ms\":[%.6f,%.6f,%.6f],\"maximum_completed_slab_ms\":%.6f,\"reciprocal_workspace_bytes\":%zu,\"warmups\":1,\"samples\":3,\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"native_pre_replay_raw_parity\":true,\"complete_original_membership\":true,\"cpu_reciprocal_parity\":true,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"all_attempts_checked\":true,\"reference_is_compute_input\":false,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            tokens,variant,names[variant],endpoints,verified[variant],external[variant],tokens-7169u,
            selected_total[variant],groups_total[variant],cpu,score_cells,qk_ms,prepared.ms,probability_ms,transpose_ms,
            sorted[1],samples[variant][0],samples[variant][1],samples[variant][2],maximum[variant],variant==3u?reciprocals.bytes:0u);std::fflush(stdout);
    }
}
} // namespace
int main(int argc,char** argv)try{
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));
    if(std::strncmp(prop.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    rounding_safety();
    if(argc==8&&(!std::strcmp(argv[1],"--q7169")||!std::strcmp(argv[1],"--q8192"))){
        arithmetic_capture(!std::strcmp(argv[1],"--q7169")?7169u:8192u,argv[2],argv[3],argv[4],argv[5],argv[6],argv[7]);return 0;
    }
    if(argc!=4||std::strcmp(argv[1],"--selftest"))throw std::runtime_error("use --selftest exp2 rcp");
    const auto exp=read_values<unsigned char>(argv[2],exp2_backend::table_bytes);
    const auto rcp=read_values<unsigned char>(argv[3],qrt_sm121_attention_rcp::table_bytes);
    if(!exp2_backend::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("register PV table layout");
    Device dex(exp.size()),drcp(rcp.size());upload(dex,exp);upload(drcp,rcp);
    for(auto shape:{std::pair<unsigned,unsigned>{0u,1u},{31u,2u},{17u,17u},{0u,128u},{127u,3u},{511u,17u},{8191u,1u}})
        for(unsigned mode=0;mode<6u;++mode)for(bool vllm:{false,true})arithmetic_generated(shape.first,shape.second,mode,vllm,dex.as<unsigned char>(),drcp.as<unsigned char>(),rcp.data());
    unchanged(dex,exp);unchanged(drcp,rcp);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"register_pv_error=%s\n",e.what());return 2;}
