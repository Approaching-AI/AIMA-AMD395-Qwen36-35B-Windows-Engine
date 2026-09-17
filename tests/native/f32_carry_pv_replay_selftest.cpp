#define main qrt_prior_register_pv_main
#include "register_pv_arithmetic_selftest.cpp"
#undef main
#include "../../native/providers/ck_fmha/f32_carry_pv_replay.h"
#include <limits>

namespace {
namespace carry_pv=qrt_f32_carry_pv;
struct CarryPlan {
    unsigned capacity,tokens;
    ArithmeticBuffer p,v,pf,vf,fallback,stats;
    double value_ms[3]{};
    CarryPlan(const Shape& q,unsigned n):capacity(q.capacity),tokens(n),
        p(size_t(q.capacity)*16u*carry_pv::pitch(q.maximum_stride)*4u),
        v(size_t(512u)*carry_pv::pitch(n)*4u),pf(size_t(q.capacity)*16u*4u),
        vf(512u*4u),fallback(size_t(q.capacity)*4096u*4u),stats(12u){}
    carry_pv::Workspace workspace(){return {p.data<uint32_t>(),v.data<uint32_t>(),
        pf.data<unsigned>(),vf.data<unsigned>(),fallback.data<unsigned>(),stats.data<unsigned>(),
        p.bytes/4u,v.bytes/4u,pf.bytes/4u,vf.bytes/4u,fallback.bytes/4u,3u};}
    void reset_replay(){p.reset();pf.reset();fallback.reset();stats.reset();}
    void guards(){for(auto* b:{&p,&v,&pf,&vf,&fallback,&stats})b->verify_guards();}
    void verify_encoding(const std::vector<uint16_t>& input,unsigned rows,unsigned stride,
        unsigned start,bool probability,bool stored){
        auto& words=probability?p:v;auto& flags=probability?pf:vf;
        std::vector<uint32_t> packed(words.bytes/4u),observed(flags.bytes/4u);
        check(hipMemcpy(packed.data(),words.data<uint32_t>(),words.bytes,hipMemcpyDeviceToHost));
        check(hipMemcpy(observed.data(),flags.data<unsigned>(),flags.bytes,hipMemcpyDeviceToHost));
        const unsigned padded=carry_pv::pitch(stride);
        for(unsigned row=0u;row<rows;++row){
            unsigned eligible=1u;
            for(unsigned k=0u;k<padded;++k){
                const unsigned count=probability?start+row/16u+1u:stride;
                const uint16_t x=k<count?input[guard+size_t(row)*stride+k]:0u;
                const unsigned e=(x>>7u)&255u;
                eligible&=unsigned(!(x&0x7fffu)||(e>=64u&&e<=190u));
                const int exponent=(x&0x7fffu)?int(e)-127:-512;
                const uint32_t expected=stored?(uint32_t(x)<<16u)|uint16_t(exponent):0xa5a5a5a5u;
                if(packed[size_t(row)*padded+k]!=expected)throw std::runtime_error("carry PV encoded operands");
            }
            if(observed[row]!=eligible)throw std::runtime_error("carry PV row eligibility");
        }
        for(size_t i=size_t(rows)*padded;i<packed.size();++i)
            if(packed[i]!=0xa5a5a5a5u)throw std::runtime_error("carry PV encoded unused tail");
        for(size_t i=rows;i<observed.size();++i)
            if(observed[i]!=0xa5a5a5a5u)throw std::runtime_error("carry PV flag unused tail");
        guards();
    }
    void prepare_values(const uint16_t* tv,const std::vector<uint16_t>& host){
        double samples[3][3]{};
        for(unsigned attempt=0;attempt<4u;++attempt)for(unsigned position=0;position<2u;++position){
            const unsigned variant=1u+(position+attempt)%2u;v.reset();vf.reset();finish();
            const auto begin=std::chrono::steady_clock::now();
            if(variant==1u)check(hipError_t(carry_pv::prepare_value<false>(tv,tokens,workspace(),nullptr)));
            else check(hipError_t(carry_pv::prepare_value<true>(tv,tokens,workspace(),nullptr)));
            finish();if(attempt)samples[variant][attempt-1u]=elapsed(begin);
            verify_encoding(host,512u,tokens,0u,false,variant==2u);
        }
        // Keep the packed view and flags resident for both replay variants.
        check(hipError_t(carry_pv::prepare_value<true>(tv,tokens,workspace(),nullptr)));finish();
        for(unsigned variant=1;variant<3u;++variant){
            std::sort(samples[variant],samples[variant]+3u);value_ms[variant]=samples[variant][1];
        }
    }
    std::array<unsigned,3> verify_fallback(const std::vector<unsigned>& selected){
        std::array<unsigned,3> counts{};check(hipMemcpy(counts.data(),stats.data<unsigned>(),12u,hipMemcpyDeviceToHost));
        if(counts[0]>selected.size()||counts[0]!=counts[1]+counts[2])throw std::runtime_error("carry PV fallback count");
        std::vector<unsigned> data(fallback.bytes/4u);check(hipMemcpy(data.data(),fallback.data<unsigned>(),fallback.bytes,hipMemcpyDeviceToHost));
        std::vector<unsigned> active(data.begin(),data.begin()+counts[0]);std::sort(active.begin(),active.end());
        for(size_t i=0;i<active.size();++i)
            if(!std::binary_search(selected.begin(),selected.end(),active[i])||(i&&active[i]==active[i-1]))
                throw std::runtime_error("carry PV fallback membership");
        for(size_t i=counts[0];i<data.size();++i)if(data[i]!=0xa5a5a5a5u)throw std::runtime_error("carry PV fallback tail");
        guards();return counts;
    }
};

void carry_replay(unsigned variant,Stage& s,Device& ids,CarryPlan& plan,
    const uint16_t* v,const uint16_t* tv,unsigned n,const unsigned char* rcp){
    const auto& q=s.shape;const auto* indices=ids.as<unsigned>()+guard;const auto* count=indices+q.cells();
    if(!variant){
        hipLaunchKernelGGL((blackwell_compacted_pv_replay_kernel<true,false,true>),
            dim3(std::min(1024u,(q.cells()+63u)/64u)),dim3(256u),0u,nullptr,v,s.p.as<uint16_t>()+guard,
            s.s.as<float>()+guard,s.o.as<float>()+guard,q.start,q.output_start,q.stride,rcp,
            s.a.as<float>()+guard,s.d.as<float>()+guard,indices,count,tv,n,0u);check(hipGetLastError());return;
    }
#define CARRY_REPLAY(P) check(hipError_t(carry_pv::launch<P>(v,s.p.as<uint16_t>()+guard,s.s.as<float>()+guard, \
    s.o.as<float>()+guard,q.start,q.queries,q.output_start,q.stride,rcp,s.a.as<float>()+guard,s.d.as<float>()+guard, \
    indices,count,tv,n,plan.workspace(),nullptr)))
    if(variant==1u){CARRY_REPLAY(false);}else if(variant==2u){CARRY_REPLAY(true);}else throw std::runtime_error("carry PV variant");
#undef CARRY_REPLAY
}

void carry_owner(unsigned variant,Stage& s,Device& ids,CarryPlan& plan,ArithmeticBuffer& unused,
    const uint16_t* v,const uint16_t* tv,unsigned n,const unsigned char* rcp){
    arithmetic_run(4u,s,ids,unused,v,tv,n,rcp,true);
    check(hipMemsetAsync(ids.as<unsigned>()+guard+s.shape.cells(),0,4u,nullptr));
    queue_collect(s,ids);carry_replay(variant,s,ids,plan,v,tv,n,rcp);
}

void carry_contract(Stage& s,Device& ids,CarryPlan& plan,const uint16_t* v,const uint16_t* tv){
    const auto& q=s.shape;const auto valid=plan.workspace();unsigned rejected=0u;
    for(unsigned test=0u;test<12u;++test){
        auto w=valid;unsigned start=q.start,queries=q.queries,stride=q.stride,n=q.stride;
        if(test==0u)w.probability_words=size_t(q.queries)*16u*carry_pv::pitch(q.stride)-1u;
        if(test==1u)w.value_words=size_t(512u)*carry_pv::pitch(q.stride)-1u;
        if(test==2u)w.probability_flag_words=q.queries*16u-1u;
        if(test==3u)w.value_flag_words=511u;
        if(test==4u)w.fallback_words=q.cells()-1u;
        if(test==5u)w.stats_words=2u;
        if(test==6u)queries=0u;
        if(test==7u)queries=129u;
        if(test==8u)start=stride;
        if(test==9u)stride=n=8193u;
        if(test==10u)n=0u;
        if(test==11u)w.probability=nullptr;
        if(carry_pv::launch<true>(v,s.p.as<uint16_t>()+guard,s.s.as<float>()+guard,s.o.as<float>()+guard,
            start,queries,q.output_start,stride,nullptr,nullptr,nullptr,ids.as<unsigned>()+guard,
            ids.as<unsigned>()+guard+q.cells(),tv,n,w,nullptr)!=int(hipErrorInvalidValue))
            throw std::runtime_error("carry PV invalid workspace or shape accepted");
        ++rejected;
    }
    if(carry_pv::prepare_value<true>(tv,8193u,valid,nullptr)!=int(hipErrorInvalidValue)||
        carry_pv::prepare_value<false>(tv,0u,valid,nullptr)!=int(hipErrorInvalidValue))
        throw std::runtime_error("carry PV invalid preparation extent accepted");
    std::printf("{\"kind\":\"carry_pv_contract\",\"invalid_launches_rejected\":%u,\"invalid_preparations_rejected\":2}\n",rejected);
}

void carry_generated(unsigned start,unsigned queries,unsigned mode,unsigned selection,const unsigned char* rcp){
    Shape q{start,queries,start+queries,queries+2u,start+queries+7u,3u};
    std::vector<uint16_t> hp(q.pwords(),0x5a5au),values(size_t(q.stride)*512u+2u*guard,0x5a5au),tv(values.size(),0x5a5au);
    std::vector<float> hs(q.swords(),-123.0f);const unsigned tiles=(q.stride+31u)/32u;
    for(unsigned row=0;row<queries*16u;++row){
        for(unsigned k=0;k<start+row/16u+1u;++k){
            uint16_t word=qrt_sm121_pv_bound::bf16(float((k*17u+row*3u)%61u+1u)/64.0f);
            if(mode==1u)word=(k&1u)?0x8000u:0u;
            if(mode==3u)word=0x2000u;
            if(mode==4u)word=0x5f7fu;
            hp[guard+size_t(row)*q.stride+k]=word;
        }
        for(unsigned tile=0;tile<tiles;++tile)hs[guard+size_t(row)*(tiles+1u)+tile]=
            mode==5u&&tile%2u?std::numeric_limits<float>::min()/2.0f:1.0f;
        hs[guard+size_t(row)*(tiles+1u)+tiles]=1.0f;
    }
    for(unsigned k=0;k<q.stride;++k)for(unsigned f=0;f<512u;++f){
        uint16_t word=qrt_sm121_pv_bound::bf16(float(int((k*173u+f*19u)%63u)-31)/64.0f);
        if(mode==2u){const uint16_t edges[]={0u,0x8000u,1u,0x8001u,0x7f80u,0xff80u,0x7fc1u,0xffc1u};word=edges[(k+f)%8u];}
        if(mode==3u)word=0x2000u;
        if(mode==4u)word=0x5f7fu;
        values[guard+size_t(k)*512u+f]=word;tv[guard+size_t(f)*q.stride+k]=word;
    }
    Device dv(values.size()*2u),dvt(tv.size()*2u),ids((size_t(q.capacity)*4096u+1u+2u*guard)*4u),bad(4u);
    upload(dv,values);upload(dvt,tv);Stage original(q),candidate(q);CarryPlan plan(q,q.stride);
    if(!start&&queries==1u&&!mode&&!selection)carry_contract(original,ids,plan,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard);
    plan.prepare_values(dvt.as<uint16_t>()+guard,tv);
    std::vector<unsigned> list(size_t(q.capacity)*4096u+1u+2u*guard,0xa5a5a5a5u),selected;
    for(unsigned cell=0;cell<q.cells();++cell)if(selection==1u||(selection==2u&&cell%257u==0u))selected.push_back(cell);
    std::copy(selected.begin(),selected.end(),list.begin()+guard);list[guard+q.cells()]=unsigned(selected.size());upload(ids,list);
    upload(original.p,hp);upload(original.s,hs);
    carry_replay(0u,original,ids,plan,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,q.stride,rcp);finish();
    const auto raw=download<float>(original.a,q.owords());unsigned cpu=0u;
    for(unsigned sample=0;sample<std::min(size_t(8u),selected.size());++sample){
        const unsigned cell=selected[size_t(sample)*(selected.size()-1u)/std::max(size_t(1u),std::min(size_t(8u),selected.size())-1u)],row=cell/256u;
        const float expected=cpu_pv(hp.data()+guard+size_t(row)*q.stride,values.data()+guard,
            hs.data()+guard+size_t(row)*(tiles+1u),start+row/16u+1u,(row%16u/8u)*256u+cell%256u);
        if(bits(expected)!=bits(raw[guard+q.output_start*4096u+cell]))throw std::runtime_error("carry PV independent CPU recurrence");++cpu;
    }
    for(unsigned variant=1u;variant<3u;++variant){
        candidate.reset();upload(candidate.p,hp);upload(candidate.s,hs);plan.reset_replay();
        carry_replay(variant,candidate,ids,plan,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,q.stride,rcp);finish();
        raw_surfaces(original,candidate,bad);const auto counts=plan.verify_fallback(selected);
        plan.verify_encoding(hp,queries*16u,q.stride,start,true,variant==2u);
        unchanged(candidate.p,hp);unchanged(candidate.s,hs);unchanged(ids,list);
        if(selection&&mode==2u&&!counts[1])throw std::runtime_error("missing exceptional operand fallback");
        if(selection&&mode==3u&&counts[2]!=selected.size())throw std::runtime_error("missing tiny product fallback");
        if(selection&&mode<2u&&counts[0])throw std::runtime_error("ordinary input unexpectedly rejected");
        std::printf("{\"kind\":\"carry_pv_safety\",\"start\":%u,\"queries\":%u,\"mode\":%u,\"selection\":%u,\"variant\":%u,\"cells\":%u,\"selected\":%zu,\"fallback\":%u,\"operand_fallback\":%u,\"arithmetic_fallback\":%u,\"cpu_dots\":%u,\"raw_bit_mismatches\":0,\"complete_original_membership\":true,\"fallback_subset_checked\":true,\"cpu_encoding_checked\":true,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
            start,queries,mode,selection,variant,q.cells(),selected.size(),counts[0],counts[1],counts[2],cpu);std::fflush(stdout);
    }
    unchanged(original.p,hp);unchanged(original.s,hs);unchanged(dv,values);unchanged(dvt,tv);
    plan.verify_encoding(tv,512u,q.stride,0u,false,true);
}
} // namespace

namespace {
void carry_capture(unsigned tokens,const char* qfile,const char* kfile,const char* vfile,
    const char* reference_file,const char* exp_file,const char* rcp_file){
    constexpr unsigned batch=128u;
    auto hq=read_values<uint16_t>(qfile,size_t(7169u)*4096u);
    auto hk=read_values<uint16_t>(kfile,size_t(7169u)*512u),hv=read_values<uint16_t>(vfile,size_t(7169u)*512u);
    const auto golden=read_values<uint16_t>(reference_file,size_t(7169u)*4096u);
    const auto exp=read_values<unsigned char>(exp_file,exp2_backend::table_bytes);
    const auto rcp=read_values<unsigned char>(rcp_file,qrt_sm121_attention_rcp::table_bytes);
    if(!exp2_backend::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("carry PV capture table layout");
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
    Shape q{0u,batch,batch,batch,tokens,0u};Stage original(q),native(q),candidate(q);CarryPlan plan(q,tokens);
    plan.prepare_values(dvt.as<uint16_t>()+guard,tv);
    ArithmeticBuffer reciprocals(size_t(batch)*16u*4u);
    Device scores(q.pwords()*4u),score_copy(q.pwords()*4u),bad(4u),stats(9u*8u);
    Device ids((size_t(batch)*4096u+1u+2u*guard)*4u);
    double samples[3][3]{},maximum[3]{},qk_ms=0.0,probability_ms=0.0;
    size_t selected_total[3]{},fallback_total[3]{},operand_total[3]{},arithmetic_total[3]{};unsigned long long groups_total[3]{};size_t endpoints=0,score_cells=0,verified[3]{},external[3]{},cpu=0;
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
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("carry PV input QK raw parity");
        original.reset();native.reset();reset_indices(ids,q);finish();begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL(blackwell_online_probability_kernel,dim3(16u,q.queries),dim3(32u),0u,nullptr,
            scores.as<float>()+guard,original.p.as<uint16_t>()+guard,original.s.as<float>()+guard,q.start,q.stride,dex.as<unsigned char>(),true);
        check(hipGetLastError());finish();probability_ms+=elapsed(begin);
        copy_probability(native,original);
        arithmetic_run(4u,native,ids,reciprocals,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>(),true);
        carry_owner(0u,original,ids,plan,reciprocals,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>());finish();
        const auto original_selected=verify_indices(ids,q);packed_expected(original);
        const auto expected=download<float>(original.o,q.owords()),raw=download<float>(original.a,q.owords());
        const auto hp=download<uint16_t>(original.p,q.pwords());const auto hs=download<float>(original.s,q.swords());
        unsigned long long original_groups=0;
        for(auto cell:original_selected)original_groups+=(start+cell/4096u+32u)/32u*2u;
        for(unsigned cell=0;cell<q.cells()&&size_t(start)*4096u+cell<golden.size();++cell)
            if(qrt_sm121_pv_bound::bf16(expected[guard+cell])!=golden[size_t(start)*4096u+cell])throw std::runtime_error("carry PV original GB10 context");
        const unsigned dots=unsigned(std::min(size_t(4u),original_selected.size()));
        for(unsigned sample=0;sample<dots;++sample){
            const unsigned cell=original_selected[size_t(sample)*(original_selected.size()-1u)/std::max(1u,dots-1u)],row=cell/256u,head=row%16u;
            const float ref=cpu_pv(hp.data()+guard+size_t(row)*q.stride,hv.data()+guard,
                hs.data()+guard+size_t(row)*((q.stride+31u)/32u+1u),start+row/16u+1u,(head/8u)*256u+cell%256u);
            if(bits(ref)!=bits(raw[guard+cell]))throw std::runtime_error("carry PV captured CPU PV");++cpu;
        }
        candidate.reset();copy_probability(candidate,original);
        arithmetic_run(4u,candidate,ids,reciprocals,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>(),true);finish();
        raw_surfaces(native,candidate,bad);
        for(unsigned attempt=0;attempt<4u;++attempt)for(unsigned position=0;position<3u;++position){
            const unsigned variant=(position+start/batch+attempt)%3u;
            candidate.reset();copy_probability(candidate,original);plan.reset_replay();reset_indices(ids,q);finish();
            begin=std::chrono::steady_clock::now();
            carry_owner(variant,candidate,ids,plan,reciprocals,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>());finish();
            const double wall=elapsed(begin);
            if(attempt){samples[variant][attempt-1u]+=wall;maximum[variant]=std::max(maximum[variant],wall);}
            const auto selected=verify_indices(ids,q);if(selected!=original_selected)throw std::runtime_error("carry PV original membership");
            raw_surfaces(original,candidate,bad);
            if(variant){
                const auto fallback_counts=plan.verify_fallback(selected);
                if(!attempt){
                    fallback_total[variant]+=fallback_counts[0];operand_total[variant]+=fallback_counts[1];arithmetic_total[variant]+=fallback_counts[2];
                    plan.verify_encoding(hp,q.queries*16u,q.stride,q.start,true,variant==2u);
                }
            }
            if(!attempt){selected_total[variant]+=selected.size();groups_total[variant]+=original_groups;}
            unchanged(candidate.p,hp);unchanged(candidate.s,hs);
            const auto output=download<float>(candidate.o,q.owords());
            for(unsigned cell=0;cell<q.cells()&&size_t(start)*4096u+cell<golden.size();++cell){
                if(qrt_sm121_pv_bound::bf16(output[guard+cell])!=golden[size_t(start)*4096u+cell])throw std::runtime_error("carry PV captured GB10 mismatch");
                ++external[variant];
            }
            verified[variant]+=q.cells();
        }
        check(hipMemset(bad.pointer,0,4u));compare_device(score_copy,scores,q.pwords(),bad);finish();
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("carry PV immutable input scores");
        endpoints+=q.cells();score_cells+=size_t(q.queries)*16u*q.stride;
    }
    unchanged(dq,hq);unchanged(dk,hk);unchanged(dv,hv);unchanged(dkt,kt_before);unchanged(dvt,tv);
    unchanged(dex,exp);unchanged(drcp,rcp);prepared.verify();
    plan.verify_encoding(tv,512u,tokens,0u,false,true);
    if(endpoints!=size_t(tokens)*4096u)throw std::runtime_error("carry PV incomplete capture");
    const char* names[]={"production_register_rescale","direct_bf16_f32_carry","decoded_bf16_f32_carry"};
    for(unsigned variant=0;variant<3u;++variant){
        std::array<double,3> sorted{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(sorted.begin(),sorted.end());
        std::printf("{\"kind\":\"carry_pv_capture\",\"tokens\":%u,\"variant\":%u,\"name\":\"%s\",\"query_batch\":128,\"unique_output_cells\":%zu,\"verified_output_cells\":%zu,\"external_gb10_cells_checked\":%zu,\"external_unique_gb10_cells\":29364224,\"repeated_first_capture_rows\":%u,\"selected\":%zu,\"replayed_groups\":%llu,\"fallback\":%zu,\"operand_fallback\":%zu,\"arithmetic_fallback\":%zu,\"cpu_pv_dots\":%zu,\"probability_cells\":%zu,\"qk_ms\":%.6f,\"qk_preparation_ms\":%.6f,\"probability_ms\":%.6f,\"value_transpose_ms\":%.6f,\"value_preparation_median_ms\":%.6f,\"producer_collect_replay_median_ms\":%.6f,\"with_value_preparation_ms\":%.6f,\"samples_ms\":[%.6f,%.6f,%.6f],\"maximum_completed_slab_ms\":%.6f,\"warmups\":1,\"samples\":3,\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"native_pre_replay_raw_parity\":true,\"complete_original_membership\":true,\"fallback_subset_checked\":true,\"cpu_encoding_checked\":true,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"all_attempts_checked\":true,\"reference_is_compute_input\":false,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            tokens,variant,names[variant],endpoints,verified[variant],external[variant],tokens-7169u,
            selected_total[variant],groups_total[variant],fallback_total[variant],operand_total[variant],arithmetic_total[variant],
            cpu,score_cells,qk_ms,prepared.ms,probability_ms,transpose_ms,plan.value_ms[variant],
            sorted[1],sorted[1]+plan.value_ms[variant]+transpose_ms,samples[variant][0],samples[variant][1],samples[variant][2],maximum[variant]);std::fflush(stdout);
    }
}

} // namespace

int main(int argc,char** argv)try{
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));
    if(std::strncmp(prop.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    rounding_safety();
    if(argc==8&&(!std::strcmp(argv[1],"--q7169")||!std::strcmp(argv[1],"--q8192"))){
        carry_capture(!std::strcmp(argv[1],"--q7169")?7169u:8192u,argv[2],argv[3],argv[4],argv[5],argv[6],argv[7]);return 0;
    }
    if(argc!=4||std::strcmp(argv[1],"--selftest"))throw std::runtime_error("use --selftest exp2 rcp");
    const auto rcp=read_values<unsigned char>(argv[3],qrt_sm121_attention_rcp::table_bytes);
    if(!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("carry PV table layout");
    Device drcp(rcp.size());upload(drcp,rcp);
    for(auto shape:{std::pair<unsigned,unsigned>{0u,1u},{31u,2u},{17u,17u},{0u,128u},{127u,3u},{511u,17u},{8191u,1u},{8064u,128u}})
        for(unsigned mode=0;mode<6u;++mode)for(unsigned selection=0;selection<3u;++selection)
            carry_generated(shape.first,shape.second,mode,selection,drcp.as<unsigned char>());
    unchanged(drcp,rcp);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"carry_pv_error=%s\n",e.what());return 2;}
