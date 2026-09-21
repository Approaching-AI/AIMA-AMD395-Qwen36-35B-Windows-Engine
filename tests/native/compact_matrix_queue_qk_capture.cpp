// Reuse the independent original-QK control, guarded owners, completion
// deadline and byte comparisons from the qualified complete-attention test.
#define QRT_COMBINED_ATTENTION_NO_MAIN
#include "combined_exact_attention_capture.cpp"
#include "../../native/providers/ck_fmha/streamed_exact_attention.h"
#include "../../native/providers/ck_fmha/fused_probability_pv.h"
#include "../../native/providers/ck_fmha/compact_matrix_queue_qk.h"
#ifdef QRT_WAVE_MATRIX_QK_CAPTURE
#include "../../native/providers/ck_fmha/wave_matrix_qk.h"
#ifdef QRT_PARTIAL_WAVE_MATRIX_QK_CAPTURE
#include "../../native/providers/ck_fmha/partial_wave_matrix_qk.h"
#include <memory>
#define QRT_MATRIX_QK_LABEL "partial_wave_matrix_qk"
#define QRT_MATRIX_QK_MARKER "PARTIAL_WAVE_MATRIX_QK"
#define QRT_MATRIX_QK_CAPTURE_VARIANTS 5u
#define QRT_MATRIX_QK_SAFETY_VARIANTS 7u
#define QRT_MATRIX_QK_FORCE_FIELD ",\"forced_all_original_wave\":true,\"forced_all_original_partial_wave\":true"
#elif defined(QRT_WAVE_MATRIX_REMAINDER_QK_CAPTURE)
#define QRT_MATRIX_QK_LABEL "wave_matrix_remainder_qk"
#define QRT_MATRIX_QK_MARKER "WAVE_MATRIX_REMAINDER_QK"
#define QRT_MATRIX_QK_CAPTURE_VARIANTS 4u
#define QRT_MATRIX_QK_SAFETY_VARIANTS 5u
#else
#define QRT_MATRIX_QK_LABEL "wave_matrix_qk"
#define QRT_MATRIX_QK_MARKER "WAVE_MATRIX_QK"
#endif
#define QRT_MATRIX_QK_QUEUE_CONTROL "false"
#ifndef QRT_MATRIX_QK_FORCE_FIELD
#define QRT_MATRIX_QK_FORCE_FIELD ",\"forced_all_original_wave\":true"
#endif
#else
#define QRT_MATRIX_QK_LABEL "compact_matrix_queue_qk"
#define QRT_MATRIX_QK_MARKER "COMPACT_MATRIX_QUEUE_QK"
#define QRT_MATRIX_QK_QUEUE_CONTROL "true"
#define QRT_MATRIX_QK_FORCE_FIELD ""
#endif
#ifndef QRT_MATRIX_QK_CAPTURE_VARIANTS
#define QRT_MATRIX_QK_CAPTURE_VARIANTS 3u
#define QRT_MATRIX_QK_SAFETY_VARIANTS 4u
#endif

namespace {
#ifdef QRT_PARTIAL_WAVE_MATRIX_QK_CAPTURE
struct PartialMetadata {
    using Row = qrt_partial_wave_matrix_qk::Row;
    Guarded query,key;
    std::vector<Row> expected_query,expected_key;
    double ms=0.0;
    PartialMetadata(const uint16_t* q,const uint16_t* k,const std::vector<uint16_t>& hq,
        const std::vector<uint16_t>& hk,unsigned tokens):
        query(size_t(tokens)*16u*16u*sizeof(Row)),key(size_t(tokens)*2u*16u*sizeof(Row)),
        expected_query(size_t(tokens)*16u*16u),expected_key(size_t(tokens)*2u*16u){
        for(unsigned is_key=0u;is_key<2u;++is_key){
            const auto& input=is_key?hk:hq;
            auto& expected=is_key?expected_key:expected_query;
            const unsigned heads=is_key?2u:16u;
            for(unsigned row=0u;row<tokens*heads;++row)for(unsigned g=0u;g<16u;++g){
                const auto encoded=qrt_sm121_partial_matrix_group::prepare(input.data()+size_t(row)*256u+g*16u);
                for(unsigned i=0u;i<16u;++i)
                    if(qrt_sm121_partial_matrix_group::original(encoded,i)!=input[size_t(row)*256u+g*16u+i])
                        throw std::runtime_error("partial metadata is not lossless");
                expected[is_key?(size_t(row%heads)*16u+g)*tokens+row/heads:size_t(row)*16u+g]=encoded;
            }
        }
        finish();const auto begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL((qrt_partial_wave_matrix_qk::prepare<false>),dim3((tokens*16u*16u+255u)/256u),dim3(256u),0u,nullptr,
            q,query.as<Row>(),tokens);check(hipGetLastError());
        hipLaunchKernelGGL((qrt_partial_wave_matrix_qk::prepare<true>),dim3((tokens*2u*16u+255u)/256u),dim3(256u),0u,nullptr,
            k,key.as<Row>(),tokens);check(hipGetLastError());finish();ms=elapsed(begin);verify();
    }
    void verify(){query.immutable(expected_query);key.immutable(expected_key);}
};
#endif
void immutable_transpose(Guarded& device,const std::vector<uint16_t>& source,unsigned tokens){
    std::vector<uint16_t> expected(source.size());
    for(unsigned token=0u;token<tokens;++token)for(unsigned feature=0u;feature<512u;++feature)
        expected[size_t(feature)*tokens+token]=source[size_t(token)*512u+feature];
    device.immutable(expected);
}
struct NarrowDomain {
    using Metadata=qrt_compact_matrix_queue_qk::Row;
    Guarded query,key,statistics,metadata_query,metadata_key;
    std::vector<unsigned> expected_query,expected_key;
    std::vector<Metadata> expected_metadata_query,expected_metadata_key;
    double ms=0.0,matrix_ms=0.0;
#ifdef QRT_PARTIAL_WAVE_MATRIX_QK_CAPTURE
    std::unique_ptr<PartialMetadata> partial;
#endif
    NarrowDomain(const uint16_t* q,const uint16_t* k,const std::vector<uint16_t>& hq,
        const std::vector<uint16_t>& hk,Prepared& prepared,unsigned tokens):
        query(size_t(tokens)*16u*4u),key(size_t(tokens)*2u*4u),statistics(8u),
        metadata_query(size_t(tokens)*16u*16u*sizeof(Metadata)),metadata_key(size_t(tokens)*2u*16u*sizeof(Metadata)),
        expected_query(size_t(tokens)*16u,1u),expected_key(size_t(tokens)*2u,1u),
        expected_metadata_query(size_t(tokens)*16u*16u),expected_metadata_key(size_t(tokens)*2u*16u){
        for(unsigned is_key=0;is_key<2u;++is_key){
            const auto& input=is_key?hk:hq;auto& flags=is_key?expected_key:expected_query;
            auto& metadata=is_key?expected_metadata_key:expected_metadata_query;
            const unsigned heads=is_key?2u:16u;
            for(unsigned row=0u;row<tokens*heads;++row){
                for(unsigned i=0u;i<256u;++i)flags[row]&=unsigned(qrt_sm121_narrow_f32_carry::eligible(input[size_t(row)*256u+i]));
                for(unsigned g=0u;g<16u;++g){
                    const auto m=qrt_sm121_compact_matrix_group::prepare(input.data()+size_t(row)*256u+g*16u);
                    for(unsigned i=0u;i<16u;++i)if(qrt_sm121_compact_matrix_group::compact::original(m.encoded,i)!=input[size_t(row)*256u+g*16u+i])throw std::runtime_error("matrix metadata is not lossless");
                    metadata[is_key?(size_t(row%heads)*16u+g)*tokens+row/heads:size_t(row)*16u+g]=m;
                }
            }
        }
        finish();auto begin=std::chrono::steady_clock::now();
        const auto w=workspace(prepared,tokens);
        check(hipError_t(qrt_narrow_domain_qk::prepare_domain(q,k,w,nullptr)));finish();ms=elapsed(begin);
        begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL((qrt_compact_matrix_queue_qk::prepare<false>),dim3((tokens*16u*16u+255u)/256u),dim3(256u),0u,nullptr,
            q,metadata_query.as<Metadata>(),tokens);check(hipGetLastError());
        hipLaunchKernelGGL((qrt_compact_matrix_queue_qk::prepare<true>),dim3((tokens*2u*16u+255u)/256u),dim3(256u),0u,nullptr,
            k,metadata_key.as<Metadata>(),tokens);check(hipGetLastError());finish();matrix_ms=elapsed(begin);
#ifdef QRT_PARTIAL_WAVE_MATRIX_QK_CAPTURE
        partial=std::make_unique<PartialMetadata>(q,k,hq,hk,tokens);
#endif
        verify();
    }
    qrt_narrow_domain_qk::Workspace workspace(Prepared& p,unsigned tokens){
        return {p.qp.as<uint32_t>()+guard,p.kp.as<uint32_t>()+guard,
            p.qf.as<unsigned>()+guard,p.kf.as<unsigned>()+guard,query.as<unsigned>(),
            key.as<unsigned>(),statistics.as<unsigned>(),tokens};
    }
    qrt_compact_matrix_queue_qk::Workspace matrix_workspace(Prepared& p,unsigned tokens){
        return {workspace(p,tokens),metadata_query.as<Metadata>(),metadata_key.as<Metadata>()};
    }
    double matrix_preparation(unsigned variant) const{
#ifdef QRT_PARTIAL_WAVE_MATRIX_QK_CAPTURE
        if(variant==4u || variant==6u)return partial->ms;
#endif
        return variant?matrix_ms:0.0;
    }
    void verify(){
        query.immutable(expected_query);key.immutable(expected_key);statistics.guards();
        metadata_query.immutable(expected_metadata_query);metadata_key.immutable(expected_metadata_key);
#ifdef QRT_PARTIAL_WAVE_MATRIX_QK_CAPTURE
        partial->verify();
#endif
    }
};

void producer(const uint16_t* q,const uint16_t* kt,const uint16_t* v,const uint16_t* vt,
    Prepared& prepared,NarrowDomain& domain,AttentionOutputs& out,unsigned start,unsigned count,unsigned tokens,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp,unsigned variant){
    const unsigned stride=start+count;
    auto* scores=out.tensor.scores.as<float>()+guard;
    auto* p=out.tensor.probability.as<uint16_t>()+guard;
    auto* s=out.tensor.scales.as<float>()+guard;
    check(hipMemsetAsync(domain.statistics.data(),0,8u,nullptr));
    if(!variant){
        const auto workspace=domain.workspace(prepared,tokens);
        check(hipError_t(qrt_narrow_domain_qk::launch_workspace(&workspace,q,kt,scores,nullptr,start,count,stride,tokens)));
    }
#ifdef QRT_PARTIAL_WAVE_MATRIX_QK_CAPTURE
    else if(variant==4u || variant==6u){
        const qrt_partial_wave_matrix_qk::Workspace workspace{domain.workspace(prepared,tokens),
            domain.partial->query.as<PartialMetadata::Row>(),domain.partial->key.as<PartialMetadata::Row>()};
        const auto launch=variant==4u?qrt_partial_wave_matrix_qk::launch<false>:
            qrt_partial_wave_matrix_qk::launch<true>;
        check(hipError_t(launch(&workspace,q,kt,scores,nullptr,start,count,stride,tokens)));
    }
#endif
    else{
        const auto workspace=domain.matrix_workspace(prepared,tokens);
#ifdef QRT_WAVE_MATRIX_QK_CAPTURE
        const auto launch=variant==1u?qrt_compact_matrix_queue_qk::launch<true>:
            variant==2u?qrt_wave_matrix_qk::launch<false>:
#ifdef QRT_PARTIAL_WAVE_MATRIX_QK_CAPTURE
            variant==3u?qrt_wave_matrix_qk::launch<false,true>:
            variant==5u?qrt_wave_matrix_qk::launch<true>:nullptr;
#elif defined(QRT_WAVE_MATRIX_REMAINDER_QK_CAPTURE)
            variant==3u?qrt_wave_matrix_qk::launch<false,true>:
            variant==4u?qrt_wave_matrix_qk::launch<true>:nullptr;
#else
            variant==3u?qrt_wave_matrix_qk::launch<true>:nullptr;
#endif
#else
        const auto launch=variant==1u?qrt_compact_matrix_queue_qk::launch<false>:
            variant==2u?qrt_compact_matrix_queue_qk::launch<true>:
            variant==3u?qrt_compact_matrix_queue_qk::launch<true,true>:nullptr;
#endif
        if(!launch)throw std::runtime_error("invalid compact matrix QK variant");
        check(hipError_t(launch(&workspace,q,kt,scores,nullptr,start,count,stride,tokens)));
    }
    const qrt_native_exp2_workspace::Workspace owner{const_cast<unsigned char*>(packed),exp};
    check(hipError_t(qrt_fused_probability_pv::launch(&owner,scores,v,p,s,
        out.output.as<float>(),out.error.as<float>(),out.accumulator.as<float>(),out.denominator.as<float>(),
        start,count,0u,stride,exp,rcp,true,nullptr)));
}
void replay(const uint16_t* v,const uint16_t* vt,AttentionOutputs& out,
    unsigned start,unsigned count,unsigned tokens,const unsigned char* rcp){
    const unsigned stride=start+count;
    auto* p=out.tensor.probability.as<uint16_t>()+guard;
    auto* s=out.tensor.scales.as<float>()+guard;
    check(hipError_t(launch_compacted_pv_replay(v,p,s,out.output.as<float>(),start,count,0u,stride,
        rcp,out.accumulator.as<float>(),out.denominator.as<float>(),out.error.as<float>(),
        out.indices.as<unsigned>(),out.count.as<unsigned>(),nullptr,nullptr,vt,tokens,0u,nullptr,true)));
}

void run_safety(const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp){
    struct Shape{unsigned tokens,start,count;};
    const Shape shapes[]={{1,0,1},{17,0,17},{33,1,32},{65,17,33},{129,1,128},
        {257,127,128},{513,385,128},{8192,8064,128}};
    unsigned cases=0u;
    for(const auto shape:shapes)for(unsigned mode=0u;mode<10u;++mode){
        const unsigned n=shape.tokens,start=shape.start,count=shape.count;
        std::vector<uint16_t> q(size_t(n)*4096u),k(size_t(n)*512u),v(k.size());
        for(size_t i=0;i<q.size();++i)q[i]=uint16_t(((i*37u+i/19u)&0x807fu)|((123u+i%8u)<<7u));
        for(size_t i=0;i<k.size();++i){
            k[i]=uint16_t(((i*53u+i/23u)&0x807fu)|((121u+i%10u)<<7u));
            v[i]=uint16_t(((i*71u+i/17u)&0x807fu)|((120u+i%12u)<<7u));
        }
        if(mode==1u){
            for(size_t i=0;i<q.size();i+=7u)q[i]=i%3u?0u:0x8000u;
            for(size_t i=0;i<k.size();i+=11u)k[i]=i%3u?0u:0x8000u;
            for(size_t i=0;i<v.size();i+=13u)v[i]=i%3u?0u:0x8000u;
        }
        if(mode==2u){
            const uint16_t edges[]={1u,0x8001u,0x007fu,0x807fu,uint16_t(63u<<7u|19u),uint16_t(192u<<7u|11u),0u,0x8000u};
            for(unsigned i=0;i<8u;++i){q[size_t(start)*4096u+i]=edges[i];k[i]=edges[7u-i];}
        }
        if(mode==3u){
            std::fill(q.begin(),q.end(),uint16_t(127u<<7u|127u));
            for(size_t i=0;i<k.size();++i)k[i]=uint16_t(127u<<7u|127u|((i/16u)&1u?0x8000u:0u));
        }
        if(mode==4u)for(size_t i=0;i<q.size();++i)q[i]=i%2u?0u:0x8000u;
        if(mode>=5u && mode<=7u){
            const unsigned exponent=mode==5u?95u:159u;
            for(size_t i=0;i<q.size();++i)q[i]=uint16_t(exponent<<7u|127u);
            for(size_t i=0;i<k.size();++i)k[i]=uint16_t(exponent<<7u|127u|((mode==6u && (i&1u))?0x8000u:0u));
        }
        if(mode==8u){
            q[size_t(start)*4096u+255u]=uint16_t(94u<<7u|37u);
            k[511u]=uint16_t(160u<<7u|17u);
        }
        if(mode==9u){
            // Both rows satisfy the prior narrow domain. Their wide K16
            // exponent spans reject compact metadata and use original groups.
            std::fill(q.begin()+size_t(start)*4096u,q.begin()+size_t(start)*4096u+16u,uint16_t(95u<<7u|1u));
            q[size_t(start)*4096u+15u]=uint16_t(127u<<7u|1u);
            std::fill(k.begin(),k.begin()+16u,uint16_t(95u<<7u|1u));k[15u]=uint16_t(126u<<7u|1u);
        }
        Guarded dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),dt(k.size()*2u),vt(v.size()*2u);
        dq.put(q);dk.put(k);dv.put(v);
        Prepared prepared(dq.as<uint16_t>(),dk.as<uint16_t>(),dt.as<uint16_t>(),q.data(),k.data(),n);
        check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),n,nullptr)));finish();
        NarrowDomain domain(dq.as<uint16_t>(),dk.as<uint16_t>(),q,k,prepared,n);
        AttentionOutputs expected(n),native_control(n),actual(n);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
        attention(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,
            expected,start,count,n,exp,nullptr,rcp,true,0u,nullptr);finish();
        producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
            prepared,domain,native_control,start,count,n,exp,packed,rcp,0u);finish();
        for(unsigned variant=0u;variant<QRT_MATRIX_QK_SAFETY_VARIANTS;++variant){
            actual.reset();producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
                prepared,domain,actual,start,count,n,exp,packed,rcp,variant);finish();
            compare(native_control,actual,bad);
            replay(dv.as<uint16_t>(),vt.as<uint16_t>(),actual,start,count,n,rcp);finish();
            compare(expected,actual,bad);++cases;
        }
        native_control.guards();domain.verify();
        dq.immutable(q);dk.immutable(k);dv.immutable(v);prepared.verify();
        immutable_transpose(dt,k,n);immutable_transpose(vt,v,n);expected.guards();
        std::fprintf(stderr,QRT_MATRIX_QK_MARKER "_SAFETY tokens=%u start=%u queries=%u mode=%u pass=1\n",n,start,count,mode);
    }
    std::printf("{\"kind\":\"" QRT_MATRIX_QK_LABEL "_safety\",\"cases\":%u,\"shapes\":8,\"data_modes\":10,\"retained_control_callback_and_isolated_candidate\":true,\"pre_replay_native_surfaces_and_complete_replay\":true,\"domain_extremes_and_nearby_rejections\":true,\"lossless_and_original_rows\":true,\"forced_all_original_queue\":" QRT_MATRIX_QK_QUEUE_CONTROL QRT_MATRIX_QK_FORCE_FIELD ",\"cpu_metadata_checked\":true,\"raw_bit_mismatches\":0,\"guards_pass\":true,\"immutable_inputs\":true}\n",cases);
}

void run_capture(unsigned tokens,const char* qfile,const char* kfile,const char* vfile,const char* reference_file,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp){
    auto q=read_words(qfile,7169u*4096u),k=read_words(kfile,7169u*512u),v=read_words(vfile,7169u*512u);
    for(auto pair:{std::make_pair(&q,4096u),std::make_pair(&k,512u),std::make_pair(&v,512u)}){
        const auto old=*pair.first;
        pair.first->insert(pair.first->end(),old.begin(),old.begin()+size_t(tokens-7169u)*pair.second);
    }
    Guarded dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),dt(k.size()*2u),vt(v.size()*2u);
    dq.put(q);dk.put(k);dv.put(v);
    Prepared prepared(dq.as<uint16_t>(),dk.as<uint16_t>(),dt.as<uint16_t>(),q.data(),k.data(),tokens);
    const auto begin=std::chrono::steady_clock::now();
    check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),tokens,nullptr)));finish();
    const double transpose_ms=elapsed(begin);
    const auto reference=read_words(reference_file,7169u*4096u);Guarded dr(reference.size()*2u);dr.put(reference);
    NarrowDomain domain(dq.as<uint16_t>(),dk.as<uint16_t>(),q,k,prepared,tokens);
    AttentionOutputs expected(tokens),native_control(tokens),actual(tokens);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
    constexpr unsigned variants=QRT_MATRIX_QK_CAPTURE_VARIANTS;
    double samples[variants][3]{};uint64_t candidates[variants]{},fast_tiles[variants]{},slow_tiles[variants]{},score_cells=0u;unsigned cpu_dots=0u;
    for(unsigned start=0;start<tokens;start+=query_batch){
        const unsigned count=std::min(query_batch,tokens-start),stride=start+count;
        expected.reset();attention(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,
            expected,start,count,tokens,exp,nullptr,rcp,true,0u,nullptr);finish();
        hipLaunchKernelGGL(external_context,dim3((count*4096u+255u)/256u),dim3(256u),0u,nullptr,
            expected.output.as<float>(),dr.as<uint16_t>(),start,count,bad.as<unsigned>());check(hipGetLastError());
        hipLaunchKernelGGL(tensor_tails,dim3((expected.tensor.cells+2u*guard+255u)/256u),dim3(256u),0u,nullptr,
            expected.tensor.scores.as<uint32_t>(),expected.tensor.probability.as<uint16_t>(),expected.tensor.scales.as<uint32_t>(),
            expected.tensor.cells,expected.tensor.scale_cells,start,count,stride,bad.as<unsigned>());check(hipGetLastError());finish();
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("control differs from GB10 or writes unused tails");
        native_control.reset();producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
            prepared,domain,native_control,start,count,tokens,exp,packed,rcp,0u);finish();
        for(unsigned attempt=0;attempt<4u;++attempt)for(unsigned position=0;position<variants;++position){
            const unsigned variant=(position+start/query_batch+attempt)%variants;
            actual.reset();finish();const auto timed=std::chrono::steady_clock::now();
            producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,domain,
                actual,start,count,tokens,exp,packed,rcp,variant);
            // Only untimed warmups stop before replay. Timed calls retain the
            // original contiguous QK -> consumer -> exact-PV submission.
            if(!attempt){finish();compare(native_control,actual,bad);}
            replay(dv.as<uint16_t>(),vt.as<uint16_t>(),actual,start,count,tokens,rcp);
            finish();const double ms=elapsed(timed);if(attempt)samples[variant][attempt-1u]+=ms;
            compare(expected,actual,bad);
            if(!attempt){unsigned n=0;check(hipMemcpy(&n,actual.count.data(),4u,hipMemcpyDeviceToHost));candidates[variant]+=n;
                unsigned counts[2]{};check(hipMemcpy(counts,domain.statistics.data(),8u,hipMemcpyDeviceToHost));
                slow_tiles[variant]+=counts[0];fast_tiles[variant]+=counts[1];}
        }
        for(unsigned s=0;s<4u;++s){
            const unsigned row=s*(count-1u)/3u,head=(start/query_batch+s*5u)%16u,key=(start+row)*s/3u;
            const float cpu=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
                q.data()+(size_t(start+row)*16u+head)*256u,k.data()+(size_t(key)*2u+head/8u)*256u,256u)*kExactScale;
            uint32_t gpu=0;check(hipMemcpy(&gpu,expected.tensor.scores.as<uint32_t>()+guard+(size_t(row)*16u+head)*stride+key,4u,hipMemcpyDeviceToHost));
            if(gpu!=bits(cpu))throw std::runtime_error("control differs from CPU QK");++cpu_dots;
        }
        score_cells+=size_t(count)*16u*stride;
    }
    dq.immutable(q);dk.immutable(k);dv.immutable(v);dr.immutable(reference);prepared.verify();domain.verify();
    immutable_transpose(dt,k,tokens);immutable_transpose(vt,v,tokens);
    for(unsigned variant=1u;variant<variants;++variant){
        if(candidates[0]!=candidates[variant])throw std::runtime_error("original PV selection differs");
        if(!fast_tiles[variant] || !slow_tiles[variant])throw std::runtime_error("domain split not exercised");
    }
    for(unsigned variant=0;variant<variants;++variant){
        auto sorted=std::vector<double>(samples[variant],samples[variant]+3u);std::sort(sorted.begin(),sorted.end());
        unsigned query_cells=2u,key_cells=variant?2u:4u;
#ifdef QRT_WAVE_MATRIX_QK_CAPTURE
        if(variant>=2u){query_cells=1u;key_cells=8u;}
#endif
        std::printf("{\"kind\":\"" QRT_MATRIX_QK_LABEL "_capture\",\"tokens\":%u,\"source_capture_tokens\":7169,\"repeated_rows\":%u,\"variant\":%u,\"query_cells\":%u,\"key_cells\":%u,\"narrow_tiles\":%llu,\"original_tiles\":%llu,\"domain_classification_ms\":%.9f,\"retained_control_callback_and_isolated_candidate\":true,\"pre_replay_native_surfaces_checked_on_warmup\":true,\"cpu_metadata_checked\":true,\"query_batch\":128,\"score_slots\":%llu,\"output_cells\":%u,\"gb10_context_cells\":29364224,\"cpu_dots\":%u,\"pv_candidates\":%llu,\"completed_attention_samples_ms\":[%.9f,%.9f,%.9f],\"median_completed_attention_ms\":%.9f,\"common_preparation_ms\":%.9f,\"raw_bit_mismatches\":0,\"gb10_context_mismatches\":0,\"all_attempts_checked\":true,\"warmups_per_slab\":1,\"timed_attempts_per_slab\":3,\"original_qk_and_pv\":true,\"reference_is_compute_input\":false,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",tokens,tokens-7169u,variant,query_cells,key_cells,(unsigned long long)fast_tiles[variant],(unsigned long long)slow_tiles[variant],domain.ms+domain.matrix_preparation(variant),(unsigned long long)score_cells,tokens*4096u,cpu_dots,(unsigned long long)candidates[variant],samples[variant][0],samples[variant][1],samples[variant][2],sorted[1],prepared.ms+transpose_ms);
    }
}
} // namespace

int main(int argc,char** argv)try{
    if(argc!=4&&argc!=8)throw std::runtime_error("usage: safety EXP RCP | 7169|8192 Q K V GB10_CONTEXT EXP RCP");
    const bool safety_mode=argc==4&&!std::strcmp(argv[1],"safety");
    const unsigned tokens=argc==8&&!std::strcmp(argv[1],"7169")?7169u:argc==8&&!std::strcmp(argv[1],"8192")?8192u:0u;
    if(!safety_mode&&!tokens)throw std::runtime_error("invalid action");
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));
    if(std::strncmp(prop.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    const auto exp=read_table(argv[argc-2],delta::source::table_bytes),rcp=read_table(argv[argc-1],qrt_sm121_attention_rcp::table_bytes);
    if(!delta::source::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("table layout");
    Guarded de(exp.size()),dc(rcp.size()),dd(delta::packed_bytes);de.put(exp);dc.put(rcp);
    Device bad(4u);check(hipMemset(bad.pointer,0,4u));
    hipLaunchKernelGGL(delta::build,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data());check(hipGetLastError());finish();
    hipLaunchKernelGGL(verify_derived_exp,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data(),bad.as<unsigned>());check(hipGetLastError());finish();
    if(download<unsigned>(bad,1u)[0])throw std::runtime_error("complete EXP domain differs");
    std::vector<unsigned char> packed(delta::packed_bytes);check(hipMemcpy(packed.data(),dd.data(),packed.size(),hipMemcpyDeviceToHost));
    if(safety_mode)run_safety(de.data(),dd.data(),dc.data());
    else run_capture(tokens,argv[2],argv[3],argv[4],argv[5],de.data(),dd.data(),dc.data());
    de.immutable(exp);dc.immutable(rcp);dd.immutable(packed);return 0;
}catch(const std::exception& e){std::fprintf(stderr,QRT_MATRIX_QK_LABEL "_error=%s\n",e.what());return 2;}
