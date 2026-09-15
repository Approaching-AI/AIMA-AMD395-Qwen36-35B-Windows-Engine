// Reuse the established guarded transport, shape, original PV and independent
// wide CPU recurrence helpers. Its old fixture entry point is not executed.
#define QRT_WMMA_PV_OPERANDS_NO_MAIN
#include "wmma_pv_operands_selftest.cpp"
#include "../../native/providers/ck_fmha/coarse_pv_transfer.h"
#include <array>

namespace {
namespace transfer = qrt_coarse_pv_transfer;
__global__ void check_part_guards(const unsigned* words,size_t count,size_t live,unsigned* bad) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)
        if((i<guard||i>=guard+live)&&words[i]!=0xa5a5a5a5u)atomicAdd(bad,1u);
}
struct Parts {
    size_t capacity; Device device,bad;
    explicit Parts(const Shape& q) : capacity(size_t(q.capacity) * 4096u * ((q.maximum_stride + 127u) / 128u)),
        device((capacity * 2u + 2u * guard) * 4u),bad(4u) { reset(); }
    transfer::Part* data() { return reinterpret_cast<transfer::Part*>(device.as<float>() + guard); }
    void reset() { check(hipMemset(device.pointer, 0xa5, (capacity * 2u + 2u * guard) * 4u)); }
    void verify(size_t live) {
        check(hipMemset(bad.pointer,0,4u));
        hipLaunchKernelGGL(check_part_guards,dim3(1024u),dim3(256u),0u,nullptr,
            device.as<unsigned>(),capacity*2u+2u*guard,live*2u,bad.as<unsigned>());
        check(hipGetLastError());finish();
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("PV part guard or unused capacity");
    }
};
void coarse_produce(unsigned variant, Stage& s, Parts& parts, const uint16_t* v, const unsigned char* rcp) {
    const auto& q = s.shape;
    if (!variant) {
        hipLaunchKernelGGL((blackwell_mantissa_value_kernel<true,false,true,true,true>),
            dim3(2u,16u,(q.queries+15u)/16u),dim3(256u),0u,nullptr,
            v,s.p.as<uint16_t>()+guard,s.s.as<float>()+guard,s.o.as<float>()+guard,
            q.start,q.queries,q.output_start,q.stride,rcp,s.a.as<float>()+guard,
            s.d.as<float>()+guard,nullptr,nullptr,s.e.as<float>()+guard);
        check(hipGetLastError()); return;
    }
#define COARSE_PV_CASE(n,k) if(variant==n) check(hipError_t(transfer::launch<k>(v,s.p.as<uint16_t>()+guard,s.s.as<float>()+guard,parts.data(),parts.capacity,s.o.as<float>()+guard,s.e.as<float>()+guard,s.a.as<float>()+guard,s.d.as<float>()+guard,q.start,q.queries,q.output_start,q.stride,rcp,nullptr)))
    COARSE_PV_CASE(1u,128u);
    else COARSE_PV_CASE(2u,256u);
    else COARSE_PV_CASE(3u,512u);
    else throw std::runtime_error("coarse PV variant");
#undef COARSE_PV_CASE
}
__global__ void all_indices(unsigned* ids, unsigned* count, unsigned cells) {
    const unsigned cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell < cells) ids[cell] = cell;
    if (!cell) *count = cells;
}
void reset_indices(Device& ids, const Shape& q) {
    check(hipMemset(ids.pointer,0xa5,(size_t(q.capacity)*4096u+1u+2u*guard)*4u));
    check(hipMemset(ids.as<unsigned>()+guard+q.cells(),0,4u));
}
void queue_collect(Stage& s, Device& ids) {
    const auto& q=s.shape;
    hipLaunchKernelGGL(blackwell_collect_pv_replay_kernel,dim3((q.cells()+255u)/256u),dim3(256u),0u,nullptr,
        s.o.as<float>()+guard,s.e.as<float>()+guard,q.output_start,q.cells(),
        ids.as<unsigned>()+guard,ids.as<unsigned>()+guard+q.cells());check(hipGetLastError());
}
void queue_exact(Stage& s,Device& ids,const uint16_t* v,const uint16_t* tv,unsigned n,const unsigned char* rcp) {
    const auto& q=s.shape;
    hipLaunchKernelGGL((blackwell_compacted_pv_replay_kernel<true>),dim3(std::min(1024u,(q.cells()+63u)/64u)),dim3(256u),0u,nullptr,
        v,s.p.as<uint16_t>()+guard,s.s.as<float>()+guard,s.o.as<float>()+guard,q.start,q.output_start,q.stride,rcp,
        s.a.as<float>()+guard,s.d.as<float>()+guard,ids.as<unsigned>()+guard,
        ids.as<unsigned>()+guard+q.cells(),tv,n,0u);check(hipGetLastError());
}
std::vector<unsigned> verify_indices(Device& ids,const Shape& q) {
    const auto data=download<unsigned>(ids,size_t(q.capacity)*4096u+1u+2u*guard);
    const unsigned count=data[guard+q.cells()];
    if(count>q.cells())throw std::runtime_error("coarse PV candidate capacity");
    std::vector<unsigned> result(data.begin()+guard,data.begin()+guard+count);
    std::sort(result.begin(),result.end());
    for(unsigned i=0u;i<count;++i)if(result[i]>=q.cells() || (i&&result[i]==result[i-1u]))
        throw std::runtime_error("coarse PV candidate identity");
    for(size_t i=0u;i<data.size();++i)
        if((i<guard || (i>=guard+count&&i<guard+q.cells()) || i>guard+q.cells())&&data[i]!=0xa5a5a5a5u)
            throw std::runtime_error("coarse PV candidate guard");
    return result;
}
void copy_probability(Stage& to,Stage& from) {
    check(hipMemcpy(to.p.pointer,from.p.pointer,from.shape.pwords()*2u,hipMemcpyDeviceToDevice));
    check(hipMemcpy(to.s.pointer,from.s.pointer,from.shape.swords()*4u,hipMemcpyDeviceToDevice));
}
void stage_guards(Stage& s) {
    const auto& q=s.shape;
    guards(s.o,q.owords(),guard+q.output_start*4096u,q.cells());
    guards(s.a,q.owords(),guard+q.output_start*4096u,q.cells());
    guards(s.d,q.dwords(),guard+q.output_start*16u,q.queries*16u);
    guards(s.e,q.ewords(),guard,q.cells());
}
void coarse_generated(unsigned start,unsigned queries,unsigned mode,bool vllm,
    const unsigned char* exp,const unsigned char* rcp) {
    Shape q{start,queries,start+queries,queries+3u,start+queries+7u,3u};
    std::vector<float> scores(q.pwords(),12345.0f);
    std::vector<uint16_t> values(size_t(q.stride)*512u+2u*guard,0x5a5au),tv(values.size(),0x5a5au);
    for(unsigned row=0u;row<queries*16u;++row)for(unsigned key=0u;key<start+row/16u+1u;++key)
        scores[guard+size_t(row)*q.stride+key]=score_value(row,key,mode);
    for(unsigned key=0u;key<q.stride;++key)for(unsigned feature=0u;feature<512u;++feature) {
        const size_t i=size_t(key)*512u+feature;
        uint16_t word=qrt_sm121_pv_bound::bf16(float(int((i*173u+i/17u)%63u)-31)/64.0f);
        if(mode==1u&&i%7u==0u)word=uint16_t(i%127u+1u);
        if(mode==2u&&i%3u==0u)word=uint16_t((i&1u?0x8000u:0u)|0x5f80u);
        if(mode==3u&&i%3u==0u)word=(i&1u)?0x8000u:0u;
        if(mode==4u)word=uint16_t((i&1u?0x8000u:0u)|(i%3u?0x0101u:0x0001u));
        if(mode==5u&&i%251u==0u)word=(i&1u)?0xffc1u:0x7fc1u;
        values[guard+i]=word;tv[guard+size_t(feature)*q.stride+key]=word;
    }
    Device ds(scores.size()*4u),dv(values.size()*2u),dvt(tv.size()*2u),bad(4u);
    upload(ds,scores);upload(dv,values);upload(dvt,tv);
    Stage original(q),candidate(q);Parts parts(q);
    Device ids((size_t(q.capacity)*4096u+1u+2u*guard)*4u);
    hipLaunchKernelGGL(blackwell_online_probability_kernel,dim3(16u,q.queries),dim3(32u),0u,nullptr,
        ds.as<float>()+guard,original.p.as<uint16_t>()+guard,original.s.as<float>()+guard,
        q.start,q.stride,exp,vllm);check(hipGetLastError());
    coarse_produce(0u,original,parts,dv.as<uint16_t>()+guard,rcp);
    reset_indices(ids,q);
    hipLaunchKernelGGL(all_indices,dim3((q.cells()+255u)/256u),dim3(256u),0u,nullptr,
        ids.as<unsigned>()+guard,ids.as<unsigned>()+guard+q.cells(),q.cells());check(hipGetLastError());
    queue_exact(original,ids,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,q.stride,rcp);finish();
    verify_indices(ids,q);packed_expected(original);
    const auto expected=download<float>(original.o,q.owords()),raw=download<float>(original.a,q.owords());
    const auto hp=download<uint16_t>(original.p,q.pwords());const auto hs=download<float>(original.s,q.swords());
    unsigned cpu=0u;
    for(unsigned sample=0u;sample<8u;++sample) {
        const unsigned cell=sample*(q.cells()-1u)/7u,row=cell/256u,head=row%16u;
        const float ref=cpu_pv(hp.data()+guard+size_t(row)*q.stride,values.data()+guard,
            hs.data()+guard+size_t(row)*((q.stride+31u)/32u+1u),q.start+row/16u+1u,(head/8u)*256u+cell%256u);
        if(bits(ref)!=bits(raw[guard+q.output_start*4096u+cell]))throw std::runtime_error("coarse PV CPU recurrence");
        ++cpu;
    }
    for(unsigned variant=1u;variant<=3u;++variant) {
        candidate.reset();copy_probability(candidate,original);parts.reset();reset_indices(ids,q);finish();
        coarse_produce(variant,candidate,parts,dv.as<uint16_t>()+guard,rcp);finish();
        const auto approximation=download<float>(candidate.o,q.owords()),errors=download<float>(candidate.e,q.ewords());
        size_t finite=0u,certified=0u;
        for(unsigned cell=0u;cell<q.cells();++cell) {
            const unsigned out=guard+q.output_start*4096u+cell;const float error=errors[guard+cell];
            if(std::isfinite(error)) {
                ++finite;
                if(!std::isfinite(expected[out])||std::abs(double(expected[out])-double(approximation[out]))>double(error))
                    throw std::runtime_error("coarse PV interval undercoverage");
            }
            if(qrt_sm121_pv_bound::same_bf16(approximation[out],error))++certified;
        }
        queue_collect(candidate,ids);queue_exact(candidate,ids,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,q.stride,rcp);finish();
        const auto selected=verify_indices(ids,q);const auto actual=download<float>(candidate.o,q.owords());
        if(selected.size()+certified!=q.cells())throw std::runtime_error("coarse PV admission count");
        for(unsigned cell=0u;cell<q.cells();++cell) {
            const unsigned out=guard+q.output_start*4096u+cell;
            if(qrt_sm121_pv_bound::bf16(actual[out])!=qrt_sm121_pv_bound::bf16(expected[out]))
                throw std::runtime_error("coarse PV corrected BF16");
        }
        const unsigned chunk=64u<<variant;parts.verify(size_t(q.cells())*((q.stride+chunk-1u)/chunk));
        stage_guards(candidate);unchanged(candidate.p,hp);unchanged(candidate.s,hs);
        std::printf("{\"kind\":\"coarse_pv_transfer_safety\",\"start\":%u,\"queries\":%u,\"mode\":%u,\"vllm_sum\":%s,\"chunk\":%u,\"cells\":%u,\"finite_intervals\":%zu,\"certified\":%zu,\"selected\":%zu,\"cpu_dots\":%u,\"interval_undercoverage\":0,\"bf16_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
            start,queries,mode,vllm?"true":"false",chunk,q.cells(),finite,certified,selected.size(),cpu);std::fflush(stdout);
    }
    unchanged(ds,scores);unchanged(dv,values);unchanged(dvt,tv);stage_guards(original);
}
void extend_capture(std::vector<uint16_t>& values,unsigned columns,unsigned tokens) {
    if(tokens<7169u||tokens>8192u)throw std::runtime_error("coarse PV capture token count");
    const std::vector<uint16_t> last(values.end()-columns,values.end());
    for(unsigned row=7169u;row<tokens;++row)values.insert(values.end(),last.begin(),last.end());
    values.insert(values.begin(),guard,0x5a5au);values.insert(values.end(),guard,0x5a5au);
}
void coarse_capture(unsigned tokens,const char* qfile,const char* kfile,const char* vfile,
    const char* reference_file,const char* exp_file,const char* rcp_file) {
    constexpr unsigned batch=128u;
    auto hq=read_values<uint16_t>(qfile,size_t(7169u)*4096u);
    auto hk=read_values<uint16_t>(kfile,size_t(7169u)*512u),hv=read_values<uint16_t>(vfile,size_t(7169u)*512u);
    const auto golden=read_values<uint16_t>(reference_file,size_t(7169u)*4096u);
    const auto exp=read_values<unsigned char>(exp_file,exp2_backend::table_bytes);
    const auto rcp=read_values<unsigned char>(rcp_file,qrt_sm121_attention_rcp::table_bytes);
    if(!exp2_backend::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))
        throw std::runtime_error("coarse PV table layout");
    extend_capture(hq,4096u,tokens);extend_capture(hk,512u,tokens);extend_capture(hv,512u,tokens);
    Device dq(hq.size()*2u),dk(hk.size()*2u),dv(hv.size()*2u),dkt(hk.size()*2u),dvt(hv.size()*2u),dex(exp.size()),drcp(rcp.size());
    upload(dq,hq);upload(dk,hk);upload(dv,hv);upload(dex,exp);upload(drcp,rcp);
    check(hipMemset(dkt.pointer,0x5a,hk.size()*2u));check(hipMemset(dvt.pointer,0x5a,hv.size()*2u));finish();
    std::vector<uint16_t> tv(hv.size(),0x5a5au);
    for(unsigned key=0u;key<tokens;++key)for(unsigned f=0u;f<512u;++f)
        tv[guard+size_t(f)*tokens+key]=hv[guard+size_t(key)*512u+f];
    auto begin=std::chrono::steady_clock::now();
    check(hipError_t(transpose_keys(dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,size_t(tokens)*512u,tokens,nullptr)));
    finish();const double transpose_ms=elapsed(begin);unchanged(dvt,tv);
    Prepared prepared(dq.as<uint16_t>()+guard,dk.as<uint16_t>()+guard,dkt.as<uint16_t>()+guard,hq.data()+guard,hk.data()+guard,tokens);
    const auto kt_before=download<uint16_t>(dkt,hk.size());
    Shape q{0u,batch,batch,batch,tokens,0u};Stage original(q),candidate(q);Parts parts(q);
    Device scores(q.pwords()*4u),score_copy(q.pwords()*4u),bad(4u);
    Device ids((size_t(batch)*4096u+1u+2u*guard)*4u);
    double samples[4][3]{},maximum[4]{},qk_ms=0.0,probability_ms=0.0;
    size_t endpoints=0u,score_cells=0u,verified[4]{},external[4]{},selected_total[4]{},cpu[4]{};
    for(unsigned start=0u;start<tokens;start+=batch) {
        q.start=start;q.queries=std::min(batch,tokens-start);q.stride=start+q.queries;original.shape=q;candidate.shape=q;
        check(hipMemset(scores.pointer,0xa5,q.pwords()*4u));finish();begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::scores<128u,true,16u,16u>),
            dim3((q.stride+15u)/16u,16u,(q.queries+15u)/16u),dim3(256u),0u,nullptr,
            dq.as<uint16_t>()+guard,dkt.as<uint16_t>()+guard,prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,
            prepared.qf.as<unsigned>()+guard,prepared.kf.as<unsigned>()+guard,scores.as<float>()+guard,start,q.queries,q.stride,tokens);
        check(hipGetLastError());finish();qk_ms+=elapsed(begin);
        check(hipMemcpy(score_copy.pointer,scores.pointer,q.pwords()*4u,hipMemcpyDeviceToDevice));
        original.reset();reset_indices(ids,q);finish();begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL(blackwell_online_probability_kernel,dim3(16u,q.queries),dim3(32u),0u,nullptr,
            scores.as<float>()+guard,original.p.as<uint16_t>()+guard,original.s.as<float>()+guard,
            q.start,q.stride,dex.as<unsigned char>(),true);check(hipGetLastError());finish();probability_ms+=elapsed(begin);
        coarse_produce(0u,original,parts,dv.as<uint16_t>()+guard,drcp.as<unsigned char>());
        queue_collect(original,ids);queue_exact(original,ids,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>());finish();
        verify_indices(ids,q);packed_expected(original);
        const auto expected=download<float>(original.o,q.owords());
        const auto expected_den=download<float>(original.d,q.dwords());
        const auto hp=download<uint16_t>(original.p,q.pwords());const auto hs=download<float>(original.s,q.swords());
        for(unsigned cell=0u;cell<q.cells()&&size_t(start)*4096u+cell<golden.size();++cell)
            if(qrt_sm121_pv_bound::bf16(expected[guard+cell])!=golden[size_t(start)*4096u+cell])
                throw std::runtime_error("coarse PV original GB10 boundary");
        for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<4u;++position) {
            const unsigned variant=(position+start/batch+attempt)%4u;
            candidate.reset();copy_probability(candidate,original);parts.reset();reset_indices(ids,q);finish();
            begin=std::chrono::steady_clock::now();
            coarse_produce(variant,candidate,parts,dv.as<uint16_t>()+guard,drcp.as<unsigned char>());
            queue_collect(candidate,ids);
            queue_exact(candidate,ids,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>());finish();
            const double wall=elapsed(begin);
            if(attempt){samples[variant][attempt-1u]+=wall;maximum[variant]=std::max(maximum[variant],wall);}
            const auto selected=verify_indices(ids,q);const auto output=download<float>(candidate.o,q.owords());
            const auto raw=download<float>(candidate.a,q.owords());
            const auto den=download<float>(candidate.d,q.dwords());
            if(std::memcmp(den.data(),expected_den.data(),den.size()*4u))throw std::runtime_error("coarse PV denominator parity");
            for(unsigned cell=0u;cell<q.cells();++cell) {
                if(qrt_sm121_pv_bound::bf16(output[guard+cell])!=qrt_sm121_pv_bound::bf16(expected[guard+cell]))
                    throw std::runtime_error("coarse PV complete corrected BF16 mismatch");
                if(size_t(start)*4096u+cell<golden.size()){
                    if(qrt_sm121_pv_bound::bf16(output[guard+cell])!=golden[size_t(start)*4096u+cell])
                        throw std::runtime_error("coarse PV external GB10 mismatch");
                    ++external[variant];
                }
            }
            if(!attempt){
                selected_total[variant]+=selected.size();
                const unsigned count=unsigned(std::min(size_t(4u),selected.size()));
                for(unsigned i=0u;i<count;++i){
                    const unsigned cell=selected[size_t(i)*(selected.size()-1u)/std::max(1u,count-1u)],row=cell/256u,head=row%16u;
                    const float ref=cpu_pv(hp.data()+guard+size_t(row)*q.stride,hv.data()+guard,
                        hs.data()+guard+size_t(row)*((q.stride+31u)/32u+1u),start+row/16u+1u,(head/8u)*256u+cell%256u);
                    if(bits(ref)!=bits(raw[guard+cell]))throw std::runtime_error("coarse PV captured independent CPU recurrence");++cpu[variant];
                }
            }
            const unsigned chunk=variant?64u<<variant:128u;
            parts.verify(variant?size_t(q.cells())*((q.stride+chunk-1u)/chunk):0u);
            stage_guards(candidate);unchanged(candidate.p,hp);unchanged(candidate.s,hs);
            verified[variant]+=q.cells();
        }
        check(hipMemset(bad.pointer,0,4u));compare_device(score_copy,scores,q.pwords(),bad);finish();
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("coarse PV scores changed");
        endpoints+=q.cells();score_cells+=size_t(q.queries)*16u*q.stride;
    }
    unchanged(dq,hq);unchanged(dk,hk);unchanged(dv,hv);unchanged(dkt,kt_before);unchanged(dvt,tv);
    unchanged(dex,exp);unchanged(drcp,rcp);prepared.verify();
    if(endpoints!=size_t(tokens)*4096u)throw std::runtime_error("coarse PV incomplete capture");
    for(unsigned variant=0u;variant<4u;++variant){
        std::array<double,3> sorted{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(sorted.begin(),sorted.end());
        std::printf("{\"kind\":\"coarse_pv_transfer_capture\",\"tokens\":%u,\"variant\":%u,\"chunk\":%u,\"query_batch\":128,\"unique_output_cells\":%zu,\"verified_output_cells\":%zu,\"external_gb10_cells_checked\":%zu,\"external_unique_gb10_cells\":29364224,\"synthetic_tail_rows\":%u,\"selected\":%zu,\"cpu_dots\":%zu,\"probability_cells\":%zu,\"qk_ms\":%.6f,\"qk_preparation_ms\":%.6f,\"probability_ms\":%.6f,\"value_transpose_ms\":%.6f,\"producer_collect_replay_median_ms\":%.6f,\"samples_ms\":[%.6f,%.6f,%.6f],\"maximum_completed_slab_ms\":%.6f,\"workspace_bytes\":%zu,\"warmups\":1,\"samples\":3,\"bf16_mismatches\":0,\"denominator_bit_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"all_attempts_checked\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            tokens,variant,variant?64u<<variant:0u,endpoints,verified[variant],external[variant],tokens-7169u,
            selected_total[variant],cpu[variant],score_cells,qk_ms,prepared.ms,probability_ms,transpose_ms,
            sorted[1],samples[variant][0],samples[variant][1],samples[variant][2],maximum[variant],
            variant?size_t(batch)*4096u*((tokens+(64u<<variant)-1u)/(64u<<variant))*sizeof(transfer::Part):0u);std::fflush(stdout);
    }
}
} // namespace

int main(int argc,char** argv)try {
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));
    if(std::strncmp(prop.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    if(argc==8&&(!std::strcmp(argv[1],"--q7169")||!std::strcmp(argv[1],"--q8192"))){
        coarse_capture(!std::strcmp(argv[1],"--q7169")?7169u:8192u,argv[2],argv[3],argv[4],argv[5],argv[6],argv[7]);return 0;
    }
    if(argc!=4||std::strcmp(argv[1],"--selftest"))throw std::runtime_error("use --selftest exp2 rcp or --q7169/--q8192 Q K V reference exp2 rcp");
    const auto exp=read_values<unsigned char>(argv[2],exp2_backend::table_bytes);
    const auto rcp=read_values<unsigned char>(argv[3],qrt_sm121_attention_rcp::table_bytes);
    if(!exp2_backend::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))
        throw std::runtime_error("coarse PV safety table layout");
    Device dex(exp.size()),drcp(rcp.size());upload(dex,exp);upload(drcp,rcp);
    for(auto shape:{std::pair<unsigned,unsigned>{0u,1u},{31u,2u},{17u,17u},{127u,3u},{511u,17u},{8191u,1u}})
        for(unsigned mode=0u;mode<6u;++mode)for(bool vllm:{false,true})
            coarse_generated(shape.first,shape.second,mode,vllm,dex.as<unsigned char>(),drcp.as<unsigned char>());
    unchanged(dex,exp);unchanged(drcp,rcp);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"coarse_pv_transfer_error=%s\n",e.what());return 2;}
