#pragma once
#include "projection_strong_replay_suite.h"
#include "../../native/providers/moe_accumulator/sm121_folded_half_projection.h"
namespace projection_safety_test {
namespace folded_projection = qrt_sm121_folded_half_projection;
namespace folded_products = qrt_sm121_folded_half_products;
using FoldedRow = folded_projection::Row;
using LegacyFoldedRow = qrt_sm121_staged_half_projection::Row;
template<unsigned Variant, bool Audit=false>
__global__ void folded_half_candidate_kernel(const LegacyFoldedRow* old_w,const LegacyFoldedRow* old_x,
    const FoldedRow* new_w,const FoldedRow* new_x,const unsigned* indices,float* output,
    unsigned rows,unsigned width,unsigned count,folded_projection::Stats* stats=nullptr,uint32_t* trace=nullptr) {
    const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;
    if(slot>=count)return;
    const unsigned cell=indices[slot],token=cell/rows,row=cell%rows,groups=width/16u;
    float value;folded_projection::Stats counts;
    if constexpr(Variant==0u) value=qrt_sm121_staged_half_projection::dot<2u>(
        old_x+size_t(token)*groups,old_w+size_t(row)*groups,width);
    else value=folded_projection::dot<(1u<<(Variant-1u)),Audit>(
        new_x+size_t(token)*groups,new_w+size_t(row)*groups,width,
        trace?trace+size_t(slot)*groups*3u:nullptr,Audit?&counts:nullptr);
    if(!(threadIdx.x&3u)) {output[cell]=value;if constexpr(Audit) if(stats)stats[slot]=counts;}
}
void run_folded_half_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,DeviceBuffer<float>& dout,
    const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,const std::vector<uint16_t>& reference,
    const std::vector<float>& initial,const std::vector<unsigned>& selected,unsigned rows,unsigned tokens,unsigned width) {
    require(!selected.empty()&&std::is_sorted(selected.begin(),selected.end())&&
        std::adjacent_find(selected.begin(),selected.end())==selected.end(),"folded candidate identity");
    const size_t cells=size_t(rows)*tokens,wg=size_t(rows)*(width/16u),ig=size_t(tokens)*(width/16u);
    constexpr unsigned marker=0xa5a5a5a5u;
    LegacyFoldedRow old_guard;FoldedRow new_guard;
    std::memset(&old_guard,0xa5,sizeof(old_guard));std::memset(&new_guard,0xa5,sizeof(new_guard));
    std::vector<LegacyFoldedRow> ow(wg+2u*kGuard,old_guard),ox(ig+2u*kGuard,old_guard);
    std::vector<FoldedRow> nw(wg+2u*kGuard,new_guard),nx(ig+2u*kGuard,new_guard);
    DeviceBuffer<LegacyFoldedRow> dow(ow),dox(ox);DeviceBuffer<FoldedRow> dnw(nw),dnx(nx);
    std::vector<unsigned> ids(selected.size()+2u*kGuard,marker);
    std::copy(selected.begin(),selected.end(),ids.begin()+kGuard);DeviceBuffer<unsigned> dids(ids);
    std::vector<float> control;double samples[5][3]{};
    const auto verify_sources=[&] {
        auto w=weights,x=inputs;auto index=ids;dw.read(w);di.read(x);dids.read(index);
        require(w==weights&&x==inputs&&index==ids,"folded source or index changed");
    };
    const auto verify_output=[&] {
        auto result=initial;dout.read(result);require(!control.empty(),"folded missing baseline");
        for(size_t i=0u;i<cells;++i) {
            require(std::isfinite(result[kGuard+i]),"folded nonfinite result");
            require(!std::memcmp(&result[kGuard+i],&control[kGuard+i],4u),"folded raw endpoint differs");
            require(bf16(result[kGuard+i])==reference[kGuard+i],"folded GB10 endpoint differs");
        }
        for(size_t i=0u;i<kGuard;++i)
            require(result[i]==kF32Guard&&result[kGuard+cells+i]==kF32Guard,"folded output guard");
    };
    const auto verify_prepared=[&](bool newer) {
        if(newer){dnw.read(nw);dnx.read(nx);}else{dow.read(ow);dox.read(ox);}
        for(unsigned side=0u;side<2u;++side) {
            const auto& source=side?inputs:weights;const size_t groups=side?ig:wg;
            for(size_t group=0u;group<groups;++group) {
                if(newer) {
                    const auto& actual=(side?nx:nw)[kGuard+group];
                    const auto expected=folded_products::prepare(source.data()+kGuard+group*16u);
                    require(!std::memcmp(&actual,&expected,sizeof(expected)),"folded prepared metadata changed");
                    for(unsigned i=0u;i<16u;++i)require(folded_products::original(actual,i)==source[kGuard+group*16u+i],"folded roundtrip");
                }else {
                    const auto expected=qrt_sm121_scaled_half_products::prepare(source.data()+kGuard+group*16u);
                    require(!std::memcmp(&(side?ox:ow)[kGuard+group],&expected,sizeof(expected)),"folded control metadata changed");
                }
            }
            for(size_t i=0u;i<kGuard;++i) {
                if(newer)require(!std::memcmp(&(side?nx:nw)[i],&new_guard,sizeof(new_guard))&&
                    !std::memcmp(&(side?nx:nw)[kGuard+groups+i],&new_guard,sizeof(new_guard)),"folded packed guard");
                else require(!std::memcmp(&(side?ox:ow)[i],&old_guard,sizeof(old_guard))&&
                    !std::memcmp(&(side?ox:ow)[kGuard+groups+i],&old_guard,sizeof(old_guard)),"folded baseline packed guard");
            }
        }
    };
    for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<5u;++position) {
        const unsigned variant=(attempt+position)%5u;
        hip_ok(hipMemcpy(dout.base,initial.data(),initial.size()*4u,hipMemcpyHostToDevice),"folded reset");
        complete_strong_projection();const auto start=std::chrono::steady_clock::now();
        if(!variant) {
            hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,dw.data(),dow.data(),rows,width);hip_ok(hipGetLastError(),"folded old weight preparation");
            hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,di.data(),dox.data(),tokens,width);hip_ok(hipGetLastError(),"folded old input preparation");
        }else {
            hipLaunchKernelGGL(folded_projection::prepare_rows,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,dw.data(),dnw.data(),rows,width);hip_ok(hipGetLastError(),"folded weight preparation");
            hipLaunchKernelGGL(folded_projection::prepare_rows,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,di.data(),dnx.data(),tokens,width);hip_ok(hipGetLastError(),"folded input preparation");
        }
        for(unsigned offset=0u;offset<selected.size();offset+=262144u) {
            const unsigned count=std::min(262144u,unsigned(selected.size())-offset);
#define QRT_FOLDED_CASE(v) if(variant==v)hipLaunchKernelGGL((folded_half_candidate_kernel<v>),dim3((count+63u)/64u),dim3(256u),0u,nullptr,dow.data(),dox.data(),dnw.data(),dnx.data(),dids.data()+offset,dout.data(),rows,width,count)
            QRT_FOLDED_CASE(0u);QRT_FOLDED_CASE(1u);QRT_FOLDED_CASE(2u);QRT_FOLDED_CASE(3u);QRT_FOLDED_CASE(4u);
#undef QRT_FOLDED_CASE
            hip_ok(hipGetLastError(),"folded bounded replay");
        }
        complete_strong_projection();if(attempt)samples[variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(!attempt&&!variant){control=initial;dout.read(control);}
        verify_output();verify_prepared(variant!=0u);verify_sources();
    }
    // Separate audit: count every selected K16 group, then compare every
    // original carry of256 spaced candidates against the wide CPU primitive.
    folded_projection::Stats stats_guard;std::memset(&stats_guard,0xa5,sizeof(stats_guard));
    std::vector<folded_projection::Stats> stats(selected.size()+2u*kGuard,stats_guard);
    DeviceBuffer<folded_projection::Stats> dstats(stats);
    for(unsigned offset=0u;offset<selected.size();offset+=262144u) {
        const unsigned count=std::min(262144u,unsigned(selected.size())-offset);
        hipLaunchKernelGGL((folded_half_candidate_kernel<2u,true>),dim3((count+63u)/64u),dim3(256u),0u,nullptr,dow.data(),dox.data(),dnw.data(),dnx.data(),dids.data()+offset,dout.data(),rows,width,count,dstats.data()+offset);
        hip_ok(hipGetLastError(),"folded count audit");
    }
    complete_strong_projection();dstats.read(stats);verify_output();
    size_t folded=0u,fallback=0u;
    for(size_t i=0u;i<selected.size();++i){const auto s=stats[kGuard+i];require(s.folded+s.fallback==width/16u,"folded audit count");folded+=s.folded;fallback+=s.fallback;}
    for(size_t i=0u;i<kGuard;++i)require(!std::memcmp(&stats[i],&stats_guard,sizeof(stats_guard))&&
        !std::memcmp(&stats[kGuard+selected.size()+i],&stats_guard,sizeof(stats_guard)),"folded stats guard");
    const unsigned sampled=std::min(256u,unsigned(selected.size())),groups=width/16u;
    std::vector<unsigned> sample_ids(sampled+2u*kGuard,marker);
    for(unsigned i=0u;i<sampled;++i)sample_ids[kGuard+i]=selected[sampled>1u?size_t(i)*(selected.size()-1u)/(sampled-1u):0u];
    DeviceBuffer<unsigned> sample_index(sample_ids);
    std::vector<uint32_t> trace(size_t(sampled)*groups*3u+2u*kGuard,marker);DeviceBuffer<uint32_t> dtrace(trace);
    hipLaunchKernelGGL((folded_half_candidate_kernel<2u,true>),dim3((sampled+63u)/64u),dim3(256u),0u,nullptr,dow.data(),dox.data(),dnw.data(),dnx.data(),sample_index.data(),dout.data(),rows,width,sampled,nullptr,dtrace.data());
    hip_ok(hipGetLastError(),"folded carry audit");complete_strong_projection();dtrace.read(trace);
    for(unsigned sample=0u;sample<sampled;++sample) {
        const unsigned cell=sample_ids[kGuard+sample];qrt_q1_moe_hawkeye::Value carry{0u,-133,false};
        for(unsigned group=0u;group<groups;++group) {
            qrt_q1_moe_hawkeye::Value terms[17];terms[0]=carry;
            for(unsigned i=0u;i<16u;++i)terms[1u+i]=qrt_q1_moe_hawkeye::multiply_bf16(
                inputs[kGuard+size_t(cell/rows)*width+group*16u+i],weights[kGuard+size_t(cell%rows)*width+group*16u+i],-133);
            carry=qrt_q1_moe_hawkeye::group_sum<26,-133>(terms,17u);
            const size_t at=kGuard+(size_t(sample)*groups+group)*3u;
            require(trace[at]==carry.significand&&trace[at+1u]==uint32_t(int32_t(carry.exponent))&&trace[at+2u]==unsigned(carry.negative),"folded independent CPU carry differs");
        }
    }
    for(size_t i=0u;i<kGuard;++i)require(trace[i]==marker&&trace[trace.size()-kGuard+i]==marker,"folded trace guard");
    auto copied_ids=sample_ids;sample_index.read(copied_ids);require(copied_ids==sample_ids,"folded audit index changed");
    verify_output();verify_prepared(true);verify_sources();
    for(unsigned variant=0u;variant<5u;++variant) {
        std::array<double,3> ordered{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(ordered.begin(),ordered.end());
        std::cout<<"{\"type\":\"folded_half_projection_replay\",\"variant\":"<<variant<<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width<<",\"elements\":"<<cells<<",\"candidates\":"<<selected.size()<<",\"staging_groups\":"<<(variant?(1u<<(variant-1u)):2u)<<",\"prepared_operand_bytes\":"<<(wg+ig)*sizeof(FoldedRow)<<",\"preparation_replay_ms\":"<<ordered[1]<<",\"samples_ms\":["<<samples[variant][0]<<','<<samples[variant][1]<<','<<samples[variant][2]<<"],\"warmups\":1,\"measured_attempts\":3,\"audit_folded_groups\":"<<folded<<",\"audit_fallback_groups\":"<<fallback<<",\"cpu_original_carry_checks\":"<<size_t(sampled)*groups<<",\"cpu_dots\":"<<sampled<<",\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"all_attempts_verified\":true,\"all_encoded_words_checked\":true,\"production_audit_parity\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
    }
}
}
