#pragma once
#include "projection_strong_replay_suite.h"
#include "../../native/providers/moe_accumulator/sm121_staged_half_projection.h"
#include "../../native/providers/moe_accumulator/sm121_coarse_projection_matrix.h"
namespace projection_safety_test {
namespace coarse_matrix=qrt_sm121_coarse_projection_matrix;
namespace coarse_bound=qrt_sm121_coarse_projection_bound;
using CoarseRow=qrt_sm121_scaled_half_projection::Row;
template<bool Indexed>
__global__ void coarse_original_replay(const CoarseRow* weights,const CoarseRow* inputs,
    const unsigned* indices,float* output,unsigned rows,unsigned width,unsigned offset,unsigned count) {
    const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;if(slot>=count)return;
    const unsigned cell=Indexed?indices[offset+slot]:offset+slot;
    const float value=qrt_sm121_staged_half_projection::dot<2u>(inputs+size_t(cell/rows)*(width/16u),weights+size_t(cell%rows)*(width/16u),width);
    if(!(threadIdx.x&3u))output[cell]=value;
}
__global__ void coarse_original_compact(const float* centers,const float* input_norm,const float* weight_norm,
    float* output,unsigned* indices,unsigned* count,unsigned cells,unsigned rows,unsigned ppb) {
    const unsigned cell=blockIdx.x*256u+threadIdx.x,lane=threadIdx.x%32u,wave=threadIdx.x/32u;
    const bool selected=cell<cells && selected_bf16_projection_hawkeye_candidate(centers[cell],cell,rows,512u,0u,ppb,nullptr,input_norm,weight_norm);
    const unsigned mask=__ballot(selected);__shared__ unsigned offsets[8],first;
    if(!lane)offsets[wave]=__popc(mask);__syncthreads();
    if(!threadIdx.x){unsigned total=0u;for(unsigned i=0u;i<8u;++i){const unsigned n=offsets[i];offsets[i]=total;total+=n;}first=atomicAdd(count,total);}
    __syncthreads();if(cell<cells)output[cell]=centers[cell];
    if(selected)indices[first+offsets[wave]+__popc(mask&((1u<<lane)-1u))]=cell;
}
void run_coarse_interval_projection(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
    DeviceBuffer<float>& dout,const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
    const std::vector<uint16_t>& reference,const std::vector<float>& initial,const std::vector<unsigned>& old_selected,
    unsigned rows,unsigned tokens,unsigned width,unsigned ppb) {
    const unsigned cells=rows*tokens;const size_t wg=size_t(rows)*(width/16u),ig=size_t(tokens)*(width/16u);
    require(tokens==8192u && ((rows==2048u&&width==4096u)||(rows==8192u&&width==2048u)),"coarse product shape");
    constexpr unsigned marker=0xa5a5a5a5u;CoarseRow row_guard;std::memset(&row_guard,0xa5,sizeof(row_guard));
    std::vector<CoarseRow> wp(wg+2u*kGuard,row_guard),ip(ig+2u*kGuard,row_guard);DeviceBuffer<CoarseRow> pw(wp),pi(ip);
    std::vector<unsigned> wf(rows+2u*kGuard,marker),xf(tokens+2u*kGuard,marker),ids(size_t(cells)+2u*kGuard,marker),count(1u+2u*kGuard,marker);
    DeviceBuffer<unsigned> dwf(wf),dxf(xf),dids(ids),dcount(count);
    std::vector<float> center=initial,error=initial,canonical=initial,wn(rows+2u*kGuard,kF32Guard),xn(tokens+2u*kGuard,kF32Guard);
    std::fill(canonical.begin()+kGuard,canonical.end()-kGuard,std::numeric_limits<float>::quiet_NaN());
    DeviceBuffer<float> dc(center),de(error),dcanonical(canonical),dwn(wn),dxn(xn);
    auto prepare=[&](){
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,dw.data(),pw.data(),rows,width);hip_ok(hipGetLastError(),"coarse_weight_encoding");
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,di.data(),pi.data(),tokens,width);hip_ok(hipGetLastError(),"coarse_input_encoding");
    };
    auto replay=[&](bool indexed,float* target,unsigned n){for(unsigned offset=0u;offset<n;offset+=262144u){const unsigned size=std::min(262144u,n-offset);const dim3 grid((size+63u)/64u);
        if(indexed)hipLaunchKernelGGL((coarse_original_replay<true>),grid,dim3(256u),0u,nullptr,pw.data(),pi.data(),dids.data(),target,rows,width,offset,size);
        else hipLaunchKernelGGL((coarse_original_replay<false>),grid,dim3(256u),0u,nullptr,pw.data(),pi.data(),nullptr,target,rows,width,offset,size);
        hip_ok(hipGetLastError(),"coarse_original_replay");}};
    // An independent complete canonical output is comparison data only. No
    // candidate kernel receives it, the GB10 reference, or selected goldens.
    prepare();replay(false,dcanonical.data(),cells);complete_strong_projection();dcanonical.read(canonical);
    for(unsigned cell=0u;cell<cells;++cell)require(std::isfinite(canonical[kGuard+cell])&&bf16(canonical[kGuard+cell])==reference[kGuard+cell],"complete canonical differs from GB10");
    for(size_t i=0u;i<kGuard;++i)require(canonical[i]==kF32Guard&&canonical[kGuard+cells+i]==kF32Guard,"complete canonical guard");
    for(unsigned sample=0u;sample<256u;++sample){const unsigned cell=unsigned((uint64_t(sample)*2654435761ull+1013904223ull)%cells);
        const float expected=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,inputs.data()+kGuard+size_t(cell/rows)*width,weights.data()+kGuard+size_t(cell%rows)*width,width);
        require(!std::memcmp(&expected,&canonical[kGuard+cell],4u),"complete canonical differs from independent CPU dot");}
    std::vector<unsigned char> initial_seen(cells,0u);
    for(unsigned cell:old_selected){require(cell<cells&&!initial_seen[cell],"initial selector duplicate or invalid cell");initial_seen[cell]=1u;}
    double samples[5][3]{},prefix_samples[5][3]{};unsigned selected_counts[5]{};
    for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<5u;++position){
        const unsigned variant=(attempt+position)%5u;
        hip_ok(hipMemset(dids.base,0xa5,ids.size()*4u),"coarse_reset_ids");hip_ok(hipMemset(dcount.data(),0,4u),"coarse_reset_counter");
        complete_strong_projection();const auto start=std::chrono::steady_clock::now();prepare();
        if(!variant){
            if(rows==2048u){std::string stage,failure;require(resident_bf16_matrix_matmul_f32_output_with_heuristic_index(dw.data(),di.data(),dc.data(),rows,width,tokens,0u,nullptr,"coarse_original_matrix",&stage,&failure),(stage+": "+failure).c_str());}
            else hip_ok(launch_selected_bf16_projection_wmma_checked(dw.data(),di.data(),dc.data(),rows,tokens,0u,0u,nullptr),"coarse_original_wmma");
            hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel,dim3(rows),dim3(256u),0u,nullptr,dw.data(),dwn.data(),rows,width);hip_ok(hipGetLastError(),"coarse_original_weight_norm");
            hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel,dim3(tokens),dim3(256u),0u,nullptr,di.data(),dxn.data(),tokens,width);hip_ok(hipGetLastError(),"coarse_original_input_norm");
            hipLaunchKernelGGL(coarse_original_compact,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,dc.data(),dxn.data(),dwn.data(),dout.data(),dids.data(),dcount.data(),cells,rows,ppb);
        }else{
            hipLaunchKernelGGL(coarse_matrix::eligibility,dim3(rows),dim3(256u),0u,nullptr,dw.data(),dwf.data(),rows,width);hip_ok(hipGetLastError(),"coarse_weight_domain");
            hipLaunchKernelGGL(coarse_matrix::eligibility,dim3(tokens),dim3(256u),0u,nullptr,di.data(),dxf.data(),tokens,width);hip_ok(hipGetLastError(),"coarse_input_domain");
#define QRT_COARSE_PRODUCE(v,c,f) if(variant==v)hipLaunchKernelGGL((coarse_matrix::produce<c,f>),dim3((rows+127u)/128u,(tokens+16u*f-1u)/(16u*f)),dim3(256u),0u,nullptr,dw.data(),di.data(),dwf.data(),dxf.data(),dc.data(),de.data(),rows,tokens,width)
            QRT_COARSE_PRODUCE(1u,64u,1u);QRT_COARSE_PRODUCE(2u,128u,1u);QRT_COARSE_PRODUCE(3u,256u,1u);QRT_COARSE_PRODUCE(4u,128u,2u);
#undef QRT_COARSE_PRODUCE
            hip_ok(hipGetLastError(),"coarse_matrix_producer");
            hipLaunchKernelGGL(coarse_matrix::compact,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,dc.data(),de.data(),dout.data(),dids.data(),dcount.data(),cells);
        }
        hip_ok(hipGetLastError(),"coarse_compaction");complete_strong_projection();unsigned selected=0u;
        hip_ok(hipMemcpy(&selected,dcount.data(),4u,hipMemcpyDeviceToHost),"coarse_candidate_count");require(selected<=cells,"coarse candidate capacity");
        const double prefix_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        replay(true,dout.data(),selected);complete_strong_projection();
        const double total_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(attempt){samples[variant][attempt-1u]=total_ms;prefix_samples[variant][attempt-1u]=prefix_ms;require(selected_counts[variant]==selected,"coarse selection changed between attempts");}else selected_counts[variant]=selected;
        auto output=initial;dout.read(output);dc.read(center);if(variant)de.read(error);dids.read(ids);dcount.read(count);
        std::vector<unsigned char> seen(cells,0u);for(unsigned i=0u;i<selected;++i){const unsigned cell=ids[kGuard+i];require(cell<cells&&!seen[cell],"coarse duplicate or invalid candidate");seen[cell]=1u;}
        // The legacy initial CPU diagnostic uses only the current-binade
        // midpoint. Production also checks adjacent midpoints. Validate every
        // live GPU decision against the actual production rule; report changes
        // from the legacy mask without making that diagnostic the authority.
        unsigned producer_changed=0u,selection_added=0u,selection_removed=0u;
        if(!variant){dwn.read(wn);dxn.read(xn);}
        for(unsigned cell=0u;cell<cells;++cell){
            if(!variant){const float value=center[kGuard+cell];const uint32_t bits=coarse_bound::scalar::bits(value),low=bits&65535u;
                const unsigned distance=low>=32768u?low-32768u:32768u-low;
                const float upper=xn[kGuard+cell/rows]*wn[kGuard+cell%rows];
                const bool expected=distance<=512u || ((bits>>23u)&255u)<32u ||
                    qrt_bf16_midpoint::within_error(value,upper*(float(ppb)*1.0e-9f));
                require(bool(seen[cell])==expected,"original live GPU selector differs from CPU production rule");
                producer_changed+=std::memcmp(&value,&initial[kGuard+cell],4u)!=0;
                selection_added+=seen[cell]&&!initial_seen[cell];selection_removed+=!seen[cell]&&initial_seen[cell];}
            require(std::isfinite(output[kGuard+cell])&&bf16(output[kGuard+cell])==reference[kGuard+cell],"coarse final GB10 endpoint");
            if(variant){const coarse_bound::State interval{center[kGuard+cell],error[kGuard+cell]};require(bool(seen[cell])==!coarse_bound::certified(interval),"coarse complete candidate mask");
                require(std::abs(double(interval.center)-double(canonical[kGuard+cell]))<=double(interval.error),"coarse canonical raw interval undercoverage");}
            if(seen[cell])require(!std::memcmp(&output[kGuard+cell],&canonical[kGuard+cell],4u),"coarse exact replay raw value");
            else require(!std::memcmp(&output[kGuard+cell],&center[kGuard+cell],4u),"coarse certified output changed");
        }
        for(size_t i=selected;i<cells;++i)require(ids[kGuard+i]==marker,"coarse unused candidate tail");
        for(size_t i=0u;i<kGuard;++i){require(ids[i]==marker&&ids[kGuard+cells+i]==marker&&count[i]==marker&&count[kGuard+1u+i]==marker,"coarse index guard");
            require(output[i]==kF32Guard&&output[kGuard+cells+i]==kF32Guard&&center[i]==kF32Guard&&center[kGuard+cells+i]==kF32Guard,"coarse output guard");
            if(variant)require(error[i]==kF32Guard&&error[kGuard+cells+i]==kF32Guard,"coarse error guard");}
        auto raw_w=weights,raw_x=inputs;dw.read(raw_w);di.read(raw_x);require(raw_w==weights&&raw_x==inputs,"coarse original input changed");pw.read(wp);pi.read(ip);
        if(variant){dwf.read(wf);dxf.read(xf);}
        for(unsigned side=0u;side<2u;++side){const auto& raw=side?inputs:weights;const auto& packed=side?ip:wp;const auto& flags=side?xf:wf;const size_t groups=side?ig:wg;const unsigned n=side?tokens:rows;
            for(size_t group=0u;group<groups;++group){const auto expected=qrt_sm121_scaled_half_products::prepare(raw.data()+kGuard+group*16u);require(!std::memcmp(&expected,&packed[kGuard+group],sizeof(expected)),"coarse complete prepared encoding");}
            for(unsigned row=0u;variant&&row<n;++row){bool eligible=true;for(unsigned k=0u;k<width;++k)eligible=eligible&&coarse_bound::eligible(raw[kGuard+size_t(row)*width+k]);require(flags[kGuard+row]==unsigned(eligible),"coarse original row domain");}
            for(size_t i=0u;i<kGuard;++i){require(!std::memcmp(&packed[i],&row_guard,sizeof(row_guard))&&!std::memcmp(&packed[kGuard+groups+i],&row_guard,sizeof(row_guard)),"coarse prepared guard");if(variant)require(flags[i]==marker&&flags[kGuard+n+i]==marker,"coarse eligibility guard");}}
        if(!variant){for(size_t i=0u;i<kGuard;++i)require(wn[i]==kF32Guard&&wn[kGuard+rows+i]==kF32Guard&&xn[i]==kF32Guard&&xn[kGuard+tokens+i]==kF32Guard,"coarse original norm guard");}
        std::cout<<"{\"type\":\"coarse_projection_attempt\",\"attempt\":"<<attempt<<",\"variant\":"<<variant<<",\"candidates\":"<<selected<<",\"all_cells_verified\":true,\"initial_legacy_selector_candidates\":"<<old_selected.size();
        if(!variant)std::cout<<",\"original_live_selector_cpu_checked\":true,\"producer_changed_cells_vs_initial\":"<<producer_changed<<",\"selection_added_vs_initial\":"<<selection_added<<",\"selection_removed_vs_initial\":"<<selection_removed;
        std::cout<<"}"<<std::endl;
    }
    for(unsigned variant=0u;variant<5u;++variant){std::array<double,3> total{samples[variant][0],samples[variant][1],samples[variant][2]},prefix{prefix_samples[variant][0],prefix_samples[variant][1],prefix_samples[variant][2]};std::sort(total.begin(),total.end());std::sort(prefix.begin(),prefix.end());
        std::cout<<"{\"type\":\"coarse_projection_full_route\",\"variant\":"<<variant<<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"width\":"<<width<<",\"cells\":"<<cells<<",\"candidates\":"<<selected_counts[variant]<<",\"chunk\":"<<(variant==1u?64u:variant==3u?256u:variant?128u:0u)<<",\"fragments\":"<<(variant==4u?2u:variant?1u:0u)<<",\"complete_route_ms\":"<<total[1]<<",\"preparation_producer_selection_ms\":"<<prefix[1]<<",\"complete_samples_ms\":["<<samples[variant][0]<<","<<samples[variant][1]<<","<<samples[variant][2]<<"],\"preparation_producer_selection_samples_ms\":["<<prefix_samples[variant][0]<<","<<prefix_samples[variant][1]<<","<<prefix_samples[variant][2]<<"],\"warmups\":1,\"measured_attempts\":3,\"maximum_candidates_per_dispatch\":262144,\"bf16_mismatches\":0,\"canonical_interval_undercoverage\":0,\"selected_raw_mismatches\":0,\"independent_cpu_dots\":256,\"all_attempts_verified\":true,\"all_prepared_words_checked\":true,\"complete_candidate_permutation_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"native_error_coefficient\":0.0000019073486328125,\"hardware_error_bound_proven\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
    }
}
}
