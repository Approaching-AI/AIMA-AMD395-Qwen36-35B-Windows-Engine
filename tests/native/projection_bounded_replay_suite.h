#pragma once
#include "../../native/providers/moe_accumulator/sm121_bounded_projection.h"

namespace projection_safety_test {
void run_bounded_projection_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
    DeviceBuffer<float>& dout,const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
    const std::vector<uint16_t>& reference,const std::vector<float>& original_output,
    const std::vector<unsigned>& selected,unsigned rows,unsigned tokens,unsigned width) {
    namespace bounded=qrt_sm121_bounded_projection;
    constexpr unsigned guard=0xa5a5a5a5u;
    const size_t cells=size_t(rows)*tokens,wgroups=size_t(rows)*(width/16u),igroups=size_t(tokens)*(width/16u);
    std::vector<unsigned> wf(rows+2u*kGuard,guard),inf(tokens+2u*kGuard,guard),index(selected.size()+2u*kGuard,guard);
    std::vector<uint32_t> wb(wgroups+2u*kGuard,guard),ib(igroups+2u*kGuard,guard),stats(selected.size()*4u+2u*kGuard,guard);
    std::copy(selected.begin(),selected.end(),index.begin()+kGuard);
    DeviceBuffer<unsigned> dfw(wf),dfi(inf),indices(index);DeviceBuffer<uint32_t> dwb(wb),dib(ib),ds(stats);
    std::vector<float> control;
    for(unsigned variant:{0u,1u}) {
        double ms=0.0;
        for(unsigned attempt=0u;attempt<2u;++attempt) {
            hip_ok(hipMemcpy(dout.base,original_output.data(),original_output.size()*4u,hipMemcpyHostToDevice),"bounded_reset");
            const auto start=std::chrono::steady_clock::now();
            if(!variant) {
                hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,dim3(rows),dim3(256u),0u,nullptr,dw.data(),dfw.data(),rows,width);
                hip_ok(hipGetLastError(),"bounded_control_weight_flags");
                hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,dim3(tokens),dim3(256u),0u,nullptr,di.data(),dfi.data(),tokens,width);
                hip_ok(hipGetLastError(),"bounded_control_input_flags");
                hipLaunchKernelGGL(scalar_projection_replay_kernel<3u>,dim3((unsigned(selected.size())*4u+255u)/256u),dim3(256u),0u,nullptr,
                    dw.data(),di.data(),nullptr,nullptr,nullptr,dfw.data(),dfi.data(),indices.data(),dout.data(),rows,width,unsigned(selected.size()));
                hip_ok(hipGetLastError(),"bounded_control_replay");
            } else {
                hip_ok(bounded::prepare(dw.data(),dwb.data(),wgroups,rows,width,nullptr),"bounded_weight_metadata");
                hip_ok(bounded::prepare(di.data(),dib.data(),igroups,tokens,width,nullptr),"bounded_input_metadata");
                hip_ok(bounded::launch(dw.data(),di.data(),dwb.data(),wgroups,dib.data(),igroups,indices.data(),unsigned(selected.size()),
                    dout.data(),rows,tokens,width,nullptr),"bounded_replay");
            }
            complete_scalar_projection();
            if(attempt) ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        auto output=original_output;dout.read(output);if(!variant) control=output;
        const auto verify=[&](const std::vector<float>& values) {
            for(size_t i=0u;i<cells;++i) {
                require(std::isfinite(values[kGuard+i]),"bounded real nonfinite");
                require(std::memcmp(&values[kGuard+i],&control[kGuard+i],4u)==0,"bounded raw output mismatch");
                require(bf16(values[kGuard+i])==reference[kGuard+i],"bounded GB10 BF16 mismatch");
            }
            for(size_t i=0u;i<kGuard;++i) require(values[i]==kF32Guard && values[kGuard+cells+i]==kF32Guard,"bounded output guard");
        };
        verify(output);uint64_t totals[4]{};
        if(!variant) {
            dfw.read(wf);dfi.read(inf);
            const auto verify_flags=[&](const auto& values,const auto& flags,unsigned count) {
                for(unsigned row=0u;row<count;++row) {
                    bool valid=true;
                    for(unsigned k=0u;k<width;++k) valid &= qrt_sm121_float_alignment::eligible(values[kGuard+size_t(row)*width+k]);
                    require(flags[kGuard+row]==unsigned(valid),"bounded control CPU flags mismatch");
                }
                for(size_t i=0u;i<kGuard;++i) require(flags[i]==guard && flags[kGuard+count+i]==guard,"bounded control flags guard");
            };
            verify_flags(weights,wf,rows);verify_flags(inputs,inf,tokens);
        }
        if(variant) {
            // Separate untimed accounting pass. Its counters and transfers are
            // excluded from the component clock; check every output again.
            hipLaunchKernelGGL(HIP_KERNEL_NAME(bounded::replay_kernel<true>),dim3((unsigned(selected.size())*4u+255u)/256u),dim3(256u),0u,nullptr,
                dw.data(),di.data(),dwb.data(),dib.data(),indices.data(),unsigned(selected.size()),dout.data(),rows,tokens,width,ds.data());
            hip_ok(hipGetLastError(),"bounded_audit");complete_scalar_projection();dout.read(output);verify(output);ds.read(stats);
            for(size_t slot=0u;slot<selected.size();++slot) {
                unsigned total=0u;
                for(unsigned i=0u;i<4u;++i) {const auto value=stats[kGuard+slot*4u+i];totals[i]+=value;total+=value;}
                require(total==width/16u,"bounded incomplete group audit");
            }
            for(size_t group=0u;group<wgroups;++group) wb[kGuard+group]=bounded::bounds::prepare(weights.data()+kGuard+group*16u);
            for(size_t group=0u;group<igroups;++group) ib[kGuard+group]=bounded::bounds::prepare(inputs.data()+kGuard+group*16u);
            auto after_wb=wb,after_ib=ib;dwb.read(after_wb);dib.read(after_ib);
            require(after_wb==wb && after_ib==ib,"bounded CPU metadata or guard mismatch");
            for(size_t i=0u;i<kGuard;++i) require(stats[i]==guard && stats[kGuard+selected.size()*4u+i]==guard,"bounded stats guard");
        }
        auto after_w=weights,after_i=inputs;dw.read(after_w);di.read(after_i);
        auto after_indices=index;indices.read(after_indices);
        require(after_w==weights && after_i==inputs && after_indices==index,"bounded immutable input changed");
        std::cout<<"{\"type\":\"bounded_projection_real_replay\",\"row_bounds\":"<<variant
            <<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width
            <<",\"elements\":"<<cells<<",\"candidates\":"<<selected.size()
            <<",\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"preparation_replay_host_ms\":"<<ms
            <<",\"metadata_bytes\":"<<(variant?(wgroups+igroups)*4u:size_t(rows+tokens)*4u)
            <<",\"carry_dominates\":"<<totals[0]<<",\"exact_higher_grid\":"<<totals[1]
            <<",\"fallback_eligible\":"<<totals[2]<<",\"fallback_ineligible\":"<<totals[3]
            <<",\"audit_outside_timing\":true,\"warmup_sequences\":1,\"timed_sequences\":1,\"redzones_pass\":true,\"immutable_inputs\":true,\"cpu_metadata_pass\":true,\"inference_acceptance\":false}"<<std::endl;
    }
}
} // namespace projection_safety_test
