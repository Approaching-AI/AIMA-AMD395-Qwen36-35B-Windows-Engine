#pragma once
#include "../../native/providers/moe_accumulator/sm121_row_max_projection.h"

namespace projection_safety_test {
void run_row_max_projection_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
    DeviceBuffer<float>& dout,const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
    const std::vector<uint16_t>& reference,const std::vector<float>& original_output,
    const std::vector<unsigned>& selected,unsigned rows,unsigned tokens,unsigned width) {
    namespace row_gpu=qrt_sm121_row_max_projection;
    constexpr unsigned guard=0xa5a5a5a5u;
    const size_t cells=size_t(rows)*tokens,wgroups=rows,igroups=tokens;
    std::vector<unsigned> wf(rows+2u*kGuard,guard),inf(tokens+2u*kGuard,guard),index(selected.size()+2u*kGuard,guard);
    std::vector<uint32_t> wb(wgroups+2u*kGuard,guard),ib(igroups+2u*kGuard,guard),stats(selected.size()*4u+2u*kGuard,guard);
    std::copy(selected.begin(),selected.end(),index.begin()+kGuard);
    DeviceBuffer<unsigned> dfw(wf),dfi(inf),indices(index);DeviceBuffer<uint32_t> dwb(wb),dib(ib),ds(stats);
    std::vector<float> control;
    for(unsigned variant:{0u,1u}) {
        double ms=0.0;
        for(unsigned attempt=0u;attempt<2u;++attempt) {
            hip_ok(hipMemcpy(dout.base,original_output.data(),original_output.size()*4u,hipMemcpyHostToDevice),"row maximum_reset");
            const auto start=std::chrono::steady_clock::now();
            if(!variant) {
                hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,dim3(rows),dim3(256u),0u,nullptr,dw.data(),dfw.data(),rows,width);
                hip_ok(hipGetLastError(),"row maximum_control_weight_flags");
                hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,dim3(tokens),dim3(256u),0u,nullptr,di.data(),dfi.data(),tokens,width);
                hip_ok(hipGetLastError(),"row maximum_control_input_flags");
                hipLaunchKernelGGL(scalar_projection_replay_kernel<3u>,dim3((unsigned(selected.size())*4u+255u)/256u),dim3(256u),0u,nullptr,
                    dw.data(),di.data(),nullptr,nullptr,nullptr,dfw.data(),dfi.data(),indices.data(),dout.data(),rows,width,unsigned(selected.size()));
                hip_ok(hipGetLastError(),"row maximum_control_replay");
            } else {
                hip_ok(row_gpu::prepare(dw.data(),dwb.data(),wgroups,rows,width,nullptr),"row maximum_weight_metadata");
                hip_ok(row_gpu::prepare(di.data(),dib.data(),igroups,tokens,width,nullptr),"row maximum_input_metadata");
                hip_ok(row_gpu::launch(dw.data(),di.data(),dwb.data(),wgroups,dib.data(),igroups,indices.data(),unsigned(selected.size()),
                    dout.data(),rows,tokens,width,nullptr),"row maximum_replay");
            }
            complete_scalar_projection();
            if(attempt) ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        auto output=original_output;dout.read(output);if(!variant) control=output;
        const auto verify=[&](const std::vector<float>& values) {
            for(size_t i=0u;i<cells;++i) {
                require(std::isfinite(values[kGuard+i]),"row maximum real nonfinite");
                require(std::memcmp(&values[kGuard+i],&control[kGuard+i],4u)==0,"row maximum raw output mismatch");
                require(bf16(values[kGuard+i])==reference[kGuard+i],"row maximum GB10 BF16 mismatch");
            }
            for(size_t i=0u;i<kGuard;++i) require(values[i]==kF32Guard && values[kGuard+cells+i]==kF32Guard,"row maximum output guard");
        };
        verify(output);uint64_t totals[4]{};
        if(!variant) {
            dfw.read(wf);dfi.read(inf);
            const auto verify_flags=[&](const auto& values,const auto& flags,unsigned count) {
                for(unsigned row=0u;row<count;++row) {
                    bool valid=true;
                    for(unsigned k=0u;k<width;++k) valid &= qrt_sm121_float_alignment::eligible(values[kGuard+size_t(row)*width+k]);
                    require(flags[kGuard+row]==unsigned(valid),"row maximum control CPU flags mismatch");
                }
                for(size_t i=0u;i<kGuard;++i) require(flags[i]==guard && flags[kGuard+count+i]==guard,"row maximum control flags guard");
            };
            verify_flags(weights,wf,rows);verify_flags(inputs,inf,tokens);
        }
        if(variant) {
            // Separate untimed accounting pass. Its counters and transfers are
            // excluded from the component clock; check every output again.
            hipLaunchKernelGGL(HIP_KERNEL_NAME(row_gpu::replay_kernel<true>),dim3((unsigned(selected.size())*4u+255u)/256u),dim3(256u),0u,nullptr,
                dw.data(),di.data(),dwb.data(),dib.data(),indices.data(),unsigned(selected.size()),dout.data(),rows,tokens,width,ds.data());
            hip_ok(hipGetLastError(),"row maximum_audit");complete_scalar_projection();dout.read(output);verify(output);ds.read(stats);
            for(size_t slot=0u;slot<selected.size();++slot) {
                unsigned total=0u;
                for(unsigned i=0u;i<4u;++i) {const auto value=stats[kGuard+slot*4u+i];totals[i]+=value;total+=value;}
                require(total==width/16u,"row maximum incomplete group audit");
            }
            for(unsigned side=0u;side<2u;++side) {
                auto& flags=side?wb:ib;const auto& values=side?weights:inputs;const unsigned count=side?rows:tokens;
                for(unsigned row=0u;row<count;++row) {
                    flags[kGuard+row]=0u;
                    for(unsigned k=0u;k<width;++k) {
                        const uint16_t v=values[kGuard+size_t(row)*width+k];const unsigned e=(v>>7u)&255u;
                        const unsigned summary=!(v&0x7fffu)?0u:e<64u||e>190u?256u:e;
                        flags[kGuard+row]=std::max(flags[kGuard+row],summary);
                    }
                }
            }
            auto after_wb=wb,after_ib=ib;dwb.read(after_wb);dib.read(after_ib);
            require(after_wb==wb && after_ib==ib,"row maximum CPU metadata or guard mismatch");
            for(size_t i=0u;i<kGuard;++i) require(stats[i]==guard && stats[kGuard+selected.size()*4u+i]==guard,"row maximum stats guard");
        }
        auto after_w=weights,after_i=inputs;dw.read(after_w);di.read(after_i);
        auto after_indices=index;indices.read(after_indices);
        require(after_w==weights && after_i==inputs && after_indices==index,"row maximum immutable input changed");
        std::cout<<"{\"type\":\"row_max_projection_real_replay\",\"row_maximum\":"<<variant
            <<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width
            <<",\"elements\":"<<cells<<",\"candidates\":"<<selected.size()
            <<",\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"preparation_replay_host_ms\":"<<ms
            <<",\"metadata_bytes\":"<<(variant?(wgroups+igroups)*4u:size_t(rows+tokens)*4u)
            <<",\"carry_dominant_groups\":"<<totals[0]<<",\"maximum_reduction_groups\":"<<totals[1]
            <<",\"ineligible_fallback_groups\":"<<totals[2]<<",\"unused_counter3\":"<<totals[3]
            <<",\"audit_outside_timing\":true,\"warmup_sequences\":1,\"timed_sequences\":1,\"redzones_pass\":true,\"immutable_inputs\":true,\"cpu_metadata_pass\":true,\"inference_acceptance\":false}"<<std::endl;
    }
}
} // namespace projection_safety_test
