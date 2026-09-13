#pragma once
#include "../../native/providers/moe_accumulator/sm121_tiled_projection.h"

namespace projection_safety_test {
void run_tiled_projection_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
    DeviceBuffer<float>& dout,const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
    const std::vector<uint16_t>& reference,const std::vector<float>& original_output,
    const std::vector<unsigned>& selected,unsigned rows,unsigned tokens,unsigned width) {
    const size_t cells=size_t(rows)*tokens,mask_words=(cells+31u)/32u;
    constexpr unsigned guard=0xa5a5a5a5u;
    std::vector<unsigned> wf(rows+2u*kGuard,guard),inf(tokens+2u*kGuard,guard);
    std::vector<unsigned> index(selected.size()+2u*kGuard,guard),mask(mask_words+2u*kGuard,guard),expected_mask=mask;
    std::copy(selected.begin(),selected.end(),index.begin()+kGuard);
    std::fill(expected_mask.begin()+kGuard,expected_mask.end()-kGuard,0u);
    for(auto cell:selected) expected_mask[kGuard+cell/32u] |= 1u<<(cell&31u);
    DeviceBuffer<unsigned> dfw(wf),dfi(inf),indices(index),dmask(mask);
    std::vector<float> control;
    for(unsigned variant:{0u,64u,128u}) {
        double ms=0.0;
        for(unsigned attempt=0u;attempt<2u;++attempt) {
            hip_ok(hipMemcpy(dout.base,original_output.data(),original_output.size()*4u,hipMemcpyHostToDevice),"tiled_reset");
            const auto start=std::chrono::steady_clock::now();
            hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,dim3(rows),dim3(256u),0u,nullptr,
                dw.data(),dfw.data(),rows,width);
            hip_ok(hipGetLastError(),"tiled_weight_flags");
            hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,dim3(tokens),dim3(256u),0u,nullptr,
                di.data(),dfi.data(),tokens,width);
            hip_ok(hipGetLastError(),"tiled_input_flags");
            if(!variant) {
                if(!selected.empty()) {
                    hipLaunchKernelGGL(scalar_projection_replay_kernel<3u>,dim3((unsigned(selected.size())*4u+255u)/256u),dim3(256u),0u,nullptr,
                        dw.data(),di.data(),nullptr,nullptr,nullptr,dfw.data(),dfi.data(),indices.data(),dout.data(),rows,width,unsigned(selected.size()));
                    hip_ok(hipGetLastError(),"tiled_original_replay");
                }
            } else {
                hip_ok(qrt_sm121_tiled_projection::mark(indices.data(),unsigned(selected.size()),unsigned(cells),dmask.data(),mask_words,nullptr),"tiled_bitmap");
                hip_ok(qrt_sm121_tiled_projection::launch(dw.data(),di.data(),dfw.data(),dfi.data(),dmask.data(),mask_words,
                    dout.data(),rows,tokens,width,variant,nullptr),"tiled_replay");
            }
            complete_scalar_projection();
            if(attempt) ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        auto output=original_output;dout.read(output);if(!variant) control=output;
        size_t raw_bad=0u,bf16_bad=0u;
        for(size_t i=0u;i<cells;++i) {
            require(std::isfinite(output[kGuard+i]),"tiled real nonfinite");
            raw_bad += std::memcmp(&output[kGuard+i],&control[kGuard+i],4u)!=0;
            bf16_bad += bf16(output[kGuard+i])!=reference[kGuard+i];
        }
        if(variant) {dmask.read(mask);require(mask==expected_mask,"tiled bitmap or guard mismatch");}
        auto after_w=weights,after_i=inputs;dw.read(after_w);di.read(after_i);
        auto after_indices=index;indices.read(after_indices);
        require(after_w==weights && after_i==inputs && after_indices==index,"tiled inputs changed");
        dfw.read(wf);dfi.read(inf);
        const auto check_flags=[&](const auto& source,const auto& flags,unsigned count) {
            for(unsigned row=0u;row<count;++row) {
                bool valid=true;
                for(unsigned k=0u;k<width;++k) valid &= qrt_sm121_float_alignment::eligible(source[kGuard+size_t(row)*width+k]);
                require(flags[kGuard+row]==unsigned(valid),"tiled CPU eligibility mismatch");
            }
        };
        check_flags(weights,wf,rows);check_flags(inputs,inf,tokens);
        for(size_t i=0u;i<kGuard;++i) {
            require(output[i]==kF32Guard && output[kGuard+cells+i]==kF32Guard,"tiled output guard");
            require(wf[i]==guard && wf[kGuard+rows+i]==guard && inf[i]==guard && inf[kGuard+tokens+i]==guard,"tiled flag guard");
        }
        size_t occupied=0u,dense=0u,maximum=0u;
        if(variant) {
            const unsigned columns=(rows+variant-1u)/variant;
            std::vector<unsigned> counts(size_t(columns)*((tokens+31u)/32u),0u);
            for(auto cell:selected) ++counts[size_t((cell/rows)/32u)*columns+(cell%rows)/variant];
            for(auto count:counts) {occupied+=count!=0u;dense+=count>variant*4u;maximum=(std::max)(maximum,size_t(count));}
        }
        std::cout<<"{\"type\":\"tiled_projection_real_replay\",\"tile_rows\":"<<variant
            <<",\"tile_tokens\":32,\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width
            <<",\"elements\":"<<cells<<",\"candidates\":"<<selected.size()<<",\"raw_bit_mismatches\":"<<raw_bad
            <<",\"bf16_mismatches\":"<<bf16_bad<<",\"preparation_bitmap_replay_host_ms\":"<<ms
            <<",\"mask_bytes\":"<<(variant?mask_words*4u:0u)<<",\"occupied_tiles\":"<<occupied
            <<",\"dense_fallback_tiles\":"<<dense<<",\"maximum_tile_candidates\":"<<maximum
            <<",\"warmup_sequences\":1,\"timed_sequences\":1,\"redzones_pass\":true,\"immutable_inputs\":true,\"cpu_metadata_pass\":true,\"inference_acceptance\":false}"<<std::endl;
        require(!raw_bad && !bf16_bad,"tiled real projection differs from original or GB10");
    }
}
} // namespace projection_safety_test
