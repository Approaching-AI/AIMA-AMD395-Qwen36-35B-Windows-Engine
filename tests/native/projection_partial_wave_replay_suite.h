#pragma once
#include "projection_matrix_replay_suite.h"
#include "../../native/providers/moe_accumulator/sm121_partial_wave_projection.h"

namespace projection_safety_test {
namespace partial_wave_projection=qrt_sm121_partial_wave_projection;
void run_partial_wave_projection_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
    DeviceBuffer<float>& dout,const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
    const std::vector<uint16_t>& reference,const std::vector<float>& initial,
    const std::vector<unsigned>& selected,unsigned rows,unsigned tokens,unsigned width) {
    using PartialRow=partial_wave_projection::Row;
    const size_t cells=size_t(rows)*tokens,wg=size_t(rows)*(width/16u),ig=size_t(tokens)*(width/16u);
    const size_t mask_words=(cells+31u)/32u;
    constexpr unsigned marker=0xa5a5a5a5u;
    MatrixControlRow control_guard;std::memset(&control_guard,0xa5,sizeof(control_guard));
    PartialRow partial_guard;std::memset(&partial_guard,0xa5,sizeof(partial_guard));
    std::vector<MatrixControlRow> control_w(wg+2u*kGuard,control_guard),control_x(ig+2u*kGuard,control_guard);
    std::vector<PartialRow> partial_w(wg+2u*kGuard,partial_guard),partial_x(ig+2u*kGuard,partial_guard);
    DeviceBuffer<MatrixControlRow> dcw(control_w),dcx(control_x);
    DeviceBuffer<PartialRow> dpw(partial_w),dpx(partial_x);
    std::vector<unsigned> wh(rows+2u*kGuard,marker),xh(tokens+2u*kGuard,marker);
    DeviceBuffer<unsigned> fw(wh),fx(xh);
    std::vector<unsigned> indices(selected.size()+2u*kGuard,marker),bits(mask_words+2u*kGuard,marker);
    std::copy(selected.begin(),selected.end(),indices.begin()+kGuard);
    std::fill(bits.begin()+kGuard,bits.end()-kGuard,0u);
    for(unsigned cell:selected)bits[kGuard+cell/32u]|=1u<<(cell&31u);
    DeviceBuffer<unsigned> ids(indices),bitmap(bits);
    std::vector<float> control;double samples[2][3]{};
    size_t fast_candidates=0u,fast_rows[2]{};
    for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<2u;++position){
        const unsigned variant=(position+attempt)%2u;
        hip_ok(hipMemcpy(dout.base,initial.data(),initial.size()*4u,hipMemcpyHostToDevice),"partial_wave_reset_output");
        hip_ok(hipMemset(dcw.base,0xa5,control_w.size()*sizeof(MatrixControlRow)),"partial_wave_reset_control_w");
        hip_ok(hipMemset(dcx.base,0xa5,control_x.size()*sizeof(MatrixControlRow)),"partial_wave_reset_control_x");
        hip_ok(hipMemset(dpw.base,0xa5,partial_w.size()*sizeof(PartialRow)),"partial_wave_reset_w");
        hip_ok(hipMemset(dpx.base,0xa5,partial_x.size()*sizeof(PartialRow)),"partial_wave_reset_x");
        hip_ok(hipMemset(fw.base,0xa5,wh.size()*4u),"partial_wave_reset_w_flags");
        hip_ok(hipMemset(fx.base,0xa5,xh.size()*4u),"partial_wave_reset_x_flags");
        complete_strong_projection();const auto start=std::chrono::steady_clock::now();
        if(!variant){
            hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,dw.data(),dcw.data(),rows,width);
            hip_ok(hipGetLastError(),"partial_wave_control_w");
            hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,di.data(),dcx.data(),tokens,width);
            hip_ok(hipGetLastError(),"partial_wave_control_x");
            if(!selected.empty())hipLaunchKernelGGL(matrix_projection_replay_kernel,dim3((unsigned(selected.size())*4u+255u)/256u),dim3(256u),0u,nullptr,dcw.data(),dcx.data(),ids.data(),dout.data(),rows,width,unsigned(selected.size()));
            hip_ok(hipGetLastError(),"partial_wave_original_replay");
        }else{
            hip_ok(partial_wave_projection::encode(dw.data(),size_t(rows)*width,dpw.data(),wg,fw.data(),rows,rows,width,nullptr),"partial_wave_encode_w");
            hip_ok(partial_wave_projection::encode(di.data(),size_t(tokens)*width,dpx.data(),ig,fx.data(),tokens,tokens,width,nullptr),"partial_wave_encode_x");
            hip_ok(qrt_sm121_tiled_projection::mark(ids.data(),unsigned(selected.size()),unsigned(cells),bitmap.data(),mask_words,nullptr),"partial_wave_bitmap");
            hip_ok(partial_wave_projection::launch(dw.data(),di.data(),dpw.data(),dpx.data(),fw.data(),fx.data(),bitmap.data(),mask_words,ids.data(),unsigned(selected.size()),dout.data(),cells,rows,tokens,width,nullptr),"partial_wave_replay");
        }
        complete_strong_projection();
        if(attempt)samples[variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        auto output=initial;dout.read(output);if(!attempt&&!variant)control=output;
        require(!control.empty(),"partial wave missing original control");
        for(size_t cell=0u;cell<cells;++cell){
            require(std::isfinite(output[kGuard+cell]),"partial wave nonfinite");
            require(!std::memcmp(&output[kGuard+cell],&control[kGuard+cell],4u),"partial wave raw candidate or inactive output differs");
            require(bf16(output[kGuard+cell])==reference[kGuard+cell],"partial wave GB10 endpoint differs");
        }
        auto raw_w=weights,raw_x=inputs;dw.read(raw_w);di.read(raw_x);
        require(raw_w==weights&&raw_x==inputs,"partial wave raw operands modified");
        auto after_indices=indices,after_bitmap=bits;ids.read(after_indices);bitmap.read(after_bitmap);
        require(after_indices==indices&&after_bitmap==bits,"partial wave candidate identities or bitmap modified");
        dcw.read(control_w);dcx.read(control_x);dpw.read(partial_w);dpx.read(partial_x);fw.read(wh);fx.read(xh);
        for(unsigned side=0u;side<2u;++side){
            const auto& raw=side?inputs:weights;const auto& old=side?control_x:control_w;
            const auto& packed=side?partial_x:partial_w;const auto& flags=side?xh:wh;
            const unsigned count=side?tokens:rows;const size_t total=size_t(count)*(width/16u);
            fast_rows[side]=0u;
            for(unsigned row=0u;row<count;++row){
                bool good=true;
                for(unsigned k=0u;k<width;++k){const uint16_t value=raw[kGuard+size_t(row)*width+k];
                    const unsigned exponent=(value>>7u)&255u;good&=!(value&0x7fffu)||(exponent>=95u&&exponent<=159u);}
                fast_rows[side]+=good;
                require(flags[kGuard+row]==(variant?unsigned(good):marker),"partial wave whole-row flag differs");
                for(unsigned g=0u;g<width/16u;++g){
                    const auto* source=raw.data()+kGuard+size_t(row)*width+g*16u;
                    const auto& actual=packed[kGuard+size_t(g)*count+row];
                    const auto& old_actual=old[kGuard+size_t(row)*(width/16u)+g];
                    if(variant){const auto expected=partial_wave_projection::partial::prepare(source);
                        require(!std::memcmp(&actual,&expected,sizeof(expected)),"partial wave encoded operand differs");
                        require(!std::memcmp(&old_actual,&control_guard,sizeof(control_guard)),"partial wave touched control storage");
                        for(unsigned i=0u;i<16u;++i)require(partial_wave_projection::partial::original(actual,i)==source[i],"partial wave lossless original differs");
                    }else{const auto expected=qrt_sm121_scaled_half_products::prepare(source);
                        require(!std::memcmp(&old_actual,&expected,sizeof(expected)),"partial wave control encoded operand differs");
                        require(!std::memcmp(&actual,&partial_guard,sizeof(partial_guard)),"control touched partial storage");}
                }
            }
            for(size_t i=0u;i<kGuard;++i){
                require(flags[i]==marker&&flags[kGuard+count+i]==marker,"partial wave flag guard");
                require(!std::memcmp(&packed[i],&partial_guard,sizeof(partial_guard))&&!std::memcmp(&packed[kGuard+total+i],&partial_guard,sizeof(partial_guard)),"partial wave operand guard");
                require(!std::memcmp(&old[i],&control_guard,sizeof(control_guard))&&!std::memcmp(&old[kGuard+total+i],&control_guard,sizeof(control_guard)),"partial wave control operand guard");
            }
        }
        if(variant){fast_candidates=0u;for(unsigned cell:selected)fast_candidates+=wh[kGuard+cell%rows]&&xh[kGuard+cell/rows];}
        for(size_t i=0u;i<kGuard;++i)require(output[i]==kF32Guard&&output[kGuard+cells+i]==kF32Guard,"partial wave output guard");
    }
    for(unsigned variant=0u;variant<2u;++variant){
        std::array<double,3> ordered{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(ordered.begin(),ordered.end());
        std::cout<<"{\"type\":\"partial_wave_projection_real_replay\",\"variant\":"<<variant
            <<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width<<",\"elements\":"<<cells
            <<",\"candidates\":"<<selected.size()<<",\"admitted_candidates\":"<<(variant?fast_candidates:0u)
            <<",\"original_candidates\":"<<(variant?selected.size()-fast_candidates:selected.size())
            <<",\"admitted_weight_rows\":"<<fast_rows[0]<<",\"admitted_input_rows\":"<<fast_rows[1]
            <<",\"packed_operand_bytes\":"<<(wg+ig)*(variant?sizeof(PartialRow):sizeof(MatrixControlRow))
            <<",\"extra_flag_and_bitmap_bytes\":"<<(variant?(rows+tokens+mask_words)*4u:0u)
            <<",\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"unrounded_candidate_bit_mismatches\":0"
            <<",\"preparation_classification_and_replay_host_ms\":"<<ordered[1]
            <<",\"completed_host_samples_ms\":["<<samples[variant][0]<<","<<samples[variant][1]<<","<<samples[variant][2]<<"]"
            <<",\"warmup_sequences\":1,\"timed_sequences\":3,\"rotated_variant_order\":true,\"all_attempts_verified\":true"
            <<",\"whole_row_flags_checked\":true,\"all_encoded_words_checked\":true,\"candidate_bitmap_checked\":true"
            <<",\"redzones_pass\":true,\"immutable_after_each_sequence\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
    }
}
} // namespace projection_safety_test
