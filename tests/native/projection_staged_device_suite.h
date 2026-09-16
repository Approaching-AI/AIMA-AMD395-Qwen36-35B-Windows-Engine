#pragma once
namespace projection_safety_test {
void staged_device_option(const char* name,const char* value) {
#ifdef _WIN32
    require(_putenv_s(name,value)==0,"set staged device option");
#else
    require(setenv(name,value,1)==0,"set staged device option");
#endif
}
void staged_device_mode(bool enabled) {
    for(const char* name:{"QRT_QWEN36_HAWKEYE_PREPARED_OPERANDS",
        "QRT_QWEN36_HAWKEYE_PREVALIDATED_FLOAT_REPLAY","QRT_QWEN36_HAWKEYE_STAGED_HALF_REPLAY"})
        staged_device_option(name,"1");
    for(const char* name:{"QRT_QWEN36_HAWKEYE_DEVICE_REPLAY","QRT_QWEN36_HAWKEYE_QUEUED_REPLAY",
        "QRT_QWEN36_HAWKEYE_PACKED_CANDIDATES","QRT_QWEN36_HAWKEYE_FLOAT_REPLAY",
        "QRT_QWEN36_HAWKEYE_PARTITION_REPLAY","QRT_QWEN36_HAWKEYE_SCALED_FALLBACK",
        "QRT_QWEN36_HAWKEYE_ABSOLUTE_PRODUCT_BOUND","QRT_QWEN36_HAWKEYE_PREPARED_K16_MAJOR"})
        staged_device_option(name,"0");
    staged_device_option("QRT_QWEN36_HAWKEYE_STAGED_DEVICE_REPLAY",enabled?"1":"0");
}
void run_staged_device_case(unsigned rows,unsigned width,unsigned mode,unsigned window) {
    constexpr unsigned tokens=8192u;
    const size_t cells=size_t(rows)*tokens;
    std::vector<uint16_t> w(size_t(rows)*width+2u*kGuard,kBf16Guard),x(size_t(tokens)*width+2u*kGuard,kBf16Guard);
    std::fill(w.begin()+kGuard,w.end()-kGuard,uint16_t(0));
    std::fill(x.begin()+kGuard,x.end()-kGuard,uint16_t(0));
    for(unsigned row=0;row<rows;++row){
        w[kGuard+size_t(row)*width+width-1u]=bf16(float(int(row%13u)-6)/8.0f);
        // Mixed extreme exponents force the original BF16 fallback for this
        // group. Their paired inputs are zero, preserving a closed-form dot.
        if(row%17u==0u){w[kGuard+size_t(row)*width]=uint16_t(62u<<7u);w[kGuard+size_t(row)*width+1u]=uint16_t(180u<<7u);}
    }
    for(unsigned token=0;token<tokens;++token)x[kGuard+size_t(token)*width+width-1u]=bf16(float(int(token%17u)-8)/16.0f);
    std::vector<float> initial(cells+2u*kGuard,kF32Guard),control,result;
    auto selected=[&](size_t i){return mode==2u||(mode==1u&&(i%65537u==0u||i+1u==cells));};
    size_t count=0;
    for(size_t i=0;i<cells;++i){initial[kGuard+i]=selected(i)?1.00390625f:1.001f;count+=selected(i);}
    DeviceBuffer<uint16_t> dw(w),dx(x);DeviceBuffer<float> out(initial);
    double timings[2]{};
    for(unsigned variant=0;variant<2u;++variant){
        staged_device_mode(variant!=0u);
        hip_ok(hipMemcpy(out.base,initial.data(),initial.size()*sizeof(float),hipMemcpyHostToDevice),"staged device reset");
        const auto start=std::chrono::steady_clock::now();
        hip_ok(launch_selected_bf16_projection_hawkeye_midpoint_correction(dw.data(),dx.data(),nullptr,nullptr,nullptr,
            out.data(),rows,tokens,width,512u,mode==2u?tokens:0u,0u,4096u,nullptr,window),"staged device production launcher");
        timings[variant]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        result=initial;out.read(result);if(!variant)control=result;
        require(!std::memcmp(control.data(),result.data(),result.size()*sizeof(float)),"staged device differs from original production correction");
        for(size_t i=0;i<cells;++i){
            float expected=1.0f;
            if(selected(i)){expected=(float(int((i%rows)%13u)-6)/8.0f)*(float(int((i/rows)%17u)-8)/16.0f);if(expected==0.0f)expected=0.0f;}
            require(bf16(result[kGuard+i])==bf16(expected),"staged device closed-form endpoint mismatch");
        }
        for(size_t i=0;i<kGuard;++i)require(result[i]==kF32Guard&&result[kGuard+cells+i]==kF32Guard,"staged device output redzone");
        auto after_w=w,after_x=x;dw.read(after_w);dx.read(after_x);require(after_w==w&&after_x==x,"staged device source or redzone changed");
    }
    staged_device_mode(false);
    std::printf("{\"kind\":\"staged_device_production_safety\",\"rows\":%u,\"tokens\":%u,\"k\":%u,\"mode\":%u,\"window\":%u,\"cells\":%zu,\"candidates\":%zu,\"control_host_ms\":%.6f,\"device_host_ms\":%.6f,\"raw_bit_mismatches\":0,\"all_closed_form_endpoints_checked\":true,\"original_bf16_fallback_exercised\":%s,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",rows,tokens,width,mode,window,cells,count,timings[0],timings[1],mode?"true":"false");std::fflush(stdout);
}
unsigned run_staged_device_suite() {
    for(unsigned mode=0;mode<3u;++mode)run_staged_device_case(1025u,16u,mode,4194299u);
    run_staged_device_case(1024u,4096u,1u,4194304u);
    run_staged_device_case(1024u,32u,2u,16777216u);
    return 5u;
}
void run_staged_device_real(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& dx,DeviceBuffer<float>& out,
    DeviceBuffer<float>& input_bounds,DeviceBuffer<float>& weight_bounds,
    const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
    const std::vector<uint16_t>& reference,const std::vector<float>& initial,
    unsigned rows,unsigned tokens,unsigned width,unsigned ppb) {
    require(tokens==8192u,"staged device real comparison requires q8192");
    const size_t cells=size_t(rows)*tokens;
    std::vector<float> control,result;double samples[2][3]{};
    for(unsigned attempt=0;attempt<4u;++attempt)for(unsigned position=0;position<2u;++position){
        const unsigned variant=(position+attempt)%2u;staged_device_mode(variant!=0u);
        hip_ok(hipMemcpy(out.base,initial.data(),initial.size()*sizeof(float),hipMemcpyHostToDevice),"staged device real reset");
        const auto start=std::chrono::steady_clock::now();
        hip_ok(launch_selected_bf16_projection_hawkeye_midpoint_correction(dw.data(),dx.data(),nullptr,
            input_bounds.data(),weight_bounds.data(),out.data(),rows,tokens,width,512u,0u,ppb,4096u,nullptr),"staged device real production launcher");
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(attempt)samples[variant][attempt-1u]=ms;
        result=initial;out.read(result);if(!attempt&&!variant)control=result;
        require(!control.empty()&&!std::memcmp(control.data(),result.data(),result.size()*sizeof(float)),"staged device real raw endpoint mismatch");
        for(size_t i=0;i<cells;++i)require(std::isfinite(result[kGuard+i])&&bf16(result[kGuard+i])==reference[kGuard+i],"staged device real GB10 endpoint mismatch");
        for(size_t i=0;i<kGuard;++i)require(result[i]==kF32Guard&&result[kGuard+cells+i]==kF32Guard,"staged device real output redzone");
        auto after_w=weights,after_x=inputs;dw.read(after_w);dx.read(after_x);require(after_w==weights&&after_x==inputs,"staged device real inputs changed");
    }
    staged_device_mode(false);
    for(unsigned variant=0;variant<2u;++variant){
        std::array<double,3> sorted{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(sorted.begin(),sorted.end());
        std::printf("{\"type\":\"staged_device_real_production_replay\",\"device_replay\":%u,\"rows\":%u,\"tokens\":%u,\"k\":%u,\"ppb\":%u,\"elements\":%zu,\"completed_host_ms\":%.6f,\"completed_host_samples_ms\":[%.6f,%.6f,%.6f],\"raw_bit_mismatches\":0,\"gb10_bf16_mismatches\":0,\"all_attempts_verified\":true,\"rotated_variant_order\":true,\"warmup_sequences\":1,\"timed_sequences\":3,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",variant,rows,tokens,width,ppb,cells,sorted[1],samples[variant][0],samples[variant][1],samples[variant][2]);std::fflush(stdout);
    }
}
} // namespace projection_safety_test
