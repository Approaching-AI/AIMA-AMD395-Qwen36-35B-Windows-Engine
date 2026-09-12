// Included after the real provider and projection_safety_test helpers.
namespace projection_safety_test {
void device_replay_mode(bool enabled) {
#ifdef _WIN32
    _putenv_s("QRT_QWEN36_HAWKEYE_DEVICE_REPLAY", enabled ? "1" : "0");
#else
    setenv("QRT_QWEN36_HAWKEYE_DEVICE_REPLAY", enabled ? "1" : "0", 1);
#endif
}
qrt_sm121_prefill_projection::Plan device_replay_plan(unsigned rows, unsigned tokens, unsigned k) {
    using namespace qrt_sm121_prefill_projection;
    if ((rows == 32 || rows == 64) && k == 2048) return plan(Stage::LinearBA, tokens);
    if (rows == 2048 && k == 4096) return plan(Stage::AttentionOutput, tokens);
    return {};
}
void run_device_replay_case(unsigned rows, unsigned tokens, unsigned k, unsigned mode,
                            unsigned window = qrt_hawkeye_dispatch::maximum_window_elements) {
    const size_t cells = size_t(rows) * tokens;
    const bool random_operands = cells < 50000;
    std::vector<uint16_t> weights(size_t(rows)*k+2*kGuard,kBf16Guard), inputs(size_t(tokens)*k+2*kGuard,kBf16Guard);
    std::fill(weights.begin()+kGuard,weights.end()-kGuard,uint16_t{0});
    std::fill(inputs.begin()+kGuard,inputs.end()-kGuard,uint16_t{0});
    uint32_t seed=0x8192395;
    if (random_operands) {
        for (auto* values : {&weights,&inputs}) for(size_t i=kGuard;i+kGuard<values->size();++i) {
            seed=seed*1664525u+1013904223u;
            (*values)[i]=uint16_t((seed&0x807fu)|((116u+((seed>>8)%20u))<<7));
            if(i%37==0)(*values)[i]&=0x807f; // Subnormal, zero and cancellation signs.
        }
    } else {
        for(unsigned r=0;r<rows;++r) weights[kGuard+size_t(r)*k+k-1]=bf16(float(int(r%13)-6)/8);
        for(unsigned t=0;t<tokens;++t) inputs[kGuard+size_t(t)*k+k-1]=bf16(float(int(t%17)-8)/16);
    }
    const auto saved_w=weights,saved_x=inputs;
    std::vector<float> original(cells+2*kGuard,kF32Guard), baseline, result;
    std::vector<float> input_norm(tokens+2*kGuard,2.0f),weight_norm(rows+2*kGuard,1000.0f);
    for(size_t i=0;i<cells;++i)original[kGuard+i]=mode==2||(mode==1&&(i%37==0||i+1==cells))?1.00390625f:mode==3?1.0f:1.001f;
    baseline=original;result=original;
    DeviceBuffer<uint16_t> dw(weights),dx(inputs);
    DeviceBuffer<float> old_out(baseline),new_out(result),xn(input_norm),wn(weight_norm);
    double times[2]{};
    for(unsigned device=0;device<2;++device) {
        device_replay_mode(device!=0);
        const auto begin=std::chrono::steady_clock::now();
        hip_ok(launch_selected_bf16_projection_hawkeye_midpoint_correction(dw.data(),dx.data(),nullptr,
            mode==3?xn.data():nullptr,mode==3?wn.data():nullptr,device?new_out.data():old_out.data(),
            rows,tokens,k,512,mode==2?tokens:0,mode==3?1000:0,4096,nullptr,window),"device_replay_launcher");
        times[device]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    }
    old_out.read(baseline);new_out.read(result);
    size_t bad=0;
    for(size_t i=0;i<result.size();++i) {
        bad+=std::memcmp(&result[i],&baseline[i],sizeof(float))!=0;
        if(i<kGuard||i>=kGuard+cells)require(result[i]==kF32Guard&&baseline[i]==kF32Guard,"device replay output redzone");
    }
    require(bad==0,"device replay differs from original ordered correction");
    const size_t samples=random_operands?(std::min)(cells,size_t(129)):cells;
    for(size_t sample=0;sample<samples;++sample) {
        const size_t i=random_operands?sample*(cells-1)/(samples>1?samples-1:1):sample;
        const unsigned token=unsigned(i/rows),row=unsigned(i%rows);
        unsigned local_token=token,plan_tokens=tokens;
        if(tokens>8192&&tokens%8192&&qrt_sm121_prefill_projection::changes_dot(device_replay_plan(rows,tokens%8192,k))) {
            const unsigned prefix=tokens-tokens%8192;
            if(token>=prefix){local_token=token-prefix;plan_tokens=tokens%8192;}else plan_tokens=prefix;
        }
        const auto plan=device_replay_plan(rows,plan_tokens,k);
        const bool selected=mode==2||mode==3||(mode==1&&(i%37==0||i+1==cells))||qrt_sm121_prefill_projection::changes_dot(plan);
        float expected=1.0f;
        if(selected) {
            if(random_operands)expected=qrt_sm121_prefill_projection::dot<1>(inputs.data()+kGuard+size_t(token)*k,
                weights.data()+kGuard+size_t(row)*k,k,plan,local_token,row);
            else {expected=(float(int(row%13)-6)/8)*(float(int(token%17)-8)/16);if(expected==0)expected=0;}
        }
        const uint16_t endpoint=bf16(expected);uint32_t bits=uint32_t(endpoint)<<16;float rounded;
        std::memcpy(&rounded,&bits,4);
        require(std::memcmp(&result[kGuard+i],&rounded,4)==0,"independent CPU dot or closed-form endpoint mismatch");
    }
    dw.read(weights);dx.read(inputs);require(weights==saved_w&&inputs==saved_x,"device replay operand/redzone modified");
    std::vector<float> norms_after=input_norm;xn.read(norms_after);require(norms_after==input_norm,"input bounds changed");
    norms_after=weight_norm;wn.read(norms_after);require(norms_after==weight_norm,"weight bounds changed");
    std::printf("{\"kind\":\"hawkeye_device_replay\",\"rows\":%u,\"tokens\":%u,\"k\":%u,\"mode\":%u,\"window\":%u,\"cells\":%zu,\"cpu_reference_cells\":%zu,\"random_operands\":%s,\"bit_mismatches\":%zu,\"control_host_ms\":%.6f,\"device_host_ms\":%.6f,\"inference_acceptance\":false}\n",
        rows,tokens,k,mode,window,cells,samples,random_operands?"true":"false",bad,times[0],times[1]);std::fflush(stdout);
    device_replay_mode(false);
}

void run_device_collection_case(unsigned rows,unsigned tokens,size_t offset,unsigned window,unsigned mode) {
    const size_t cells=size_t(rows)*tokens;
    std::vector<float> original(cells+2*kGuard,kF32Guard),old_values,new_values;
    for(size_t i=0;i<cells;++i)original[kGuard+i]=(mode==2||(mode==1&&i%7==0))?1.00390625f:1.001f;
    old_values=original;new_values=original;
    DeviceBuffer<float> old_out(old_values),new_out(new_values);
    constexpr unsigned sentinel=0xa5a5a5a5;
    std::vector<unsigned> control(window+2+2*kGuard,sentinel),fused=control;
    DeviceBuffer<unsigned> old_scratch(control),new_scratch(fused);
    for(unsigned device=0;device<2;++device) {
        unsigned* scratch=device?new_scratch.data():old_scratch.data();
        hip_ok(hipMemsetAsync(scratch,0,8,nullptr),"zero count");
        if(device)hipLaunchKernelGGL((selected_bf16_projection_hawkeye_compact_kernel<true>),dim3((window+255)/256),dim3(256),0,nullptr,
            nullptr,nullptr,nullptr,new_out.data(),rows,512,0,0,scratch,scratch+2,offset,window);
        else hipLaunchKernelGGL((selected_bf16_projection_hawkeye_compact_kernel<false>),dim3((window+255)/256),dim3(256),0,nullptr,
            nullptr,nullptr,nullptr,old_out.data(),rows,512,0,0,scratch,scratch+2,offset,window);
        hip_ok(hipGetLastError(),"collector launch");hip_ok(hipDeviceSynchronize(),"collector completion");
    }
    old_scratch.read(control);new_scratch.read(fused);old_out.read(old_values);new_out.read(new_values);
    const unsigned count=control[kGuard];
    require(count==fused[kGuard]&&count<=window&&control[kGuard+1]==fused[kGuard+1],"fused collector counts differ");
    std::vector<unsigned> expected;
    for(unsigned i=0;i<window;++i)if(mode==2||(mode==1&&(offset+i)%7==0))expected.push_back(unsigned(offset+i));
    std::vector<unsigned> a(control.begin()+kGuard+2,control.begin()+kGuard+2+count),b(fused.begin()+kGuard+2,fused.begin()+kGuard+2+count);
    std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());require(a==b&&a==expected,"candidate membership, uniqueness or absolute index");
    for(size_t i=0;i<control.size();++i)if(i<kGuard||i>=kGuard+2+count)
        require(control[i]==sentinel&&fused[i]==sentinel,"counter/index arena redzone or unused slot changed");
    require(old_values==original,"ordinary collector mutated outputs");
    for(size_t i=0;i<new_values.size();++i){float value=original[i];
        if(i>=kGuard+offset&&i<kGuard+offset+window){uint32_t bits=uint32_t(bf16(value))<<16;std::memcpy(&value,&bits,4);}
        require(std::memcmp(&new_values[i],&value,4)==0,"fused rounding crossed window or changed endpoint");}
    std::printf("{\"kind\":\"hawkeye_device_collection\",\"window\":%u,\"offset\":%zu,\"mode\":%u,\"candidates\":%u,\"bit_mismatches\":0,\"inference_acceptance\":false}\n",window,offset,mode,count);std::fflush(stdout);
}

unsigned run_device_replay_suite() {
    for(unsigned mode=0;mode<4;++mode)run_device_replay_case(129,67,2048,mode,257);
    run_device_replay_case(2048,2049,16,2); // Full 4M window and partial next window.
    run_device_replay_case(2048,2048,4096,2); // Maximum K and dense bounded device window.
    for(unsigned tokens : {16u,19u,23u,1024u,8211u})run_device_replay_case(32,tokens,2048,2);
    for(unsigned tokens : {19u,55u})run_device_replay_case(2048,tokens,4096,2);
    for(unsigned mode=0;mode<3;++mode){run_device_collection_case(129,67,13,257,mode);run_device_collection_case(2048,2049,0,4194304,mode);}
    return 19;
}
} // namespace projection_safety_test
