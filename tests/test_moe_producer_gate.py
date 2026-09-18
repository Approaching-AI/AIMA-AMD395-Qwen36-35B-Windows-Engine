"""Exercise actual producer admission and current-operand guards."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class MoeProducerGateTests(unittest.TestCase):
    def test_admission_and_current_operand_views(self):
        provider = (ROOT / 'native/providers/triton_moe/qrt_triton_moe_q8192_provider.cpp').read_text()
        bounds = provider.split('struct MoeCorrectionBounds {', 1)[1].split('\n};', 1)[0]
        actual = function(provider, 'bool configure_moe_producer_gate()')
        actual += '\n' + function(provider, 'bool make_moe_producer_gate_bounds(')
        source = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
enum class MoeL2 { Input, RoutedGateUp };
struct State {
    bool moe_producer_gate=false,moe_producer_gate_active=false;
    bool sm121_routed_hawkeye=true,compact_routed_hawkeye=true;
    bool prevalidated_float=true,staged_half_replay=true;
    bool partition_replay=false,scaled_significand_fallback=false;
    bool parallel_routed_gate=false,moe_class_expert_order=false;
    bool staged_half_replay_active=true,prevalidated_float_active=true;
    uint32_t sm121_moe_absolute_error_ppb=512;
    std::array<float*,2> moe_l2{};
    uint16_t *prepared_replay_inputs=nullptr,*prepared_replay_weights=nullptr;
} g_state;
unsigned errors=0;
void set_error_text(const char*) { ++errors; }
''' + 'struct MoeCorrectionBounds {' + bounds + '\n};\n' + actual + r'''
int main() {
    const char* option="QRT_QWEN36_MOE_PRODUCER_GATE_REPLAY";
    const char* audit="QRT_QWEN36_MOE_CONSUMER_INTERVAL_AUDIT";
    unsetenv(option);unsetenv(audit);
    assert(configure_moe_producer_gate()&&!g_state.moe_producer_gate);
    for(const char* value:{"","0"}) {
        setenv(option,value,1);g_state=State{};g_state.staged_half_replay=false;
        assert(configure_moe_producer_gate()&&!g_state.moe_producer_gate);
    }
    for(const char* value:{"2","-1","true","01","1 "," 1"}) {
        setenv(option,value,1);g_state=State{};const auto before=errors;
        assert(!configure_moe_producer_gate()&&errors==before+1);
    }
    setenv(option,"1",1);g_state=State{};
    assert(configure_moe_producer_gate()==bool(QRT_MOE_PRODUCER_GATE_SUPPORTED));
    for(unsigned fault=0;fault<9;++fault) {
        g_state=State{};
        if(fault==0)g_state.sm121_routed_hawkeye=false;
        if(fault==1)g_state.compact_routed_hawkeye=false;
        if(fault==2)g_state.prevalidated_float=false;
        if(fault==3)g_state.staged_half_replay=false;
        if(fault==4)g_state.sm121_moe_absolute_error_ppb=0;
        if(fault==5)g_state.partition_replay=true;
        if(fault==6)g_state.scaled_significand_fallback=true;
        if(fault==7)g_state.parallel_routed_gate=true;
        if(fault==8)g_state.moe_class_expert_order=true;
        assert(!configure_moe_producer_gate());
    }
    for(const char* value:{"1","2","true"}) {
        setenv(audit,value,1);g_state=State{};assert(!configure_moe_producer_gate());
    }
    setenv(audit,"0",1);g_state=State{};
    assert(configure_moe_producer_gate()==bool(QRT_MOE_PRODUCER_GATE_SUPPORTED));
    unsetenv(option);unsetenv(audit);
    float input=3.0f,weight=4.0f;uint16_t pi=1,pw=2;
    for(unsigned fault=0;fault<9;++fault) {
        g_state=State{};g_state.moe_producer_gate_active=true;
        g_state.moe_l2={&input,&weight};g_state.prepared_replay_inputs=&pi;g_state.prepared_replay_weights=&pw;
        if(fault==0)g_state.moe_producer_gate_active=false;
        if(fault==1)g_state.staged_half_replay_active=false;
        if(fault==2)g_state.prevalidated_float_active=false;
        if(fault==3)g_state.sm121_moe_absolute_error_ppb=0;
        if(fault==4)g_state.moe_l2[0]=nullptr;
        if(fault==5)g_state.moe_l2[1]=nullptr;
        if(fault==6)g_state.prepared_replay_inputs=nullptr;
        if(fault==7)g_state.prepared_replay_weights=nullptr;
        MoeCorrectionBounds b{};
        assert(make_moe_producer_gate_bounds(&b)==(fault==8));
        if(fault==8) {
            assert(b.input_l2==&input&&b.weight_l2==&weight&&b.error_scale==512.0f*1.0e-9f);
            assert(b.prepared_input==&pi&&b.prepared_weights==&pw&&b.staged_half_replay&&b.prevalidated_float);
            assert(!b.first_block&&!b.compacted_indices&&!b.compacted_count&&!b.replay_class_range);
            assert(!b.partition_capacity&&!b.consumer_interval_audit&&!b.down_consumer_filter);
            assert(!make_moe_producer_gate_bounds(nullptr));
        }
    }
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-producer-gate-') as directory:
            for supported in (0, 1):
                exe = str(Path(directory) / f'guard-{supported}')
                subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                                '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                                f'-DQRT_MOE_PRODUCER_GATE_SUPPORTED={supported}',
                                '-x', 'c++', '-', '-o', exe], input=source, text=True,
                               check=True, timeout=30)
                subprocess.run([exe], check=True, timeout=15)
