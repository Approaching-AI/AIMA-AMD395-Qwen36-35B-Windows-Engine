from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class MtpMoeLaunchTests(unittest.TestCase):
    def test_disjoint_workspace_and_rejected_extents(self):
        with tempfile.TemporaryDirectory() as temporary:
            exe = Path(temporary) / 'mtp_moe_workspace'
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra',
                            '-Werror', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            '-I', str(ROOT), str(ROOT / 'tests/native/mtp_moe_workspace_host.cpp'),
                            '-o', str(exe)], check=True, timeout=60)
            subprocess.run([str(exe)], check=True, timeout=30)

    def test_launch_contract_and_partial_submission_stop(self):
        header = (ROOT / 'native/providers/gdn/sm121_mtp_moe.h').read_text()
        types = header[header.index('struct MoeWeights {'):header.index('namespace mtp_moe_detail {')]
        actual = types + '\nnamespace mtp_moe_detail {\n' + function(header, 'inline hipError_t dense(')
        actual += '\ntemplate<bool Down>\n' + function(header, 'inline hipError_t routed(') + '\n}\n'
        actual += function(header, 'inline hipError_t launch_moe(')
        prelude = r'''
#include "native/providers/gdn/sm121_mtp_moe_layout.h"
#include <array>
#include <cassert>
using namespace qrt_sm121_mtp;
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
using hipStream_t = void*;
struct dim3 { unsigned x,y,z; explicit dim3(unsigned a,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
unsigned submitted=0,failed_submission=0,block_bound=0;
hipError_t hipMemsetAsync(void* pointer,int value,size_t bytes,hipStream_t stream) {
    assert(pointer && !value && bytes==4 && !stream);++submitted;
    return submitted==failed_submission?hipErrorUnknown:hipSuccess;
}
void projection_kernel() {}
namespace mtp_moe_detail {
void router() {} void shared_gate() {} void activate() {} void finish() {}
template<bool Down> void routed_projection() {}
}
template<class... Args> void record(void(*kernel)(),dim3 grid,dim3 block,size_t shared,hipStream_t stream,Args...) {
    assert(grid.x && grid.y==1u && grid.z==1u && !shared && !stream);
    if(kernel==mtp_moe_detail::router || kernel==mtp_moe_detail::shared_gate) {
        assert(block.x==32u && grid.x<=2u);
    } else {
        assert(block.x==256u);
        if(kernel==projection_kernel || kernel==mtp_moe_detail::routed_projection<false> ||
           kernel==mtp_moe_detail::routed_projection<true>)assert(grid.x<=block_bound);
    }
    ++submitted;
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
hipError_t hipGetLastError(){return submitted==failed_submission?hipErrorUnknown:hipSuccess;}
'''
        main = r'''
int main() {
    alignas(256) std::array<unsigned char,160000> workspace{};
    auto* bf=reinterpret_cast<const uint16_t*>(uintptr_t(1)<<36u);
    auto* fp=reinterpret_cast<const uint32_t*>(uintptr_t(1)<<37u);
    MoeWeights weights{bf,bf,bf,bf,bf,bf};MoeTables tables{bf,bf,fp};
    const auto launch=[&](unsigned rows,unsigned blocks,size_t bytes) {
        block_bound=blocks;return launch_moe(bf,weights,tables,workspace.data(),bytes,rows,blocks);
    };
    for(unsigned rows:{0u,3u,~0u})assert(launch(rows,7,workspace.size())==hipErrorInvalidValue);
    for(unsigned blocks:{0u,4097u,~0u})assert(launch(1,blocks,workspace.size())==hipErrorInvalidValue);
    assert(launch(1,7,moe_workspace_bytes(1)-1u)==hipErrorInvalidValue);
    assert(launch_moe(nullptr,weights,tables,workspace.data(),workspace.size(),1)==hipErrorInvalidValue);
    auto altered=weights;altered.routed_down=nullptr;
    assert(launch_moe(bf,altered,tables,workspace.data(),workspace.size(),1)==hipErrorInvalidValue);
    auto broken=tables;broken.silu=nullptr;
    assert(launch_moe(bf,weights,broken,workspace.data(),workspace.size(),1)==hipErrorInvalidValue);
    assert(launch_moe(reinterpret_cast<uint16_t*>(workspace.data()+256),weights,tables,
        workspace.data(),workspace.size(),1)==hipErrorInvalidValue);
    assert(!submitted);
    for(unsigned rows:{1u,2u})for(unsigned blocks:{1u,7u,1024u}) {
        submitted=0;failed_submission=0;
        assert(launch(rows,blocks,workspace.size())==hipSuccess);
        const unsigned complete=submitted;assert(complete>=10u);
        for(unsigned fail=1;fail<=complete;++fail) {
            submitted=0;failed_submission=fail;
            assert(launch(rows,blocks,workspace.size())==hipErrorUnknown);
            assert(submitted==fail);
        }
    }
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'mtp_moe.cpp'
            exe = Path(temporary) / 'mtp_moe'
            path.write_text(prelude + actual + main)
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra',
                            '-Werror', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            '-I', str(ROOT), str(path), '-o', str(exe)], check=True, timeout=60)
            subprocess.run([str(exe)], check=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
