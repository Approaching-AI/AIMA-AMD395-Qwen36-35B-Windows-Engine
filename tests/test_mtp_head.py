from pathlib import Path
import subprocess
import tempfile
import unittest
from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class MtpHeadTests(unittest.TestCase):
    def build_run(self, text):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'test.cpp'
            exe = directory / 'test'
            source.write_text(text)
            subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            '-I', str(ROOT), str(source), '-o', str(exe)], check=True, timeout=60)
            subprocess.run([str(exe)], check=True, timeout=30)

    def test_finite_sampling_and_lower_id_ties(self):
        self.build_run(r'''
#include "native/providers/gdn/sm121_mtp_head_math.h"
#include <cassert>
#include <algorithm>
#include <array>
using namespace qrt_sm121_mtp;
int main() {
    HeadBest best;
    assert(head_candidate(0xbf80u,7u,&best) && best.token==7u && best.logit==-1.0f);
    assert(head_candidate(0xbf80u,3u,&best) && best.token==3u);
    assert(head_candidate(0xc000u,1u,&best) && best.token==3u);
    assert(head_candidate(0x8000u,9u,&best) && best.token==9u);
    assert(head_candidate(0x0000u,5u,&best) && best.token==5u);
    for(unsigned bits:{0x7f80u,0xff80u,0x7f81u,0xffffu})
        assert(!head_candidate(uint16_t(bits),0u,&best) && best.token==5u);
    assert(!head_candidate(0x4000u,head_vocabulary,&best));
    assert(!head_candidate(0x4000u,UINT32_MAX,&best));
    // The complete vocabulary includes the last token, and reduction order
    // cannot choose a later ID among equal BF16 maxima.
    std::array<HeadBest,256> lanes;
    for(unsigned token=0;token<head_vocabulary;++token) {
        uint16_t bits=token==head_vocabulary-1u || token==12345u?0x4180u:0xc000u;
        assert(head_candidate(bits,token,&lanes[token%256u]));
    }
    for(unsigned step=128;step;step/=2)for(unsigned lane=0;lane<step;++lane)
        if(head_better(lanes[lane+step].logit,lanes[lane+step].token,lanes[lane]))lanes[lane]=lanes[lane+step];
    assert(lanes[0].token==12345u && lanes[0].logit==16.0f);
}
''')

    def test_bounded_launches_aliases_and_partial_failure(self):
        header = (ROOT / 'native/providers/gdn/sm121_mtp_head.h').read_text()
        prelude = r'''
#include "native/providers/gdn/sm121_mtp_moe_layout.h"
#include "native/providers/gdn/sm121_mtp_head_math.h"
#include <cassert>
#include <vector>
using namespace qrt_sm121_mtp;
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
struct dim3 {unsigned x,y,z;explicit dim3(unsigned a,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
static unsigned submitted=0,failed=0,bound=0,covered=0,rows_expected=0;
static bool sampled=false;
static std::vector<unsigned> extents;
void head_projection(){} void head_argmax(){}
hipError_t hipMemsetAsync(void* p,int value,size_t bytes,hipStream_t stream){
    assert(p && !value && bytes==4u && !stream);++submitted;return submitted==failed?hipErrorUnknown:hipSuccess;
}
void record(void(*)(),dim3 grid,dim3 block,size_t shared,hipStream_t stream,
    const uint16_t*,const uint16_t*,uint16_t*,unsigned first,unsigned end){
    assert(grid.x && grid.x<=bound && grid.y==1u && block.x==256u && !shared && !stream);
    assert(first==covered && end>first && end-first<=bound*16u && grid.x==(end-first+15u)/16u);
    covered=end;++submitted;extents.push_back(end);
}
void record(void(*)(),dim3 grid,dim3 block,size_t shared,hipStream_t stream,
    uint16_t*,uint32_t*,float*,uint32_t*){
    assert(grid.x==rows_expected && block.x==256u && !shared && !stream);
    assert(covered==rows_expected*head_vocabulary);sampled=true;++submitted;
}
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
hipError_t hipGetLastError(){return submitted==failed?hipErrorUnknown:hipSuccess;}
'''
        main = r'''
int main(){
    auto* weights=reinterpret_cast<const uint16_t*>(uintptr_t(1)<<36u);
    auto* input=reinterpret_cast<const uint16_t*>(uintptr_t(1)<<37u);
    auto* logits=reinterpret_cast<uint16_t*>(uintptr_t(1)<<38u);
    auto* ids=reinterpret_cast<uint32_t*>(uintptr_t(1)<<39u);
    auto* values=reinterpret_cast<float*>(uintptr_t(1)<<40u);
    auto* invalid=reinterpret_cast<uint32_t*>(uintptr_t(1)<<41u);
    for(unsigned rows:{0u,3u,~0u})assert(launch_head(weights,input,logits,ids,values,invalid,rows)==hipErrorInvalidValue);
    for(unsigned blocks:{0u,4097u,~0u})assert(launch_head(weights,input,logits,ids,values,invalid,1u,blocks)==hipErrorInvalidValue);
    assert(launch_head(nullptr,input,logits,ids,values,invalid,1u)==hipErrorInvalidValue);
    assert(launch_head(weights,input,logits,ids,values,nullptr,1u)==hipErrorInvalidValue);
    assert(launch_head(weights,input,const_cast<uint16_t*>(input+16u),ids,values,invalid,1u)==hipErrorInvalidValue);
    assert(launch_head(weights,input,logits,reinterpret_cast<uint32_t*>(logits+16u),values,invalid,1u)==hipErrorInvalidValue);
    assert(launch_head(weights,input,logits,ids,values,ids,1u)==hipErrorInvalidValue);
    assert(launch_head(weights,input,reinterpret_cast<uint16_t*>(UINTPTR_MAX-128u),ids,values,invalid,1u)==hipErrorInvalidValue);
    assert(!submitted);
    for(unsigned rows:{1u,2u})for(unsigned blocks:{7u,1024u,4096u}){
        rows_expected=rows;bound=blocks;submitted=failed=covered=0;sampled=false;extents.clear();
        assert(launch_head(weights,input,logits,ids,values,invalid,rows,blocks)==hipSuccess);
        assert(sampled && covered==rows*head_vocabulary);unsigned complete=submitted;
        // Every call in a normal batch; small block bounds also cover the
        // first, middle, penultimate and last partial-submission boundaries.
        std::vector<unsigned> faults{1u,2u,complete/2u,complete-1u,complete};
        if(blocks>=1024u)for(unsigned i=1;i<=complete;++i)faults.push_back(i);
        for(unsigned fault:faults){
            submitted=covered=0;failed=fault;sampled=false;extents.clear();
            assert(launch_head(weights,input,logits,ids,values,invalid,rows,blocks)==hipErrorUnknown);
            assert(submitted==fault && sampled==(fault==complete));
        }
    }
}
'''
        self.build_run(prelude + function(header, 'inline hipError_t launch_head(') + main)


if __name__ == '__main__':
    unittest.main()
