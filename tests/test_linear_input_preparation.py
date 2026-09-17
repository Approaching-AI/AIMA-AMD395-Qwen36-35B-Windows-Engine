import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("linear_input_control", Path(__file__).with_name("linear_input_control.py"))
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


class LinearInputPreparation(unittest.TestCase):
    def test_control_contains_the_original_production_functions(self):
        source = (ROOT / "native/providers/whole_provider.cpp").read_text()
        control = CONTROL.extract(source)
        for name in CONTROL.FUNCTIONS:
            self.assertIn(CONTROL.function(source, name), control)
        self.assertIn("kCudaTritonSiluCorrectionTableElements = 8192u;", control)
        with self.assertRaises(ValueError):
            CONTROL.extract(source.replace("selected_conv_qkv_window_kernel", "unavailable_kernel"))

    @unittest.skipUnless(shutil.which("c++"), "requires a portable compiler")
    def test_spans_halos_aliases_and_alignment(self):
        source = r'''
#include "native/providers/gdn/linear_input_contract.h"
#include <cassert>
namespace p = qrt_linear_input_preparation;
int main() {
    p::Inputs in{reinterpret_cast<float*>(0x100000000ull),8192ull*8192,
        reinterpret_cast<uint16_t*>(0x200000000ull),8192*4,
        reinterpret_cast<unsigned char*>(0x300000000ull),qrt_sm121_silu::table_bytes,
        reinterpret_cast<unsigned char*>(0x400000000ull),qrt_sm121_rsqrt::table_bytes,8192};
    p::Outputs out{reinterpret_cast<uint16_t*>(0x500000000ull),reinterpret_cast<uint16_t*>(0x600000000ull),
        reinterpret_cast<uint16_t*>(0x700000000ull),8192*2048,8192*2048,8192*4096};
    for (unsigned first : {0u,1u,2u,3u,1023u,8191u})
        for (unsigned count : {1u,2u,3u,4u,1024u,8192u})
            assert(p::valid(in,out,first,count) == (count <= 8192-first));
    for (unsigned option=0;option<23;++option) {
        auto a=in; auto b=out;
        switch(option) {
        case 0:a.projected=nullptr;break; case 1:a.weights=nullptr;break;
        case 2:a.silu=nullptr;break;case 3:a.rsqrt=nullptr;break;
        case 4:b.q=nullptr;break;case 5:b.k=nullptr;break;case 6:b.v=nullptr;break;
        case 7:--a.projected_cells;break;case 8:--a.weight_cells;break;
        case 9:--a.silu_bytes;break;case 10:--a.rsqrt_bytes;break;
        case 11:--b.q_cells;break;case 12:--b.k_cells;break;case 13:--b.v_cells;break;
        case 14:a.tokens=0;break;case 15:a.tokens=8193;break;
        case 16:b.raw_cells=1;break;case 17:b.raw=reinterpret_cast<float*>(0x800000000ull);b.raw_cells=1;break;
        case 18:a.projected=reinterpret_cast<float*>(0x100000002ull);break;
        case 19:a.weights=reinterpret_cast<uint16_t*>(0x200000001ull);break;
        case 20:a.silu=reinterpret_cast<unsigned char*>(0x300000001ull);break;
        case 21:a.rsqrt=reinterpret_cast<unsigned char*>(0x400000001ull);break;
        case 22:b.k=reinterpret_cast<uint16_t*>(0x600000001ull);break;
        }
        assert(!p::valid(a,b,0,8192));
    }
    assert(!p::valid(in,out,0,0));assert(!p::valid(in,out,8192,1));
    assert(!p::valid(in,out,0xffffffffu,1));assert(!p::valid(in,out,1,0xffffffffu));
    const void* sources[]={in.projected,in.weights,in.silu,in.rsqrt};
    for (const void* source:sources) for(unsigned target=0;target<4;++target) {
        auto b=out;b.raw=reinterpret_cast<float*>(0x800000000ull);b.raw_cells=8192ull*8192;
        if(target==0)b.q=reinterpret_cast<uint16_t*>(const_cast<void*>(source));
        if(target==1)b.k=reinterpret_cast<uint16_t*>(const_cast<void*>(source));
        if(target==2)b.v=reinterpret_cast<uint16_t*>(const_cast<void*>(source));
        if(target==3)b.raw=reinterpret_cast<float*>(const_cast<void*>(source));
        assert(!p::valid(in,b,0,8192));
    }
    auto b=out;b.k=b.q+1;assert(!p::valid(in,b,0,8192));
    b=out;b.v=b.k+1;assert(!p::valid(in,b,0,8192));
    b=out;b.raw=reinterpret_cast<float*>(b.v);b.raw_cells=8192ull*8192;assert(!p::valid(in,b,0,8192));
    b=out;b.raw=reinterpret_cast<float*>(0x800000000ull);b.raw_cells=8192ull*8192;
    assert(p::valid(in,b,0,8192));
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "contract.cpp"
            exe = Path(directory) / "contract"
            cpp.write_text("#include <initializer_list>\n" + source)
            subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined", "-I", str(ROOT), str(cpp), "-o", str(exe)], check=True, capture_output=True, timeout=30)
            subprocess.run([str(exe)], check=True, capture_output=True, timeout=15)


if __name__ == "__main__":
    unittest.main()
