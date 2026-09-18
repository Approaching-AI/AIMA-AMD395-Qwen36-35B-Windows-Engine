"""Exercise original seeded boundaries without HIP or model substitution."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('c++'), 'requires a portable compiler')
class PrefixLinearCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        source = cls.root / 'probe.cpp'
        source.write_text(r'''
#include "native/providers/prefix_linear_capture.h"
#include <cassert>
#include <iostream>
using namespace qrt_prefix_linear_capture;
int main(int argc, char **argv) {
    assert(argc == 3);
    const std::string mode(argv[2]); std::string error; Plan plan;
    if (mode == "parse") {
        assert(parse(nullptr,nullptr,nullptr,nullptr,plan,error) && !plan.directory);
        for (auto prefix : {"8192","90112","253952"})
            assert(parse(argv[1],"16",prefix,"8192",plan,error));
        assert(parse(argv[1],"38","262144","1024",plan,error));
        for (auto layer : {"3","39","-1","+16","16x"," 16","999999999999",""})
            assert(!parse(argv[1],layer,"90112","8192",plan,error));
        for (auto prefix : {"0","8191","90113","262144","270336","999999999999"})
            assert(!parse(argv[1],"16",prefix,"8192",plan,error));
        for (auto tokens : {"0","1023","1025","8193","-1024","999999999999"})
            assert(!parse(argv[1],"16","90112",tokens,plan,error));
        assert(!parse(argv[1],nullptr,"90112","8192",plan,error));
        assert(!parse(argv[1],"16",nullptr,"8192",plan,error));
        assert(!parse(argv[1],"16","90112",nullptr,plan,error));
        std::cout << "{}\n"; return 0;
    }
    assert(parse(argv[1],"16","90112","1024",plan,error));
    constexpr unsigned tokens=1024;
    std::vector<float> raw(size_t(tokens)*8192u,1), gates(size_t(tokens)*64u,2),
                       output(size_t(tokens)*4096u,-1), state(524288,3);
    const auto original_raw=raw, original_gates=gates;
    unsigned copies=0, executions=0;
    auto copy = [&](void *dst,const void *src,size_t bytes) {
        assert(bytes && bytes <= copy_chunk_bytes); ++copies;
        if (mode == "copy_failure" && copies == 2) return false;
        if (mode == "post_copy_failure" && executions) return false;
        std::memcpy(dst,src,bytes); return true;
    };
    auto execute = [&] {
        ++executions;
        if (mode == "execute_failure") return false;
        std::fill(output.begin(),output.end(),4); state.front()=5; state.back()=6;
        return true;
    };
    unsigned layer=16,prefix=90112,count=tokens;
    if (mode == "other_layer") layer=17;
    if (mode == "other_prefix") prefix=81920;
    if (mode == "other_tokens") count=8192;
    if (mode == "disabled") plan.directory=nullptr;
    if (mode == "existing") { std::filesystem::create_directory(argv[1]);
        std::ofstream(std::filesystem::path(argv[1])/"sentinel") << "preserve"; }
    const bool selected=plan.matches(layer,prefix,count);
    const bool ok=run(plan,layer,prefix,count,raw.data(),gates.data(),output.data(),state.data(),
                      mode != "value_major",copy,execute,error);
    assert(raw == original_raw && gates == original_gates && executions <= 1);
    if (!selected) {
        assert(ok && executions == 1 && !copies && !std::filesystem::exists(argv[1]));
        assert(row(plan,layer,prefix,count,nullptr,nullptr,0,0,copy,error));
    } else if (ok) {
        auto root=std::filesystem::path(argv[1]);
        const unsigned saved=copies;
        assert(!run(plan,layer,prefix,count,raw.data(),gates.data(),output.data(),state.data(),
                    true,copy,execute,error));
        assert(executions == 1 && copies == saved);
        std::vector<uint16_t> bf16(size_t(tokens)*8u,0);
        for(unsigned i=0;i<8;++i) bf16[size_t(tokens-1)*8u+i]=uint16_t(0x3f80+i);
        output[size_t(tokens-1)*4096u]=7;
        assert(row(plan,layer,prefix,count,"qkv_projection",bf16.data(),8,2,copy,error));
        assert(row(plan,layer,prefix,count,"core",output.data(),4096,4,copy,error));
        const unsigned after_rows=copies;
        assert(row(plan,layer,prefix,count,"postconv",nullptr,8192,4,copy,error));
        assert(!row(plan,layer,prefix,count,"core",output.data(),4096,4,copy,error));
        for(const char *bad : {"", "../bad", "Bad", "stage1"})
            assert(!row(plan,layer,prefix,count,bad,output.data(),4096,4,copy,error));
        assert(!row(plan,layer,prefix,count,"bad_width",output.data(),8193,4,copy,error));
        assert(!row(plan,layer,prefix,count,"bad_type",output.data(),4096,3,copy,error));
        assert(copies == after_rows);
    } else {
        assert(!std::filesystem::exists(std::filesystem::path(argv[1])/"capture.json"));
        const unsigned saved=copies;
        assert(!row(plan,layer,prefix,count,"core",output.data(),4096,4,copy,error));
        assert(copies == saved);
    }
    std::cout << "{\"success\":" << ok << ",\"copies\":" << copies
              << ",\"executions\":" << executions << "}\n";
}
''')
        cls.binary = cls.root / 'probe'
        subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=address,undefined', '-I'+str(ROOT), str(source),
                        '-o', str(cls.binary)], check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def case(self, name):
        path = self.root / name
        result = subprocess.run([str(self.binary), str(path), name],
                                check=True, capture_output=True, text=True, timeout=20)
        return path, json.loads(result.stdout)

    def test_bounded_original_transaction_selection(self):
        self.case('parse')

    def test_disabled_and_other_transactions_execute_once_without_reads(self):
        for name in ('disabled', 'other_layer', 'other_prefix', 'other_tokens'):
            with self.subTest(name=name):
                path, result = self.case(name)
                self.assertEqual(result, dict(success=1, copies=0, executions=1))
                self.assertFalse(path.exists())

    def test_original_state_boundaries_outputs_and_terminal_row_offsets(self):
        for name in ('key_major', 'value_major'):
            with self.subTest(name=name):
                path, result = self.case(name)
                self.assertEqual(result['executions'], 1)
                self.assertTrue(result['success'])
                record = json.loads((path/'capture.json').read_text())
                self.assertEqual(record['state_layout'], name.replace('_','-'))
                self.assertEqual((record['layer'],record['first_position'],record['tokens']), (16,90112,1024))
                self.assertTrue(record['complete'])
                self.assertFalse(record['inference_acceptance'])
                for file, size, first, last in (
                    ('raw-f32.bin',1024*8192*4,1,1), ('gates-f32.bin',1024*64*4,2,2),
                    ('initial-state-f32.bin',524288*4,3,3),
                    ('output-f32.bin',1024*4096*4,4,4), ('final-state-f32.bin',524288*4,5,6),
                    ('terminal-core-f32.bin',4096*4,7,4)):
                    p=path/file
                    self.assertEqual(p.stat().st_size,size)
                    with p.open('rb') as f:
                        self.assertEqual(struct.unpack('<f',f.read(4))[0],first)
                        f.seek(-4,2)
                        self.assertEqual(struct.unpack('<f',f.read(4))[0],last)
                self.assertEqual(struct.unpack('<8H',(path/'terminal-qkv_projection-bf16.bin').read_bytes()),
                                 tuple(range(0x3f80,0x3f88)))

    def test_failures_leave_no_completion_and_preserve_existing_directory(self):
        for name, executions in (('copy_failure',0),('post_copy_failure',1),('execute_failure',1),('existing',0)):
            with self.subTest(name=name):
                path,result=self.case(name)
                self.assertFalse(result['success'])
                self.assertEqual(result['executions'],executions)
                self.assertFalse((path/'capture.json').exists())
                if name == 'existing':
                    self.assertEqual([p.name for p in path.iterdir()],['sentinel'])
                    self.assertEqual((path/'sentinel').read_text(),'preserve')
