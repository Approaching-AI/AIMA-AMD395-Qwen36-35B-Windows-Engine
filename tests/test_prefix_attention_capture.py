"""Verify observation preserves the original attention invocation and inputs."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('c++'), 'requires a portable compiler')
class PrefixAttentionCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        source = cls.root / 'probe.cpp'
        source.write_text(r'''
#include "native/providers/prefix_attention_capture.h"
#include <cassert>
#include <iostream>
using namespace qrt_prefix_attention_capture;
int main(int argc, char **argv) {
    assert(argc == 3);
    const std::string mode(argv[2]); std::string error; Plan plan;
    if (mode == "parse") {
        assert(parse(nullptr,nullptr,nullptr,nullptr,plan,error) && !plan.directory);
        for (auto prefix : {"8192","90112","131072"})
            assert(parse(argv[1],"15",prefix,"8192",plan,error));
        assert(surface_bytes(90112,8192) == 402653184u);
        assert(parse(argv[1],"39","131072","1024",plan,error));
        for (auto layer : {"0","16","40","-1","+15","15x"," 15","999999999999",""})
            assert(!parse(argv[1],layer,"90112","8192",plan,error));
        for (auto prefix : {"0","8191","90113","139264","999999999999"})
            assert(!parse(argv[1],"15",prefix,"8192",plan,error));
        for (auto tokens : {"0","1023","1025","8193","-1024","999999999999"})
            assert(!parse(argv[1],"15","90112",tokens,plan,error));
        assert(!parse(argv[1],nullptr,"90112","8192",plan,error));
        assert(!parse(argv[1],"15",nullptr,"8192",plan,error));
        assert(!parse(argv[1],"15","90112",nullptr,plan,error));
        std::cout << "{}\n"; return 0;
    }
    assert(parse(argv[1],"15","8192","1024",plan,error));
    constexpr unsigned tokens=1024, prefix=8192;
    std::vector<uint16_t> q(size_t(tokens)*4096u,0x3f80), pk(size_t(prefix)*512u,0x4000),
        pv(size_t(prefix)*512u,0x4040), tk(size_t(tokens)*512u,0x4080), tv(size_t(tokens)*512u,0x40a0);
    std::vector<float> output(size_t(tokens)*4096u,-1);
    unsigned copies=0,executions=0;
    auto copy=[&](void *dst,const void *src,size_t bytes) {
        assert(bytes && bytes <= qrt_prefix_linear_capture::copy_chunk_bytes); ++copies;
        if (mode == "copy_failure" && copies == 2) return false;
        if (mode == "post_copy_failure" && executions) return false;
        std::memcpy(dst,src,bytes); return true;
    };
    auto execute=[&] {
        ++executions;
        for (const auto *surface : {&q,&pk,&pv,&tk,&tv})
            assert(std::all_of(surface->begin(),surface->end(),[&](uint16_t x){return x==surface->front();}));
        if (mode == "execute_failure") return false;
        std::fill(output.begin(),output.end(),6); output.back()=7; return true;
    };
    unsigned layer=15,before=prefix,count=tokens;
    if(mode=="other_layer") layer=19;
    if(mode=="other_prefix") before=16384;
    if(mode=="other_tokens") count=8192;
    if(mode=="disabled") plan.directory=nullptr;
    if(mode=="existing") { std::filesystem::create_directory(argv[1]);
        std::ofstream(std::filesystem::path(argv[1])/"sentinel") << "preserve"; }
    const bool selected=plan.matches(layer,before,count);
    auto invoke=[&] { return run(plan,layer,before,count,q.data(),pk.data(),pv.data(),tk.data(),tv.data(),
        mode=="missing"?nullptr:output.data(),copy,execute,error); };
    const bool ok=invoke();
    assert(executions<=1);
    if(!selected) assert(ok && executions==1 && !copies && !std::filesystem::exists(argv[1]));
    else if(ok) { const unsigned saved=copies; assert(!invoke()); assert(copies==saved && executions==1); }
    else assert(!std::filesystem::exists(std::filesystem::path(argv[1])/"capture.json"));
    std::cout << "{\"success\":" << ok << ",\"copies\":" << copies << ",\"executions\":" << executions << "}\n";
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
        result = subprocess.run([str(self.binary), str(path), name], check=True,
                                capture_output=True, text=True, timeout=20)
        return path, json.loads(result.stdout)

    def test_bounded_original_transaction_selection(self):
        self.case('parse')

    def test_disabled_and_other_transactions_execute_once_without_reads(self):
        for name in ('disabled', 'other_layer', 'other_prefix', 'other_tokens'):
            with self.subTest(name=name):
                path, result = self.case(name)
                self.assertEqual(result, dict(success=1, copies=0, executions=1))
                self.assertFalse(path.exists())

    def test_exact_original_inputs_and_original_output(self):
        path, result = self.case('capture')
        self.assertEqual(result['executions'], 1)
        self.assertTrue(result['success'])
        record = json.loads((path/'capture.json').read_text())
        self.assertEqual((record['layer'], record['first_position'], record['tokens']), (15,8192,1024))
        self.assertEqual(record['surface_bytes'], 8192*2048+1024*26624)
        self.assertTrue(record['complete'])
        self.assertFalse(record['inference_acceptance'])
        for file, count, bits in (
                ('q-bf16.bin',1024*4096,0x3f80), ('prefix-k-bf16.bin',8192*512,0x4000),
                ('prefix-v-bf16.bin',8192*512,0x4040), ('tail-k-bf16.bin',1024*512,0x4080),
                ('tail-v-bf16.bin',1024*512,0x40a0)):
            self.assertEqual((path/file).read_bytes(), struct.pack('<H',bits)*count)
        self.assertEqual((path/'context-f32.bin').read_bytes(), struct.pack('<f',6)*(1024*4096-1)+struct.pack('<f',7))

    def test_failures_leave_no_completion_and_preserve_existing_directory(self):
        for name, executions in (('copy_failure',0),('post_copy_failure',1),('execute_failure',1),('existing',0),('missing',0)):
            with self.subTest(name=name):
                path, result = self.case(name)
                self.assertFalse(result['success'])
                self.assertEqual(result['executions'],executions)
                self.assertFalse((path/'capture.json').exists())
                if name == 'existing':
                    self.assertEqual([p.name for p in path.iterdir()],['sentinel'])
                    self.assertEqual((path/'sentinel').read_text(),'preserve')
