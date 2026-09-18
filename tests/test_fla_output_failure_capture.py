"""Bound diagnostic copies and preserve failures without changing device data."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OutputFailureCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="qrt-fla-output-failure-")
        cls.root = Path(cls.temp.name)
        cls.exe = cls.root / "probe"
        source = r'''
#include "native/providers/gdn/output_failure_capture.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
int main(int argc,char** argv){
 assert(argc==3);const std::string mode(argv[2]);
 const unsigned n=mode=="full"?1024u:65u;
 std::vector<uint16_t> q(n*2048u,1),k(n*2048u,2),v(n*4096u,3),
     h(((n+63u)/64u)*524288u,4),s(n*2048u,5);
 std::vector<float> g(n*32u,6),o(n*4096u,7);
 q[0]=0x7fc1u; // Diagnostic data must retain NaN payload bits too.
 struct Span{const void* data;size_t bytes;};
 const Span spans[]={{q.data(),q.size()*2u},{k.data(),k.size()*2u},{v.data(),v.size()*2u},
     {h.data(),h.size()*2u},{g.data(),g.size()*4u},{s.data(),s.size()*2u},{o.data(),o.size()*4u}};
 qrt_fla_completion::Observation clock{true,283.885010,284.029600};
 unsigned calls=0;size_t copied=0;
 auto copy=[&](void* host,const void* device,size_t bytes){
   assert(bytes>0&&bytes<=qrt_fla_output_failure::copy_bytes);
   const auto address=reinterpret_cast<uintptr_t>(device);bool inside=false;
   for(const auto& span:spans){const auto start=reinterpret_cast<uintptr_t>(span.data);
     inside|=address>=start&&address+bytes<=start+span.bytes;}
   assert(inside);++calls;copied+=bytes;
   if(mode=="copy_failure"&&calls==3)return false;
   if(mode=="copy_exception")throw std::runtime_error("read failed");
   std::memcpy(host,device,bytes);return true;
 };
 if(mode=="incomplete")clock.completed=false;
 if(mode=="accepted_gpu")clock.gpu_ms=99;
 if(mode=="accepted_host")clock.host_ms=99;
 if(mode=="invalid_clock")clock.gpu_ms=std::numeric_limits<double>::infinity();
 if(mode=="negative_clock")clock.gpu_ms=-1;
 const char* directory=mode=="disabled"?nullptr:argv[1];
 bool result=qrt_fla_output_failure::capture(directory,mode=="invalid_count"?1025u:n,
   mode=="null_input"?nullptr:q.data(),k.data(),v.data(),h.data(),g.data(),s.data(),o.data(),clock,copy);
 assert(q[0]==0x7fc1u&&q.back()==1&&k.front()==2&&k.back()==2&&v.front()==3&&v.back()==3);
 assert(h.front()==4&&h.back()==4&&s.front()==5&&s.back()==5&&g.front()==6&&g.back()==6&&o.front()==7&&o.back()==7);
 std::cout<<"{\"pass\":"<<(result?"true":"false")<<",\"copies\":"<<calls<<",\"copied_bytes\":"<<copied<<"}\n";
}
'''
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all", "-I", str(ROOT), "-x", "c++", "-",
            "-o", str(cls.exe)], input=source, text=True, check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_mode(self, mode, path=None):
        path = path or self.root / mode
        result = subprocess.run([str(self.exe), str(path), mode], check=True,
            capture_output=True, text=True, timeout=10)
        return path, json.loads(result.stdout)

    def test_complete_partial_and_maximum_segment_copies(self):
        for mode, count in (("partial", 65), ("full", 1024)):
            with self.subTest(mode=mode):
                path, result = self.run_mode(mode)
                record = json.loads((path / "capture.json").read_text())
                self.assertTrue(result["pass"] and record["complete"] and record["stage_completed"])
                self.assertFalse(record["guard_accepted"] or record["kernel_retried"] or record["numerical_acceptance"])
                self.assertEqual(record["tokens"], count)
                self.assertEqual(record["bytes"], result["copied_bytes"])
                self.assertLessEqual(record["bytes"], 64 << 20)
                self.assertEqual((path / "q-bf16.bin").read_bytes(), b'\xc1\x7f' + b'\x01\x00' * (count * 2048 - 1))
                self.assertEqual(sum(p.stat().st_size for p in path.glob('*.bin')), record["bytes"])
                self.assertEqual((path / "chunk-state-bf16.bin").stat().st_size, ((count + 63) // 64) * 1048576)

    def test_disabled_and_unqualified_operations_do_not_read(self):
        for mode in ("disabled", "incomplete", "accepted_gpu", "accepted_host", "invalid_clock",
                     "negative_clock", "invalid_count", "null_input"):
            with self.subTest(mode=mode):
                path, result = self.run_mode(mode)
                self.assertEqual(result["pass"], mode == "disabled")
                self.assertEqual(result["copies"], 0)
                self.assertFalse(path.exists())

    def test_copy_failure_cannot_publish_completion(self):
        for mode in ("copy_failure", "copy_exception"):
            path, result = self.run_mode(mode)
            self.assertFalse(result["pass"])
            self.assertGreater(result["copies"], 0)
            self.assertFalse((path / "capture.json").exists())

    def test_existing_evidence_is_preserved(self):
        path = self.root / "existing"
        path.mkdir()
        sentinel = path / "capture.json"
        sentinel.write_bytes(b'ORIGINAL\n')
        _, result = self.run_mode("existing", path)
        self.assertFalse(result["pass"])
        self.assertEqual(result["copies"], 0)
        self.assertEqual(sentinel.read_bytes(), b'ORIGINAL\n')
