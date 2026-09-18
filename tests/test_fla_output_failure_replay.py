"""Exercise replay ownership and bounds with a CPU HIP shim, not GPU numerics."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

HIP = r'''
#pragma once
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdint>
enum hipError_t { hipSuccess, hipErrorUnknown };
using hipEvent_t=void*;
using hipStream_t=void*;
struct hipDeviceProp_t { char gcnArchName[256]; };
enum hipMemcpyKind { hipMemcpyHostToDevice,hipMemcpyDeviceToHost };
inline unsigned allocations=0,frees=0,creates=0,destroys=0,records=0,waits=0,drains=0;
inline unsigned score_calls=0,output_calls=0,table_releases=0;
inline bool fail(const char* name) {
 const char* value=std::getenv("REPLAY_TEST_FAILURE");return value && !std::strcmp(value,name);
}
inline const char* hipGetErrorString(hipError_t) {return "injected HIP error";}
inline hipError_t hipSetDevice(int) {return hipSuccess;}
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* p,int) {
 std::strcpy(p->gcnArchName,fail("architecture")?"gfx0000":"gfx1151");return hipSuccess;
}
inline hipError_t hipMemGetInfo(size_t* f,size_t* t) {*f=*t=fail("memory")?0:1ull<<30;return hipSuccess;}
inline hipError_t hipMalloc(void** p,size_t n) {
 *p=std::malloc(n);if(!*p)return hipErrorUnknown;++allocations;return hipSuccess;
}
inline hipError_t hipFree(void* p) {++frees;std::free(p);return hipSuccess;}
inline hipError_t hipMemcpy(void* a,const void* b,size_t n,hipMemcpyKind) {std::memcpy(a,b,n);return hipSuccess;}
inline hipError_t hipStreamCreate(hipStream_t* p) {*p=reinterpret_cast<void*>(1);return hipSuccess;}
inline hipError_t hipStreamSynchronize(hipStream_t) {++drains;return hipSuccess;}
inline hipError_t hipStreamDestroy(hipStream_t) {return hipSuccess;}
inline hipError_t hipEventCreate(hipEvent_t* p) {*p=reinterpret_cast<void*>(uintptr_t(++creates));return hipSuccess;}
inline hipError_t hipEventDestroy(hipEvent_t) {++destroys;return hipSuccess;}
inline hipError_t hipEventRecord(hipEvent_t,hipStream_t) {++records;return fail("record")?hipErrorUnknown:hipSuccess;}
inline hipError_t hipEventSynchronize(hipEvent_t) {
 ++waits;if(fail("slow"))std::this_thread::sleep_for(std::chrono::milliseconds(120));
 return fail("sync")?hipErrorUnknown:hipSuccess;
}
inline hipError_t hipEventElapsedTime(float* value,hipEvent_t,hipEvent_t) {
 *value=fail("slow")?150.0f:fail("negative")?-0.5f:1.0f;
 return fail("elapsed")?hipErrorUnknown:hipSuccess;
}
'''

OPERATIONS = r'''
#include "blackwell_state.h"
#include "blackwell_cooperative.h"
#include "blackwell_wu_output.h"
#include <algorithm>
namespace qrt_fla_blackwell_state {
hipError_t prepare_exp2_table() {return hipSuccess;}
void release_exp2_table() {++table_releases;}
const unsigned char* exp2_table_device() {static unsigned char table;return &table;}
}
namespace qrt_fla_blackwell_cooperative {
hipError_t scores(const uint16_t* q,const uint16_t* k,const float* g,uint16_t* s,
 unsigned n,const unsigned char*,hipStream_t) {
 ++score_calls;if(fail("operation"))return hipErrorUnknown;
 for(unsigned t=0;t<n;++t)std::fill_n(s+size_t(t)*2048u,2048u,uint16_t(q[size_t(t)*2048u]+k[size_t(t)*2048u]+g[size_t(t)*32u]));
 if(fail("input"))const_cast<uint16_t*>(q)[0]^=1u;
 if(fail("redzone"))s[-1]=0u;
 return hipSuccess;
}
hipError_t output(const uint16_t* q,const uint16_t* v,const uint16_t* h,const float* g,
 const uint16_t* s,float* out,unsigned n,const unsigned char*,hipStream_t) {
 ++output_calls;
 for(unsigned t=0;t<n;++t)std::fill_n(out+size_t(t)*4096u,4096u,
  float(q[size_t(t)*2048u]+v[size_t(t)*4096u]+h[size_t(t/64u)*524288u]+s[size_t(t)*2048u]+g[size_t(t)*32u]));
 if(fail("mismatch"))out[0]+=1.0f;
 return hipSuccess;
}
}
namespace qrt_fla_blackwell_aux {
hipError_t output_segment(const uint16_t* q,const uint16_t* k,const uint16_t* v,const uint16_t* h,
 const float* g,uint16_t* s,float* out,unsigned n,hipStream_t stream) {
 const auto* table=qrt_fla_blackwell_state::exp2_table_device();
 auto e=qrt_fla_blackwell_cooperative::scores(q,k,g,s,n,table,stream);
 if(e!=hipSuccess)return e;
 return qrt_fla_blackwell_cooperative::output(q,v,h,g,s,out,n,table,stream);
}
}
struct Report {
 ~Report() {
  std::fprintf(stderr,"{\"shim\":true,\"allocations\":%u,\"frees\":%u,\"creates\":%u,\"destroys\":%u,\"drains\":%u,\"score_calls\":%u,\"output_calls\":%u,\"table_releases\":%u}\n",
   allocations,frees,creates,destroys,drains,score_calls,output_calls,table_releases);
 }
} report;
'''


@unittest.skipUnless(shutil.which("c++"), "requires C++ compiler")
class FlaOutputFailureReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "hip").mkdir()
        (cls.root / "hip/hip_runtime.h").write_text(HIP)
        (cls.root / "operations.cpp").write_text(OPERATIONS)
        cls.exe = cls.root / "replay"
        subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                        "-fno-sanitize-recover=all", "-I"+str(cls.root),
                        "-I"+str(ROOT / "native/providers/gdn"),
                        str(ROOT / "native/providers/gdn/output_failure_replay.cpp"),
                        str(cls.root / "operations.cpp"), "-o", str(cls.exe)],
                       check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def fixture(self, count):
        root = self.root / ("capture"+str(count))
        root.mkdir(exist_ok=True)
        shapes = (("q-bf16.bin", "H", count, 2048, lambda t:t+1),
                  ("k-bf16.bin", "H", count, 2048, lambda t:t+2),
                  ("v-new-bf16.bin", "H", count, 4096, lambda t:t+3),
                  ("chunk-state-bf16.bin", "H", (count+63)//64, 524288, lambda t:t+4),
                  ("g-cumsum-f32.bin", "f", count, 32, lambda t:t+5),
                  ("scores-bf16.bin", "H", count, 2048, lambda t:3*t+8),
                  ("output-f32.bin", "f", count, 4096, lambda t:6*t+21+t//64))
        for name,kind,rows,columns,value in shapes:
            with (root/name).open("wb") as f:
                for t in range(rows):
                    f.write(struct.pack("<"+kind,value(t))*columns)
        return root

    def run_probe(self, root, count, mode, failure=""):
        result = subprocess.run([str(self.exe),str(root),str(count),mode],
                                env=dict(os.environ,REPLAY_TEST_FAILURE=failure),
                                text=True,capture_output=True,timeout=10)
        shim = json.loads(result.stderr.splitlines()[-1])
        self.assertEqual(shim["allocations"],shim["frees"],result.stderr)
        self.assertEqual(shim["creates"],shim["destroys"],result.stderr)
        records = [json.loads(line) for line in result.stdout.splitlines()]
        return result,shim,records

    def test_modes_offsets_tails_and_comparison_spans(self):
        for count in (1,65,1024):
            root = self.fixture(count)
            for mode in ("combined","scores","output","split","tiles"):
                result,shim,records = self.run_probe(root,count,mode)
                self.assertEqual(result.returncode,0,result.stderr+result.stdout)
                summary = records[-1]
                self.assertEqual(summary["score_rows_recomputed"],0 if mode=="output" else count)
                self.assertEqual(summary["output_rows_recomputed"],0 if mode=="scores" else count)
                self.assertFalse(summary["gb10_acceptance"])
                self.assertFalse(summary["performance_acceptance"])
                if mode=="tiles":
                    self.assertEqual([x["first_token"] for x in records[:-1]],list(range(0,count,64)))
                    self.assertEqual(shim["score_calls"],(count+63)//64)
                self.assertEqual(shim["table_releases"],1)

    def test_rejected_bound_stops_further_submission(self):
        root = self.fixture(65)
        for mode,score_calls,output_calls in (("split",1,0),("tiles",1,1)):
            result,shim,records = self.run_probe(root,65,mode,"slow")
            self.assertEqual(result.returncode,3,result.stderr+result.stdout)
            self.assertEqual((shim["score_calls"],shim["output_calls"]),(score_calls,output_calls))
            self.assertEqual(records[0]["selected_clock"],"unavailable")
            self.assertFalse(records[-1]["completed_bound_pass"])
        result,_,records = self.run_probe(root,65,"split","negative")
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(all(x["selected_clock"]=="host" for x in records[:-1]))

    def test_api_errors_redzones_immutable_inputs_and_mismatch(self):
        root = self.fixture(65)
        for failure in ("record","sync","elapsed","operation","input","redzone","architecture","memory"):
            result,shim,records = self.run_probe(root,65,"split",failure)
            self.assertEqual(result.returncode,1,(failure,result.stderr,result.stdout))
            self.assertFalse(any(x["kind"]=="fla_output_failure_replay" for x in records))
            if failure in ("record","sync","elapsed","operation"):
                self.assertEqual(shim["output_calls"],0)
        result,_,records = self.run_probe(root,65,"combined","mismatch")
        self.assertEqual(result.returncode,4,result.stderr)
        self.assertEqual(records[-1]["output_bit_mismatches"],1)

    def test_bad_sizes_and_arguments_fail_before_allocation(self):
        root = self.fixture(1)
        for count,mode in (("", "combined"),("-1","combined"),("1025","combined"),
                           ("9999999999999","combined"),("1x","combined"),("1","bad")):
            result,shim,_ = self.run_probe(root,count,mode)
            self.assertEqual(result.returncode,1)
            self.assertEqual(shim["allocations"],0)
        (root/"q-bf16.bin").write_bytes(b"")
        result,shim,_ = self.run_probe(root,1,"combined")
        self.assertEqual(result.returncode,1)
        self.assertEqual(shim["allocations"],0)


if __name__ == "__main__":
    unittest.main()
