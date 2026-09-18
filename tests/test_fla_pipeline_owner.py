"""Exercise production host ownership with a dependency-graph HIP model."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class PipelineOwnerTests(unittest.TestCase):
    def test_all_preallocated_storage_is_reported(self):
        provider = (ROOT / "native/providers/gdn/qrt_fla_chunk_gdn_q8192_provider.cpp").read_text()
        export = function(provider, "QRT_FLA_GDN_EXPORT uint64_t qrt_fla_chunk_gdn_scratch_bytes(")
        code = r'''
#include "native/providers/gdn/pipelined_segment_policy.h"
#include <cassert>
#include <cstdint>
#include <initializer_list>
constexpr int32_t kSegmentTokens=1024;
constexpr unsigned kMainScratchBytesPerToken=49344;
constexpr unsigned kTailPaddingBytes=3162112,kStateElements=524288,kValueHeads=32,kChunk=64;
bool compatible=true,diagnostic=false;
struct {bool ready=false;} g_pipeline;
struct {void *blackwell_temporary_state=nullptr,*blackwell_residual=nullptr,*seeded_row_state=nullptr;} g_state;
bool supported_tokens(int n){return n>0&&n<=65536;}
int padded_tokens(int n){return (n+63)/64*64;}
bool pipeline_compatible(){return compatible;}bool pipeline_diagnostic(){return diagnostic;}
bool blackwell_state_enabled(){return true;}
uint64_t blackwell_state_scratch_bytes(){return 6291456;}
namespace qrt_fla_blackwell_state {uint64_t exp2_table_storage_bytes(){return 0;}}
namespace qrt_fla_blackwell_norm {uint64_t table_storage_bytes(){return 0;}}
namespace qrt_fla_checkpoint {constexpr uint64_t kStateBytes=2097152;}
#define QRT_FLA_GDN_EXPORT
''' + export + r'''
int main(){
 for(const char* mode:{"0","1","2"})for(bool ready:{false,true})
 for(bool c:{false,true})for(bool d:{false,true})for(int tokens:{0,1,64,65,1025,8192,65536,65537}){
  setenv("QRT_FLA_GDN_PIPELINED_SEGMENTS",mode,1);g_pipeline.ready=ready;compatible=c;diagnostic=d;
  const uint64_t base=supported_tokens(tokens)?uint64_t(tokens>1024?1024:padded_tokens(tokens))*49344+3162112+6291456:0;
  const uint64_t expected=base+(base&&(ready||(mode[0]!='0'&&c&&!d))?179945472:0);
  assert(qrt_fla_chunk_gdn_scratch_bytes(tokens)==expected);
 }
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-pipeline-storage-") as directory:
            exe = str(Path(directory) / "storage")
            subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=undefined", "-fno-sanitize-recover=all", "-I", str(ROOT),
                            "-x", "c++", "-", "-o", exe],
                           input=code, text=True, check=True, timeout=60)
            subprocess.run([exe], check=True, timeout=60)

    def test_dependencies_reuse_and_partial_failures(self):
        with tempfile.TemporaryDirectory(prefix="qrt-pipeline-owner-") as directory:
            exe = str(Path(directory) / "owner")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/pipelined_segment_owner_host.cpp"),
                            "-o", exe], check=True, timeout=60)
            subprocess.run([exe], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
