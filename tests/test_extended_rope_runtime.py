"""Exercise real RoPE file/position guards and partial allocation recovery.

HIP and Windows cryptography are mocked here. The actual table SHA256 and
constructor equality are verified by the separate GB10 capture evidence.
"""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class ExtendedRopeRuntimeTests(unittest.TestCase):
    def test_q1_loader_checks_actual_extent_and_recovers_before_publication(self):
        header = (ROOT / "native/providers/sm121_q1_full_runtime.h").read_text()
        for dependency in ("sm121_q1_runtime.h", "gdn/sm121_q1_full.h"):
            header = header.replace(f'#include "{dependency}"', "")
        source = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#include "sm121_rope_cache.h"
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorOutOfMemory,hipErrorUnknown};
constexpr int hipMemcpyHostToDevice=1;
static unsigned allocations=0,frees=0,hashes=0;
static std::string fault,path;
hipError_t hipMemGetInfo(size_t *available,size_t *total) {
    *total=1ull<<32;*available=fault=="reserve" ? 0u : *total;
    if(fault=="short_read") std::filesystem::resize_file(path,std::filesystem::file_size(path)-1u);
    return fault=="mem_info" ? hipErrorUnknown : hipSuccess;
}
hipError_t hipMalloc(void **out,size_t size) {
    if(fault=="allocate") return hipErrorOutOfMemory;
    *out=std::malloc(size);if(!*out)return hipErrorOutOfMemory;++allocations;return hipSuccess;
}
hipError_t hipFree(void *value) {std::free(value);++frees;return hipSuccess;}
hipError_t hipMemcpy(void *out,const void *input,size_t bytes,int) {
    if(fault=="copy")return hipErrorUnknown;
    std::memcpy(out,input,bytes);return hipSuccess;
}
using BCRYPT_ALG_HANDLE=void*;using NTSTATUS=int32_t;using ULONG=uint32_t;
constexpr const char *BCRYPT_SHA256_ALGORITHM="sha256";
int BCryptOpenAlgorithmProvider(BCRYPT_ALG_HANDLE *out,const char*,void*,int) {
    if(fault=="crypto_open")return -1;
    *out=reinterpret_cast<void*>(1);return 0;
}
int BCryptCloseAlgorithmProvider(BCRYPT_ALG_HANDLE,int) {return 0;}
NTSTATUS BCryptHash(BCRYPT_ALG_HANDLE,void*,int,unsigned char*,ULONG bytes,unsigned char *out,int count) {
    ++hashes;const auto *layout=qrt_sm121_rope_cache::layout_for_bytes(bytes);
    if(fault=="crypto_hash" || !layout || count!=32)return -1;
    std::memcpy(out,layout->sha256,32);if(fault=="digest")out[0]^=1;return 0;
}
namespace qrt_sm121_q1_runtime {
struct Tables {const unsigned char *exp2=nullptr;};
hipError_t prepare(Tables *out) {
    out->exp2=reinterpret_cast<const unsigned char*>(1);
    return fault=="core" ? hipErrorUnknown : hipSuccess;
}
}
#define _WIN32 1
''' + header + r'''
#undef _WIN32
int main(int argc,char **argv) {
    if(argc!=4)return 1;
    path=argv[1];const size_t rows=std::strtoull(argv[2],nullptr,10);fault=argv[3];
    setenv("QRT_QWEN36_Q1_SM121_ROPE_TABLE",path.c_str(),1);
    using namespace qrt_sm121_q1_full_runtime;
    Tables out;
    if(prepare(nullptr,0u)!=hipErrorInvalidValue)return 2;
    if(fault=="extent") {
        const size_t positions[]={rows,rows+1u,SIZE_MAX};
        for(size_t position:positions) {
            if(prepare(&out,position)!=hipErrorInvalidValue || out.rope || out.rope_rows || allocations || hashes)return 3;
        }
        fault.clear();
    } else if(fault!="valid") {
        if(prepare(&out,0u)==hipSuccess || out.rope || out.rope_rows || allocations!=frees)return 4;
        if(fault=="short_read")std::filesystem::resize_file(path,rows*128u);
        fault.clear();
    }
    if(prepare(&out,rows-1u)!=hipSuccess || !out.rope || out.rope_rows!=rows || allocations!=frees+1u)return 5;
    const auto *resident=out.rope;const auto before=hashes;
    for(size_t position:{size_t(0),rows-1u}) {
        if(prepare(&out,position)!=hipSuccess || out.rope!=resident || out.rope_rows!=rows || hashes!=before)return 6;
    }
    for(size_t position:{rows,rows+1u,SIZE_MAX}) {
        if(prepare(&out,position)!=hipErrorInvalidValue || out.rope || out.rope_rows || hashes!=before)return 7;
    }
    setenv("QRT_QWEN36_Q1_SM121_ROPE_TABLE","changed-table",1);
    if(prepare(&out,0u)!=hipErrorInvalidValue || out.rope || out.rope_rows)return 8;
    setenv("QRT_QWEN36_Q1_SM121_ROPE_TABLE",path.c_str(),1);
    if(prepare(&out,rows-1u)!=hipSuccess || out.rope!=resident)return 9;
    hipFree(const_cast<uint16_t*>(resident));
    return allocations==frees ? 0 : 10;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-rope-runtime-") as temporary:
            executable = str(Path(temporary) / "loader")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-I", str(ROOT / "native/providers"), "-x", "c++", "-", "-o", executable],
                input=source, text=True, check=True, timeout=30,
            )
            for rows in (262144, 264736):
                table = Path(temporary) / f"rope-{rows}.bin"
                with table.open("xb") as stream:
                    stream.truncate(rows * 128)
                for fault in ("valid", "extent", "core", "mem_info", "reserve", "short_read",
                              "crypto_open", "crypto_hash", "digest", "allocate", "copy"):
                    with self.subTest(rows=rows, fault=fault):
                        subprocess.run([executable, str(table), str(rows), fault],
                                       check=True, capture_output=True, text=True, timeout=10)

    def test_prefill_request_and_authoritative_table_extent(self):
        provider = (ROOT / "native/providers/whole_provider.cpp").read_text()
        request_expression = "const bool arbitrary_prefill_requested =" + provider.split(
            "const bool arbitrary_prefill_requested =", 1)[1].split("const bool cold_q8192_requested =", 1)[0]
        loader = function(provider, "bool load_gb10_full_attention_rope_cache_locked(")
        source = r'''
#include "qrt.h"
#include "sm121_rope_cache.h"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
constexpr size_t kGb10FullAttentionRopeCacheRowBytes=128u,kGb10FullAttentionRopeCacheColumns=64u;
struct FullAttentionCompactRopeTableState {
    std::string authoritative_source_path;
    std::vector<uint16_t> authoritative_bf16;
    unsigned int authoritative_row_count=0;
};
bool request_supported(const qrt_qwen36_whole_provider_request_t *request) {
''' + request_expression + "return arbitrary_prefill_requested;\n}\n" + loader + r'''
int main(int argc,char **argv) {
    if(argc!=2)return 1;
    qrt_qwen36_whole_provider_request_t request{};
    request.flags=QRT_QWEN36_WHOLE_PROVIDER_FLAG_ARBITRARY_PREFILL;
    for(size_t count:{size_t(0),size_t(262144),size_t(262145),size_t(263168),size_t(263169),SIZE_MAX}) {
        request.input_token_count=count;
        if(request_supported(&request)!=(count>0u && count<=263168u))return 2;
    }
    request.input_token_count=263168u;request.flags=0;
    if(request_supported(&request) || request_supported(nullptr))return 3;
    std::string path=argv[1],failure;FullAttentionCompactRopeTableState state;
    for(size_t rows:{size_t(262144),size_t(263680),size_t(264736),size_t(264737),size_t(0)}) {
        {std::ofstream file(path,std::ios::binary|std::ios::trunc);}
        std::filesystem::resize_file(path,rows*128u);state={};
        const bool valid=rows>0u && rows<=264736u;
        if(load_gb10_full_attention_rope_cache_locked(path,&state,&failure)!=valid)return 4;
        if(valid && (state.authoritative_row_count!=rows || state.authoritative_bf16.size()!=rows*64u))return 5;
        if(!valid && (!state.authoritative_bf16.empty() || state.authoritative_row_count))return 6;
    }
    for(size_t bytes:{size_t(1),size_t(262144)*128u-1u,size_t(264736)*128u+1u}) {
        std::filesystem::resize_file(path,bytes);state={};
        if(load_gb10_full_attention_rope_cache_locked(path,&state,&failure))return 7;
        if(qrt_sm121_rope_cache::layout_for_bytes(bytes))return 8;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-rope-extent-") as temporary:
            executable = str(Path(temporary) / "extent")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-I", str(ROOT / "native/src"), "-I", str(ROOT / "native/providers"),
                 "-x", "c++", "-", "-o", executable],
                input=source, text=True, check=True, timeout=30,
            )
            subprocess.run([executable, str(Path(temporary) / "table.bin")],
                           check=True, capture_output=True, text=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
