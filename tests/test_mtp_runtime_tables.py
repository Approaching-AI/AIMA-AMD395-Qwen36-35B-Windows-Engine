"""Exercise table joining and actual file/upload lifetime with API doubles."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MtpRuntimeTableTests(unittest.TestCase):
    def test_bound_format_completed_publication_and_unknown_copy_lifetime(self):
        header = (ROOT/'native/providers/sm121_mtp_runtime_tables.h').read_text()
        header = '\n'.join(line for line in header.splitlines()
            if not line.startswith(('#include', '#pragma once')))
        declarations = []
        for filename, name in (('sm121_mtp_moe.h','MoeTables'),('sm121_mtp_drafter.h','DrafterTables')):
            text = (ROOT/'native/providers/gdn'/filename).read_text()
            start = text.index('struct '+name+' {')
            declarations.append(text[start:text.index('\n};',start)+3])
        source = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#define _WIN32 1
using hipError_t=int;using hipStream_t=void*;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipErrorOutOfMemory=2,hipMemcpyHostToDevice=3,injected=99;
static int mode=0;static unsigned full_calls=0,moe_calls=0,rcp_calls=0,copies=0,waits=0;
static std::map<void*,bool> allocated;
static void* pending_device=nullptr;static const void* pending_host=nullptr;static size_t pending_bytes=0;
static hipError_t hipMalloc(void** out,size_t bytes){
    if(mode==5)return hipErrorOutOfMemory;*out=std::malloc(bytes);assert(*out);allocated[*out]=false;return hipSuccess;
}
static hipError_t hipHostMalloc(void** out,size_t bytes){
    if(mode==4)return hipErrorOutOfMemory;*out=std::malloc(bytes);assert(*out);allocated[*out]=true;return hipSuccess;
}
static hipError_t hipFree(void* pointer){
    assert(allocated.count(pointer)&&!allocated.at(pointer)&&pointer!=pending_device);
    allocated.erase(pointer);std::free(pointer);return hipSuccess;
}
static hipError_t hipHostFree(void* pointer){
    assert(allocated.count(pointer)&&allocated.at(pointer)&&pointer!=pending_host);
    allocated.erase(pointer);std::free(pointer);return hipSuccess;
}
static hipError_t hipMemcpyAsync(void* to,const void* from,size_t bytes,int kind,hipStream_t stream){
    assert(kind==hipMemcpyHostToDevice&&!stream);++copies;
    pending_device=to;pending_host=from;pending_bytes=bytes;return mode==6||mode==8?injected:hipSuccess;
}
static hipError_t hipStreamSynchronize(hipStream_t stream){
    assert(!stream);++waits;if(mode==7||mode==8)return injected;
    if(pending_bytes)std::memcpy(pending_device,pending_host,pending_bytes);
    pending_device=nullptr;pending_host=nullptr;pending_bytes=0;return hipSuccess;
}
using BCRYPT_ALG_HANDLE=void*;using NTSTATUS=long;using ULONG=unsigned long;
constexpr const wchar_t* BCRYPT_SHA256_ALGORITHM=L"SHA256";
static NTSTATUS BCryptOpenAlgorithmProvider(BCRYPT_ALG_HANDLE* out,const wchar_t*,void*,int){
    if(mode==9)return -1;*out=reinterpret_cast<void*>(1);return 0;
}
static NTSTATUS BCryptCloseAlgorithmProvider(BCRYPT_ALG_HANDLE,int){return 0;}
// The Windows hash API is a double. It recognizes the entire synthetic input;
// these checks verify rejection/publication flow, not SHA256 or sigmoid math.
static NTSTATUS BCryptHash(BCRYPT_ALG_HANDLE,void*,int,unsigned char* input,ULONG bytes,unsigned char* output,size_t count){
    if(mode==10)return -1;assert(count==32&&bytes==131072);
    const unsigned char digest[]={0x32,0x92,0x3b,0x94,0xec,0xa9,0x38,0xcd,0x0f,0x96,0x6f,0x39,0xef,0xb5,0xfc,0xbd,
        0xa4,0x0f,0x2c,0x2b,0xb7,0x48,0xcd,0xd9,0x0b,0xfd,0x3d,0xde,0xff,0x9e,0x8f,0x97};
    bool exact=true;for(size_t i=0;i<bytes;++i)exact=exact&&input[i]==0x5a;
    if(exact)std::memcpy(output,digest,count);else std::memset(output,0,count);return 0;
}
static const unsigned char *exp2_table=reinterpret_cast<const unsigned char*>(0x1000),
    *rsqrt_table=reinterpret_cast<const unsigned char*>(0x2000),*rcp_table=reinterpret_cast<const unsigned char*>(0x3000);
static const uint16_t *rope_table=reinterpret_cast<const uint16_t*>(0x4000),*silu_table=reinterpret_cast<const uint16_t*>(0x5000);
static const uint32_t* router_table=reinterpret_cast<const uint32_t*>(0x6000);
namespace qrt_sm121_q1_runtime {struct Tables{const unsigned char *exp2=nullptr,*rsqrt=nullptr;const float* beta=nullptr;};}
namespace qrt_sm121_q1_full_runtime {
struct Tables{qrt_sm121_q1_runtime::Tables core;const uint16_t* rope=nullptr;size_t rope_rows=0;};
hipError_t prepare(Tables* out,size_t){++full_calls;if(mode==11)return injected;
    *out={{exp2_table,rsqrt_table,reinterpret_cast<const float*>(0x7000)},rope_table,262144};return hipSuccess;}
}
namespace qrt_sm121_q1_moe_runtime {
struct Tables{qrt_sm121_q1_runtime::Tables core;const uint32_t* router=nullptr;const uint16_t* silu=nullptr;};
hipError_t prepare(Tables* out){++moe_calls;if(mode==12)return injected;
    *out={{exp2_table,mode==14?nullptr:rsqrt_table,reinterpret_cast<const float*>(0x7000)},router_table,silu_table};return hipSuccess;}
}
namespace qrt_sm121_q1_attention_runtime {
hipError_t prepare(const unsigned char** out){++rcp_calls;if(mode==13)return injected;*out=rcp_table;return hipSuccess;}
}
namespace qrt_sm121_mtp {
'''+'\n'.join(declarations)+'\n}\n'+header+r'''
int main(int argc,char** argv){
    assert(argc==3);mode=std::stoi(argv[1]);const std::string path=argv[2];
    std::vector<unsigned char> data(mode==2?262144u:131072u,0x5a);
    if(mode==3)data[65537]=0x59;
    if(mode!=1){std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(data.data()),data.size());}
    assert(!setenv("QRT_QWEN36_MTP_BF16_SIGMOID_TABLE",path.c_str(),1));
    qrt_sm121_mtp::DrafterTables tables;
    if(mode==15){
        assert(qrt_sm121_mtp_runtime::prepare(&tables,262144)==hipErrorInvalidValue&&!full_calls&&!tables.rsqrt);return 0;
    }
    auto status=qrt_sm121_mtp_runtime::prepare(&tables,8191);
    if(mode==0){
        assert(status==hipSuccess&&tables.rsqrt==rsqrt_table&&tables.exp2==exp2_table&&tables.reciprocal==rcp_table);
        assert(tables.rope==rope_table&&tables.rope_rows==262144&&tables.moe.silu==silu_table&&tables.moe.router_exp_fraction==router_table);
        assert(tables.moe.sigmoid&&std::memcmp(tables.moe.sigmoid,data.data(),data.size())==0);
        assert(copies==1&&waits==1&&allocated.size()==1);
        const auto* original=tables.moe.sigmoid;
        assert(qrt_sm121_mtp_runtime::prepare(&tables,262143)==hipSuccess&&tables.moe.sigmoid==original&&copies==1);
        assert(!setenv("QRT_QWEN36_MTP_BF16_SIGMOID_TABLE",(path+".different").c_str(),1));
        assert(qrt_sm121_mtp_runtime::prepare(&tables,8191)==hipErrorInvalidValue&&!tables.rsqrt&&!tables.moe.sigmoid);
    }else{
        assert(status!=hipSuccess&&!tables.rsqrt&&!tables.exp2&&!tables.rope&&!tables.moe.sigmoid);
        if(mode==7||mode==8){
            assert(status==injected&&copies==1&&allocated.size()==2&&pending_bytes==131072);
            const auto previous_waits=waits;
            assert(qrt_sm121_mtp_runtime::prepare(&tables,8191)==injected&&copies==1&&waits==previous_waits);
            mode=0;assert(hipStreamSynchronize(nullptr)==hipSuccess);
        }else{assert(allocated.empty()&&!pending_bytes);}
    }
    while(!allocated.empty()){
        auto item=*allocated.begin();if(item.second)hipHostFree(item.first);else hipFree(item.first);
    }
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); path = directory/'check.cpp'; path.write_text(source)
            exe = directory/'check'
            build = subprocess.run(['c++','-std=c++17','-O1','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-fno-sanitize-recover=all',str(path),'-o',str(exe)],
                capture_output=True,text=True,timeout=60)
            self.assertEqual(build.returncode,0,build.stderr)
            for case in range(16):
                with self.subTest(case=case):
                    run = subprocess.run([str(exe),str(case),str(directory/f'case-{case}.bin')],
                        capture_output=True,text=True,timeout=15)
                    self.assertEqual(run.returncode,0,run.stdout+run.stderr)


if __name__ == '__main__':
    unittest.main()
