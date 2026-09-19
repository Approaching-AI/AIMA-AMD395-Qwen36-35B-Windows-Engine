"""Exercise real temporary-owner replacement under a strict allocation budget."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DiscardableWorkspaceTests(unittest.TestCase):
    def test_peak_budget_failure_ownership_and_retry(self):
        code = r'''
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <map>
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
std::map<void*,size_t> live;
size_t used=0,limit=0,peak=0,allocations=0,frees=0;
bool fail_allocate=false,fail_free=false;
hipError_t hipMalloc(void** out,size_t bytes){
    ++allocations;*out=nullptr;
    if(fail_allocate||used+bytes>limit)return hipErrorUnknown;
    *out=std::malloc(bytes);assert(*out);live[*out]=bytes;used+=bytes;
    if(used>peak)peak=used;return hipSuccess;
}
hipError_t hipFree(void* pointer){
    if(fail_free)return hipErrorUnknown;
    assert(live.count(pointer));used-=live.at(pointer);live.erase(pointer);
    std::free(pointer);++frees;return hipSuccess;
}
#include "native/providers/ck_fmha/discardable_workspace.h"
int main(){
    using qrt_discardable_workspace::grow;
    unsigned* pointer=nullptr;unsigned capacity=0;
    assert(grow(pointer,capacity,0u,16u)==hipErrorInvalidValue&&!allocations);
    assert(grow(pointer,capacity,4u,0u)==hipErrorInvalidValue&&!allocations);
    capacity=1;assert(grow(pointer,capacity,4u,16u)==hipErrorInvalidValue&&!allocations);capacity=0;
    fail_allocate=true;limit=16;
    assert(grow(pointer,capacity,4u,16u)==hipErrorUnknown&&!pointer&&!capacity&&live.empty());
    fail_allocate=false;assert(grow(pointer,capacity,4u,16u)==hipSuccess&&capacity==4&&used==16);
    pointer[0]=123;const auto saved=pointer;const auto before=allocations;
    fail_allocate=true;
    assert(grow(pointer,capacity,2u,8u)==hipSuccess&&pointer==saved&&pointer[0]==123&&allocations==before);
    fail_free=true;
    assert(grow(pointer,capacity,8u,32u)==hipErrorUnknown&&pointer==saved&&capacity==4&&pointer[0]==123&&allocations==before);
    fail_free=false;limit=32;
    assert(grow(pointer,capacity,8u,32u)==hipErrorUnknown&&!pointer&&!capacity&&live.empty()&&used==0);
    fail_allocate=false;
    assert(grow(pointer,capacity,8u,32u)==hipSuccess&&capacity==8&&used==32);
    // Only the larger allocation fits; an allocate-before-free implementation
    // would require96 bytes and fail this64-byte budget.
    limit=64;
    assert(grow(pointer,capacity,16u,64u)==hipSuccess&&capacity==16&&used==64&&peak==64);
    assert(frees==2);assert(hipFree(pointer)==hipSuccess&&live.empty()&&used==0);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-discardable-') as directory:
            exe = str(Path(directory)/'check')
            subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            '-I', str(ROOT), '-x', 'c++', '-', '-o', exe],
                           input=code, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
