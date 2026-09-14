"""Exercise the replay tool's actual completion check with both PV routes."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AttentionCapturePhaseTests(unittest.TestCase):
    def test_skipped_stages_and_incomplete_observers_are_distinguished(self):
        replay = (ROOT / "native/providers/ck_fmha/full_attention_capture_replay.cpp").read_text()
        body = "struct CompletedAttentionPhases" + replay.split("struct CompletedAttentionPhases", 1)[1].split("\nint main(", 1)[0]
        source = r'''
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <limits>
using Clock=std::chrono::steady_clock;
using hipStream_t=void*;
enum hipError_t { hipSuccess,hipErrorInvalidValue,hipErrorUnknown };
bool fail_sync=false;
hipError_t hipStreamSynchronize(hipStream_t) { return fail_sync?hipErrorUnknown:hipSuccess; }
''' + body + r'''
int main() {
    for(bool all:{false,true}) for(unsigned slabs:{1u,57u}) {
        CompletedAttentionPhases valid;valid.previous=Clock::now();
        if(valid.complete(slabs,all) || valid.complete(0u,all))return 1;
        for(unsigned slab=0;slab<slabs;++slab)for(unsigned stage=0;stage<5u;++stage) {
            if(all && (stage==2u || stage==3u))continue;
            if(CompletedAttentionPhases::observe(&valid,stage,nullptr)!=hipSuccess)return 2;
        }
        if(!valid.complete(slabs,all) || valid.complete(slabs,!all))return 3;
        for(unsigned stage=0;stage<5u;++stage) {
            auto extra=valid;++extra.samples[stage];
            if(extra.complete(slabs,all))return 4;
            if(valid.samples[stage]) {
                auto missing=valid;--missing.samples[stage];
                if(missing.complete(slabs,all))return 5;
            } else {
                auto timed=valid;timed.milliseconds[stage]=1.0;
                if(timed.complete(slabs,all))return 6;
            }
            for(double bad:{-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
                auto invalid=valid;invalid.milliseconds[stage]=bad;
                if(invalid.complete(slabs,all))return 7;
            }
        }
        const auto previous=valid.previous;const auto samples=valid.samples[4];
        fail_sync=true;
        if(CompletedAttentionPhases::observe(&valid,4u,nullptr)!=hipErrorUnknown ||
            valid.samples[4]!=samples || valid.previous!=previous)return 8;
        fail_sync=false;
        if(CompletedAttentionPhases::observe(&valid,5u,nullptr)!=hipErrorInvalidValue)return 9;
        if(!valid.complete(slabs,all))return 10;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            cpp = path / "capture-phases.cpp"
            cpp.write_text(source)
            binary = path / "capture-phases"
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-fsanitize=address,undefined", "-fno-omit-frame-pointer", str(cpp), "-o", str(binary)],
                check=True, timeout=30,
            )
            subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
