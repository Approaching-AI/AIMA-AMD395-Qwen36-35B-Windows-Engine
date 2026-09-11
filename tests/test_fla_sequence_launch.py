"""Exercise actual bounded FLA submission and failure cleanup without a GPU."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FlaSequenceLaunchTests(unittest.TestCase):
    def test_bounded_ordered_submission_and_partial_failure_drain(self):
        provider = (ROOT / "native/providers/gdn/qrt_fla_chunk_gdn_q8192_provider.cpp").read_text()
        timing = "template<class Operation>\nbool launch_blackwell_math(" + provider.split(
            "template<class Operation>\nbool launch_blackwell_math(", 1
        )[1].split("bool launch_blackwell_kkt(", 1)[0]
        auxiliary = "template<class Operation>\nbool launch_blackwell_aux(" + provider.split(
            "template<class Operation>\nbool launch_blackwell_aux(", 1
        )[1].split("bool load_kernels(", 1)[0]
        source = r'''
#include <cstdint>
#include <cstdio>
#include <initializer_list>
enum hipError_t { hipSuccess, hipErrorUnknown };
using hipEvent_t = void*;
using hipStream_t = void*;
constexpr unsigned kSegmentTokens = 1024u, kChunk = 64u;
unsigned creates, records, waits, drains, destroys, operations;
unsigned fail_create, fail_record, fail_operation;
bool fail_wait, enabled = true, table = true, ordered = true;
float duration = 12.0f;
hipError_t hipEventCreate(hipEvent_t* event) {
    if (++creates == fail_create) return hipErrorUnknown;
    *event = reinterpret_cast<void*>(uintptr_t(creates)); return hipSuccess;
}
hipError_t hipEventDestroy(hipEvent_t) { ++destroys; return hipSuccess; }
hipError_t hipEventRecord(hipEvent_t, hipStream_t) {
    return ++records == fail_record ? hipErrorUnknown : hipSuccess;
}
hipError_t hipEventSynchronize(hipEvent_t) {
    ++waits; return fail_wait ? hipErrorUnknown : hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) { ++drains; return hipSuccess; }
hipError_t hipEventElapsedTime(float* value, hipEvent_t, hipEvent_t) {
    *value = duration; return hipSuccess;
}
void set_error(const char*, hipError_t) {}
void set_error_text(const char*) {}
bool blackwell_state_enabled() { return enabled; }
namespace qrt_fla_blackwell_state {
const unsigned char* exp2_table_device() {
    return table ? reinterpret_cast<const unsigned char*>(uintptr_t(1)) : nullptr;
}
}
''' + timing + auxiliary + r'''
void reset() {
    creates = records = waits = drains = destroys = operations = 0u;
    fail_create = fail_record = fail_operation = UINT32_MAX;
    fail_wait = false; enabled = table = ordered = true; duration = 12.0f;
}
bool run(unsigned tokens, unsigned calls) {
    return launch_blackwell_aux("test", tokens, calls, nullptr,
        [&](unsigned offset, unsigned call) {
            const unsigned index = operations++;
            ordered = ordered && offset == (index / calls) * 64u && call == index % calls;
            return index == fail_operation ? hipErrorUnknown : hipSuccess;
        });
}
int main() {
    for (unsigned tokens : {0u, 1u, 63u, 65u, 1025u, UINT32_MAX}) {
        reset(); if (run(tokens, 2u) || creates || operations) return 1;
    }
    for (unsigned calls : {0u, 3u, UINT32_MAX}) {
        reset(); if (run(1024u, calls) || creates || operations) return 2;
    }
    reset(); enabled = false;
    if (run(1024u, 2u) || creates || operations) return 3;
    reset(); table = false;
    if (run(1024u, 2u) || creates || operations) return 4;
    for (unsigned tokens : {64u, 128u, 1024u}) for (unsigned calls : {1u, 2u}) {
        reset();
        if (!run(tokens, calls) || !ordered || operations != tokens / 64u * calls ||
            creates != 2u || records != 2u || waits != 1u || drains || destroys != 2u)
            return 5;
    }
    for (unsigned position : {0u, 7u, 31u}) {
        reset(); fail_operation = position;
        if (run(1024u, 2u) || !ordered || operations != position + 1u ||
            records != 1u || waits || drains != 1u || destroys != 2u) return 6;
    }
    reset(); fail_create = 2u;
    if (run(1024u, 2u) || operations || records || drains || destroys != 1u) return 7;
    reset(); fail_record = 1u;
    if (run(1024u, 2u) || operations || waits || drains || destroys != 2u) return 8;
    reset(); fail_record = 2u;
    if (run(1024u, 2u) || operations != 32u || waits || drains != 1u) return 9;
    reset(); fail_wait = true;
    if (run(1024u, 2u) || operations != 32u || waits != 1u || drains != 1u) return 10;
    reset(); duration = 100.01f;
    if (run(1024u, 2u) || operations != 32u || waits != 1u || drains) return 11;
    reset(); duration = 100.0f;
    if (!run(1024u, 2u) || operations != 32u || waits != 1u) return 12;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-fla-sequence-") as temporary:
            executable = str(Path(temporary) / "sequence-check")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                 "-Werror", "-x", "c++", "-", "-o", executable],
                input=source, text=True, check=True, timeout=30,
            )
            subprocess.run([executable], check=True, timeout=5, capture_output=True)


if __name__ == "__main__":
    unittest.main()
