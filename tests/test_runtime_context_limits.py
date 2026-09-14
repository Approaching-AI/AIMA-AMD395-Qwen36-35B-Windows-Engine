"""Execute C request and provider-prefix bounds without model/GPU work."""
from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest

from test_attention_workspace import attention_capacity, function

ROOT = Path(__file__).resolve().parents[1]


class RuntimeContextLimitsTests(unittest.TestCase):
    def test_runtime_context_extends_metadata_without_losing_request_guards(self):
        runtime = (ROOT / "native/src/qrt.c").read_text()
        provider = (ROOT / "native/providers/whole_provider.cpp").read_text()
        constants = "\n".join(re.findall(
            r"^#define QRT_QWEN36_\w+_CONTEXT_TOKENS \d+u$", runtime, re.M))
        eligible = function(runtime, "static int qrt_qwen36_whole_provider_direct_entry_eligible(")
        prefix = function(provider, "constexpr bool qwen36_resident_session_prefix_supported(")
        source = r'''
#include "qrt.h"
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
struct qrt_engine {
    bool ready, manifest_loaded, qwen36_whole_provider_requested;
    size_t context_tokens;
    bool prefix_cache_state_attached;
};
''' + constants + attention_capacity() + eligible + prefix + r'''
int main() {
    static_assert(QRT_QWEN36_MAX_POSITION_EMBEDDINGS == 262144u);
    static_assert(QRT_QWEN36_MAX_PROMPT_TOKENS == 263168u);
    static_assert(QRT_QWEN36_MAX_REQUEST_CONTEXT_TOKENS == 263680u);
    static_assert(qrt_sm121_attention_capacity::kTokens == 264736u);
    qrt_engine engine{true,true,true,263680u,false};
    auto accepted=[&](size_t tokens,size_t output=512u) {
        return qrt_qwen36_whole_provider_direct_entry_eligible(&engine,tokens,output)!=0;
    };
    for(const char* option:{"QRT_QWEN36_Q16384_COLD_PROBE","QRT_QWEN36_Q32768_COLD_PROBE",
        "QRT_QWEN36_Q65536_COLD_PROBE","QRT_QWEN36_Q131072_COLD_PROBE"}) unsetenv(option);
    setenv("QRT_QWEN36_WHOLE_PROVIDER_ARBITRARY_CONTEXT","1",1);
    for(size_t count:{size_t(0),size_t(1),size_t(131072),size_t(132096),size_t(262144),
        size_t(262145),size_t(263168),size_t(263169),size_t(UINT32_MAX),SIZE_MAX}) {
        const bool valid=count>0u && count<=263168u;
        if(accepted(count)!=valid || qwen36_resident_session_prefix_supported(count)!=valid) return 1;
    }
    for(size_t output:{size_t(0),size_t(513),SIZE_MAX}) if(accepted(263168u,output)) return 2;
    if(!accepted(263168u,1u) || !accepted(263168u,512u)) return 3;
    engine.context_tokens=263167u;
    if(accepted(263168u)) return 4;
    engine.context_tokens=263680u;
    for(unsigned field=0;field<4u;++field) {
        engine.ready=field!=0u;engine.manifest_loaded=field!=1u;
        engine.qwen36_whole_provider_requested=field!=2u;engine.prefix_cache_state_attached=field==3u;
        if(accepted(263168u)) return 5;
    }
    engine={true,true,true,263680u,false};
    for(const char* flag:{"0","true","", "01"}) {
        setenv("QRT_QWEN36_WHOLE_PROVIDER_ARBITRARY_CONTEXT",flag,1);
        if(accepted(263168u)) return 6;
    }
    unsetenv("QRT_QWEN36_WHOLE_PROVIDER_ARBITRARY_CONTEXT");
    if(accepted(263168u) || qrt_qwen36_whole_provider_direct_entry_eligible(nullptr,263168u,512u)) return 7;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-runtime-context-") as temporary:
            executable = str(Path(temporary) / "limits")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-I", str(ROOT / "native/src"), "-x", "c++", "-", "-o", executable],
                input=source, text=True, check=True, timeout=30,
            )
            subprocess.run([executable], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
