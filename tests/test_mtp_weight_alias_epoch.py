"""Exercise the runtime's borrowed MTP lookups across storage replacement."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class MtpWeightAliasEpochTests(unittest.TestCase):
    def test_stale_aliases_are_never_returned_by_runtime_lookups(self):
        whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        actual = '\n'.join(function(whole, signature) for signature in (
            'bool qwen36_mtp_weight_aliases_current() {',
            'const uint16_t *find_whole_repeated_layer_fixed_weight(\n'
            '    const std::string &tensor_name,\n    uint64_t bytes\n) {',
            'const uint16_t *find_whole_repeated_layer_fixed_weight_by_kind(\n'
            '    unsigned int layer_index,\n    qrt_qwen36_tensor_kind_t tensor_kind,\n'
            '    uint64_t bytes\n) {',
            'bool whole_repeated_layer_fixed_weight_contains(const void *ptr) {'))
        source = r'''
#include <atomic>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
using qrt_qwen36_tensor_kind_t=unsigned;
struct WholeRepeatedLayerFixedWeightEntry {
 unsigned layer_index=0,tensor_kind=0;std::string tensor_name;
 uint16_t* device_weights=nullptr;uint64_t bytes=0;
};
std::vector<WholeRepeatedLayerFixedWeightEntry> g_whole_repeated_layer_fixed_weights;
thread_local std::vector<WholeRepeatedLayerFixedWeightEntry> g_qwen36_mtp_fixed_weight_aliases;
std::atomic<uint64_t> g_qwen36_mtp_weight_storage_epoch{1};
thread_local uint64_t g_qwen36_mtp_weight_alias_epoch=0;
''' + actual + r'''
int main(){
 uint16_t target=1,old_mtp=2,new_mtp=3;
 g_whole_repeated_layer_fixed_weights.push_back({3,1,"target",&target,2});
 g_qwen36_mtp_fixed_weight_aliases.push_back({3,2,"mtp",&old_mtp,2});
 assert(find_whole_repeated_layer_fixed_weight("target",2)==&target);
 assert(!find_whole_repeated_layer_fixed_weight("mtp",2));
 g_qwen36_mtp_weight_alias_epoch=g_qwen36_mtp_weight_storage_epoch.load();
 assert(find_whole_repeated_layer_fixed_weight("mtp",2)==&old_mtp);
 assert(find_whole_repeated_layer_fixed_weight_by_kind(3,2,2)==&old_mtp);
 assert(whole_repeated_layer_fixed_weight_contains(&old_mtp));
 g_qwen36_mtp_weight_storage_epoch.fetch_add(1);
 assert(!find_whole_repeated_layer_fixed_weight("mtp",2));
 assert(!find_whole_repeated_layer_fixed_weight_by_kind(3,2,2));
 assert(!whole_repeated_layer_fixed_weight_contains(&old_mtp));
 assert(find_whole_repeated_layer_fixed_weight("target",2)==&target);
 assert(find_whole_repeated_layer_fixed_weight_by_kind(3,1,2)==&target);
 assert(whole_repeated_layer_fixed_weight_contains(&target));
 g_qwen36_mtp_fixed_weight_aliases[0].device_weights=&new_mtp;
 g_qwen36_mtp_weight_alias_epoch=g_qwen36_mtp_weight_storage_epoch.load();
 assert(find_whole_repeated_layer_fixed_weight("mtp",2)==&new_mtp);
 assert(!whole_repeated_layer_fixed_weight_contains(&old_mtp));
 assert(whole_repeated_layer_fixed_weight_contains(&new_mtp));
 assert(!find_whole_repeated_layer_fixed_weight("mtp",4));
 assert(!find_whole_repeated_layer_fixed_weight_by_kind(3,2,4));
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'test.cpp'
            path.write_text(source)
            executable = directory / 'test'
            build = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', str(path), '-o', str(executable)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
