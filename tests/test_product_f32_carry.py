"""Checked K16 product-domain carry, compared with the independent wide sum."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / 'native/providers/moe_accumulator/sm121_product_f32_carry.h'


class ProductF32CarryTests(unittest.TestCase):
    def test_ordered_boundaries_rejection_and_guard_controls(self):
        with tempfile.TemporaryDirectory(prefix='qrt-product-carry-') as directory:
            work = Path(directory)
            common = [os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                      '-ffp-contract=off', '-fsanitize=address,undefined,float-cast-overflow',
                      '-fno-sanitize-recover=all', '-I', str(ROOT), '-I', str(HEADER.parent),
                      str(ROOT / 'tests/native/product_f32_carry_host.cpp')]
            executable = work / 'check'
            build = subprocess.run(common + ['-o', str(executable)], capture_output=True, text=True, timeout=40)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=55)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads(run.stdout)
            self.assertGreater(result['accepted_dots'], 100000)
            self.assertGreater(result['declined_dots'], 100000)
            print(run.stdout.strip())
            text = HEADER.read_text()
            for name, old, new in [
                ('subnormal', 'subnormal_maximum <= normal_maximum - 27', 'subnormal_maximum <= normal_maximum - 25'),
                ('floor', 'normal_maximum >= -64', 'normal_maximum >= -126'),
            ]:
                self.assertEqual(text.count(old), 1)
                control = work / f'{name}.h'
                control.write_text(text.replace(old, new))
                binary = work / name
                build = subprocess.run(common + [f'-DQRT_PRODUCT_CARRY_TEST_HEADER="{control}"', '-o', str(binary)],
                                       capture_output=True, text=True, timeout=40)
                self.assertEqual(build.returncode, 0, build.stderr)
                run = subprocess.run([str(binary), f'--probe-{name}'], capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, 3, run.stdout + run.stderr)
                self.assertIn(f'negative_control_detected={name}', run.stdout)
                print(run.stdout.strip())


if __name__ == '__main__':
    unittest.main()
