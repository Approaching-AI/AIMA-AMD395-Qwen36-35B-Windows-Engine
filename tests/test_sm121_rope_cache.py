import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Sm121RopeCacheTests(unittest.TestCase):
    def run_case(self, wrong_hash=False, wrong_geometry=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parameters = dict(mrope_interleaved=True, mrope_section=[11, 11, 10],
                              partial_rotary_factor=0.25, rope_theta=10000000, rope_type="default")
            config = root / "config.json"
            config.write_text(json.dumps(dict(text_config=dict(head_dim=128 if wrong_geometry else 256,
                                         max_position_embeddings=262144, rope_parameters=parameters))))
            source = root / "source.json"
            source.write_text('{}')
            digest = hashlib.sha256(config.read_bytes()).hexdigest()
            output = root / "capture"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/capture_sm121_rope_cache.py"),
                                     "--source-commit", "a" * 40, "--model-config", str(config),
                                     "--expected-config-sha256", "0" * 64 if wrong_hash else digest,
                                     "--source-manifest", str(source), "--output-dir", str(output)],
                                    text=True, capture_output=True, timeout=10)
            if wrong_hash or wrong_geometry:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads(result.stdout)
                self.assertFalse(record['completed'])
                self.assertFalse(record['model_weights_loaded'])
                self.assertFalse(record['prompt_inputs'])
                self.assertFalse(record['inference_acceptance'])
                self.assertFalse((output / 'sm121-rope-bf16.bin').exists())
                self.assertEqual(record['model_config_sha256'], digest)

    def test_dry_run_writes_no_cache(self):
        self.run_case()

    def test_changed_model_config_fails_before_execution(self):
        self.run_case(wrong_hash=True)

    def test_geometry_mismatch_fails_before_execution(self):
        self.run_case(wrong_geometry=True)
