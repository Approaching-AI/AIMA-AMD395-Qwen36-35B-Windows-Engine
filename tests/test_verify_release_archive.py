"""Exercise complete archive ownership and pinned bytes without running a model."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_release_archive", ROOT / "scripts/verify_release_archive.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def inventory(files):
    return [dict(path=name, bytes=len(data), sha256=digest(data)) for name, data in sorted(files.items())]


class ReleaseArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "candidate.zip"
        self.root = verify.PROJECT + "-vtest"
        self.commit = "a" * 40
        self.runtime_files = {name: (name + " fixture\n").encode() for name in verify.REQUIRED_RUNTIME}
        self.runtime_files["ck-fmha/current.dll"] = b"mock DLL, never executed"
        self.runtime = dict(schema_version=1, dirty_tree=False, offload_arch="gfx1151",
                            repo_commit=self.commit, artifacts=inventory(self.runtime_files))

    def package(self, *, runtime=None, release_change=None, actual_change=None, additions=()):
        files = copy.deepcopy(self.runtime_files)
        files[verify.RUNTIME_MANIFEST] = encode(self.runtime if runtime is None else runtime)
        files["README.md"] = b"Synthetic inventory fixture; no inference qualification.\n"
        release = dict(schema_version=1, project=verify.PROJECT, version="test", target=verify.TARGET,
                       source_commit=self.commit, files=inventory(files))
        if release_change:
            release_change(release)
        files[verify.RELEASE_MANIFEST] = encode(release)
        if actual_change:
            actual_change(files)
        with zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(self.root + "/" + name, data)
            for name, data in additions:
                archive.writestr(name, data)
        return digest(self.path.read_bytes())

    def check_rejected(self, **kwargs):
        expected = self.package(**kwargs)
        with self.assertRaises(verify.InventoryError):
            verify.verify_archive(self.path, expected)

    def test_complete_archive_and_cli_sidecar(self):
        expected = self.package()
        runtime_digest = digest(encode(self.runtime))
        result = verify.verify_archive(self.path, expected, expected_runtime_manifest_sha256=runtime_digest)
        self.assertEqual(result["runtime_artifacts"], len(self.runtime_files))
        self.assertEqual(result["release_files"], len(self.runtime_files) + 2)
        self.assertEqual(result["archive_files"], len(self.runtime_files) + 3)
        self.assertTrue(result["pinned_runtime_manifest_checked"])
        self.assertFalse(result["runtime_executed"])
        self.assertFalse(result["inference_acceptance"])
        self.assertFalse(result["release_qualified"])
        sidecar = self.path.with_suffix(".zip.sha256")
        sidecar.write_text(expected + "  " + self.path.name + "\n")
        run = subprocess.run([sys.executable, str(ROOT / "scripts/verify_release_archive.py"),
                              str(self.path), "--checksum-file", str(sidecar),
                              "--runtime-manifest-sha256", runtime_digest],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), result)

    def test_pinned_archive_and_runtime_manifest(self):
        expected = self.package()
        with self.assertRaisesRegex(verify.InventoryError, "archive SHA256 mismatch"):
            verify.verify_archive(self.path, "0" * 64)
        with self.assertRaisesRegex(verify.InventoryError, "pinned runtime manifest"):
            verify.verify_archive(self.path, expected, expected_runtime_manifest_sha256="0" * 64)
        sidecar = self.path.with_suffix(".sha256")
        for text in (expected + "  another.zip\n", expected + "  candidate.zip\nextra\n"):
            sidecar.write_text(text)
            with self.assertRaises(verify.InventoryError):
                verify.checksum_sidecar(sidecar, self.path)

    def test_historical_profile_requires_the_fixed_published_archive(self):
        expected = self.package()
        with self.assertRaisesRegex(verify.InventoryError, "exact published v1.0.1"):
            verify.verify_archive(self.path, expected, profile="published-v1.0.1")
        with self.assertRaisesRegex(verify.InventoryError, "unknown inventory profile"):
            verify.verify_archive(self.path, expected, profile="unrecognized")

    def test_current_profile_requires_the_separate_product_cli(self):
        self.runtime_files.pop("product-cli/qrt-product.exe")
        self.runtime["artifacts"] = inventory(self.runtime_files)
        expected = self.package()
        with self.assertRaisesRegex(verify.InventoryError, "required runtime artifact"):
            verify.verify_archive(self.path, expected)

    def test_changed_missing_and_unlisted_files(self):
        for change in (
            lambda files: files.__setitem__("runtime.env", b"changed"),
            lambda files: files.pop("runtime.env"),
            lambda files: files.__setitem__("unlisted.dll", b"extra"),
        ):
            with self.subTest(change=change):
                self.check_rejected(actual_change=change)

    def test_runtime_inventory_cannot_disagree_with_release_inventory(self):
        for key, value in (("sha256", "0" * 64), ("bytes", 0), ("path", "not-in-release.dll")):
            runtime = copy.deepcopy(self.runtime)
            runtime["artifacts"][0][key] = value
            with self.subTest(key=key):
                self.check_rejected(runtime=runtime)
        runtime = copy.deepcopy(self.runtime)
        runtime["artifacts"] = [row for row in runtime["artifacts"] if row["path"] != "runtime.env"]
        self.check_rejected(runtime=runtime)

    def test_source_target_and_dirty_tree(self):
        for key, value in (("repo_commit", "b" * 40), ("offload_arch", "gfx0000"), ("dirty_tree", True)):
            runtime = copy.deepcopy(self.runtime)
            runtime[key] = value
            with self.subTest(key=key):
                self.check_rejected(runtime=runtime)
        self.check_rejected(release_change=lambda release: release.__setitem__("version", "different"))
        self.check_rejected(release_change=lambda release: release.__setitem__("target", "linux"))

    def test_declared_component_must_identify_the_packaged_artifact(self):
        runtime = copy.deepcopy(self.runtime)
        name = "whole-provider/qrt_qwen36_whole_provider.dll"
        artifact = dict(sha256=digest(self.runtime_files[name]), bytes=len(self.runtime_files[name]))
        runtime["source_component_builds"] = {"whole": dict(commit=self.commit, package_path=name, artifact=artifact)}
        runtime["component_commits"] = {"whole": self.commit}
        expected = self.package(runtime=runtime)
        self.assertEqual(verify.verify_archive(self.path, expected)["declared_component_artifacts_checked"], 1)
        for field, value in (("sha256", "0" * 64), ("bytes", 0)):
            changed = copy.deepcopy(runtime)
            changed["source_component_builds"]["whole"]["artifact"][field] = value
            self.check_rejected(runtime=changed)
        changed = copy.deepcopy(runtime)
        changed["component_commits"]["whole"] = "b" * 40
        self.check_rejected(runtime=changed)

    def test_duplicate_paths_and_file_directory_collisions(self):
        for name in (self.root + "/runtime.env", self.root + "/RUNTIME.ENV",
                     self.root + "/engine/qrt.exe/child", self.root + "/engine/QRT.EXE/child"):
            with self.subTest(name=name), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.check_rejected(additions=[(name, b"duplicate or child")])
        self.check_rejected(release_change=lambda release: release["files"].append(dict(release["files"][0])))
        runtime = copy.deepcopy(self.runtime)
        duplicate = dict(runtime["artifacts"][0]); duplicate["path"] = duplicate["path"].upper()
        runtime["artifacts"].append(duplicate)
        self.check_rejected(runtime=runtime)

    def test_nonportable_entries_and_symbolic_links(self):
        paths = ("../outside", "/absolute", "C:/drive", "other/outside", self.root + "/CON.txt",
                 self.root + "/space /x", self.root + "/dot./x", self.root + "/double//x",
                 self.root + "/back\\slash", self.root + "/stream:ads", self.root + "/./x")
        for name in paths:
            with self.subTest(name=name):
                self.check_rejected(additions=[(name, b"invalid")])
        link = zipfile.ZipInfo(self.root + "/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.check_rejected(additions=[(link, b"runtime.env")])

    def test_duplicate_json_keys_and_inventory_value_types(self):
        def duplicate(files):
            data = files[verify.RELEASE_MANIFEST]
            files[verify.RELEASE_MANIFEST] = data.replace(b'{', b'{"version":"test",', 1)
        self.check_rejected(actual_change=duplicate)
        for key, value in (("bytes", True), ("bytes", -1), ("sha256", "z" * 64), ("path", "../bad")):
            def change(release, key=key, value=value):
                release["files"][0][key] = value
            with self.subTest(key=key, value=value):
                self.check_rejected(release_change=change)

    def test_size_bound_and_successful_explicit_directory_entries(self):
        expected = self.package(additions=[(self.root + "/", b""), (self.root + "/empty/", b"")])
        self.assertTrue(verify.verify_archive(self.path, expected)["windows_paths_pass"])
        with self.assertRaisesRegex(verify.InventoryError, "size limit"):
            verify.verify_archive(self.path, expected, maximum_bytes=10)


if __name__ == "__main__":
    unittest.main()
