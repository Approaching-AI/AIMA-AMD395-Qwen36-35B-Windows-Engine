from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PackagingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.build = (ROOT / "scripts/build-runtime.ps1").read_text(encoding="utf-8")
        cls.package = (ROOT / "scripts/package-runtime.ps1").read_text(
            encoding="utf-8"
        )
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.runtime = (ROOT / "engine/runtime.env").read_text(encoding="utf-8")

    def test_base_aot_is_copied_hashed_and_packaged(self) -> None:
        for fragment in (
            '$baseAotDir = Join-Path $OutDir ("aot\\" + $OffloadArch)',
            'Copy-Item -LiteralPath $file.FullName -Destination $baseAotDir',
            'base_aot = "aot/$OffloadArch"',
        ):
            self.assertIn(fragment, self.build)
        for fragment in ('"aot\\gfx1151"', 'prefix = "aot/gfx1151/"; count = 20'):
            self.assertIn(fragment, self.package)

    def test_windows_powershell_51_path_compatibility(self) -> None:
        for script in (self.build, self.package):
            self.assertIn("function Get-PortableRelativePath", script)
            self.assertNotIn("[IO.Path]::GetRelativePath", script)
            self.assertIn("[IO.Path]::GetFullPath", script)
            self.assertIn("[StringComparison]::OrdinalIgnoreCase", script)

    def test_quick_start_binds_base_aot_and_production_scopes_eval_policy(self) -> None:
        self.assertIn(
            "QRT_PREFILL_DESCRIPTOR_BATCH_Q1_MOE_TRITON_0626_MODULE_DIR=$rt\\aot\\gfx1151",
            self.readme,
        )
        self.assertIn(
            "QRT_QWEN36_EXACT_ARBITRARY_LM_HEAD_BF16_WINDOW_HIGH_ID=1",
            self.runtime,
        )
        self.assertIn(
            "QRT_QWEN36_EXACT_ARBITRARY_LM_HEAD_BF16_WINDOW_HIGH_ID_GLOBAL=0",
            self.runtime,
        )


if __name__ == "__main__":
    unittest.main()
