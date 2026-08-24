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
        cls.smooth_tail_build = (
            ROOT / "scripts/baiying_build_smooth_tail_moe.ps1"
        ).read_text(encoding="utf-8")
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.runtime = (ROOT / "engine/runtime.env").read_text(encoding="utf-8")
        cls.product_cli_build = (
            ROOT / "scripts/build-product-cli.ps1"
        ).read_text(encoding="utf-8")

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

    def test_ck_snapshot_uses_tree_fingerprint_without_requiring_git(self) -> None:
        self.assertIn('$ckGitMetadata = Join-Path $CkRoot ".git"', self.build)
        self.assertIn("if (Test-Path -LiteralPath $ckGitMetadata)", self.build)
        self.assertIn("composable_kernel_tree_sha256 = $ckTreeSha256", self.build)
        self.assertIn("composable_kernel_source_file_count", self.build)

    def test_release_archive_and_checksum_are_cross_platform(self) -> None:
        self.assertNotIn("Compress-Archive", self.package)
        self.assertIn("System.IO.Compression.ZipArchive", self.package)
        self.assertIn('$entryName = "$baseName/$relative"', self.package)
        self.assertIn("[IO.File]::WriteAllText", self.package)
        self.assertIn('"$sha256  $baseName.zip`n"', self.package)

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

    def test_smooth_tail_providers_are_built_and_packaged_for_every_tile(self) -> None:
        self.assertIn("baiying_build_smooth_tail_moe.ps1", self.build)
        self.assertIn('smooth_tail_moe_root = "smooth-tail"', self.build)
        self.assertIn('prefix = "smooth-tail/q$tokens/"', self.package)
        self.assertIn("count = 6", self.package)
        self.assertIn(
            "QRT_QWEN36_SMOOTH_TAIL_BOUNDED_TRANSACTIONS=1",
            self.runtime,
        )
        self.assertIn(
            "QRT_QWEN36_SMOOTH_TAIL_PARALLEL_TRANSACTIONS=1",
            self.runtime,
        )
        for tokens in (32, 64, 128, 256, 512, 1024, 2048, 4096):
            self.assertIn(str(tokens), self.smooth_tail_build)
        for fragment in (
            "Invoke-BoundedProcess",
            '"-DQRT_TRITON_MOE_NATIVE_WMMA_GATE=1"',
            '"-DQRT_TRITON_MOE_NATIVE_WMMA_DOWN=1"',
            '"-DQRT_TRITON_MOE_TRANSPOSED_ROUTER=1"',
            '"-DQRT_TRITON_MOE_FULL_V3_FUSED_COMBINE=1"',
            "provider_backend_mask = 15",
        ):
            self.assertIn(fragment, self.smooth_tail_build)

    def test_q8192_release_build_uses_the_qualified_native_shared_route(
        self,
    ) -> None:
        for fragment in (
            "-BlockM 64 -GroupM 1",
            "-RoutedProjectionDebug 0 -BatchedHawkeye 1 -ExactShared 0",
            "-ConditionalExactGate 1 -SortedConditionalExactGate 1",
            "-RowMajorSortedConditionalExactGate 1 -ConditionalExactGateRows 256",
            "-ConditionalExactDown 1 -ConditionalExactDownRows 4",
            "-NativeFusedRouteLayout 1 -NativeWmmaLdsB 1",
            "-NativeWmmaLdsBM64LoadThreads 192",
            "-NativeWmmaLdsBM64FusedOverflow32 1",
            "-NativeWmmaLosslessPalette 1 -NativeWmmaLosslessRowPalette 1",
            "-NativeWmmaKStage 32",
        ):
            self.assertIn(fragment, self.build)
        for name, value in (
            ("QRT_QWEN36_Q8192_ROUTER_HIPBLASLT_BF16", "1"),
            ("QRT_QWEN36_CUDA_VLLM_ROUTER_HAWKEYE_MIDPOINT_RADIUS", "173"),
            ("QRT_QWEN36_CUDA_VLLM_SHARED_HAWKEYE_MIDPOINT_RADIUS", "0"),
            ("QRT_QWEN36_Q8192_VLLM_BF16_RESIDUAL_CARRIER", "1"),
            ("QRT_QWEN36_EXACT_ARBITRARY_VLLM_SPLIT_VARIANCE", "1"),
            ("QRT_QWEN36_Q8192_VLLM_SORTED_BF16_ROUTE_SUM", "1"),
            ("QRT_QWEN36_CUDA_VLLM_MOE_HAWKEYE_MIDPOINT_RADIUS", "0"),
            ("QRT_QWEN36_CUDA_VLLM_MOE_UP_HAWKEYE_MIDPOINT_RADIUS", "0"),
            (
                "QRT_QWEN36_CUDA_VLLM_ROUTED_DOWN_CONTRIBUTION_HAWKEYE_MIDPOINT_RADIUS",
                "0",
            ),
            (
                "QRT_QWEN36_CUDA_VLLM_ROUTED_GATE_HAWKEYE_LOW_EXPONENT_THRESHOLD",
                "0",
            ),
            (
                "QRT_QWEN36_CUDA_VLLM_ROUTED_UP_HAWKEYE_LOW_EXPONENT_THRESHOLD",
                "0",
            ),
            (
                "QRT_QWEN36_CUDA_VLLM_ROUTED_DOWN_HAWKEYE_LOW_EXPONENT_THRESHOLD",
                "0",
            ),
        ):
            self.assertIn(f"{name}={value}", self.runtime)

    def test_product_cli_has_a_reproducible_large_stack_build(self) -> None:
        for fragment in (
            "build-product-cli.ps1",
            '(Join-Path $productCliDir "qrt-product.exe")',
            'product_cli = "product-cli"',
        ):
            self.assertIn(fragment, self.build)
        self.assertIn('"product-cli\\qrt-product.exe"', self.package)
        for fragment in (
            "StackReserveBytes = 268435456",
            '"/STACK:$StackReserveBytes"',
            "WaitForExit($TimeoutSeconds * 1000)",
            "stack_reserve_bytes = $StackReserveBytes",
        ):
            self.assertIn(fragment, self.product_cli_build)


if __name__ == "__main__":
    unittest.main()
