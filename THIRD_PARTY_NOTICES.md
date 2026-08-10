# Third-party notices

Project-authored files are Apache-2.0 unless a file states another license.
The following upstream material is present or used to build the release.

## AMD AITER

Selected fixed-shape MoE and fused-GDN algorithms and generators are adapted
from `amd-aiter 0.1.13`, under the MIT License. The license is preserved at
`third_party/licenses/AMD-AITER.txt` and next to the adapted provider sources.

## AMD Composable Kernel

The FMHA provider uses and adapts AMD Composable Kernel interfaces under the
MIT License. A Composable Kernel checkout is a source-build input and is not
vendored in full. The applicable text is
`third_party/licenses/AMD-Composable-Kernel.txt` and is also kept beside the
FMHA provider.

## Flash Linear Attention

Parts of the linear-attention/GDN implementation derive from Flash Linear
Attention concepts and source under the MIT License. See
`third_party/licenses/Flash-Linear-Attention.txt` and the adjacent provider
license file.

## gpu-simulator

The BF16 accumulator helper includes an attributed adaptation from
gpu-simulator under the MIT License. See
`third_party/licenses/gpu-simulator.txt` and the adjacent header notice.

## Triton and AMD ROCm

Triton 3.6 is used at build time to generate `gfx1151` code objects. AMD ROCm
HIP SDK 7.1, hipBLASLt, and rocBLAS are build/runtime dependencies. This
repository does not redistribute the Triton package, ROCm SDK, driver, or AMD
runtime DLLs; obtain them from their publishers under their respective terms.

## Rust crates

`Cargo.lock` pins the Rust dependency graph. Package versions, declared
licenses, and registry sources are listed in
`third_party/RUST_DEPENDENCIES.md`. The release archive carries this notice and
the project license; crate source is obtained from crates.io during builds.

## MMLU-Pro

The repository publishes sanitized evaluation results and harness-compatible
question identifiers, predictions, and hashes. It does not redistribute the
MMLU-Pro question text or answer choices. Users who reproduce the evaluation
must obtain the dataset independently and comply with its terms.

## Model weights

`Qwen3.6-35B-A3B` weights and tokenizer assets are not part of the repository
or release archive. Users must obtain them separately and comply with the model
license and acceptable-use requirements.
