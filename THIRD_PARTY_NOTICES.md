# Third-party notices

The repository is licensed under Apache License 2.0 except where a file carries
a different SPDX identifier.

## Initial staging revision

This revision contains release documentation and a Python-standard-library
hygiene checker. It does not contain model weights, generated GPU objects,
AMD redistributable binaries, Rust dependencies, or copied third-party source.

## Qualified runtime inventory pending publication

The qualified private runtime uses or derives material from the following
upstream surfaces. They are listed here for release planning; this staging
revision does not distribute them:

- AMD HIP SDK 7.1, hipBLASLt, rocBLAS, and their Windows runtime dependencies;
- Rust crates pinned by `Cargo.lock` for the resident HTTP service and CLI;
- AMD Composable Kernel, AMD AITER-derived algorithms, and selected
  Triton-generated `gfx1151` code objects;
- diagnostic-only adaptations from Flash Linear Attention and gpu-simulator;
- the pinned MMLU-Pro dataset and official evaluation-harness semantics.

Before any of those files or binaries are published, the release must include
their exact versions, provenance, redistribution classification, complete
license texts, and the notices required by each upstream project.

## Model weights not bundled

Qwen3.6-35B-A3B model weights and tokenizer assets are not part of this
repository. Users must obtain them independently and comply with the model's
license and acceptable-use terms.
