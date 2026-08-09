# AIMA AMD395 Qwen3.6 35B Windows Engine

Public release staging for a TileRT-style, model-specific native Windows
inference engine for `Qwen3.6-35B-A3B-BF16` on AMD Ryzen AI Max+ 395
(`gfx1151`). The runtime is designed for batch size 1 and uses a C core with
thin Rust tooling.

> **Release status:** the target runtime has passed its private qualification
> matrix, but the source and binary package are not published yet. This first
> public revision contains the release contract, governance, and publication
> readiness record. It is not an installable engine release.

[中文说明](README.zh-CN.md)

## Qualified target

The latest retained qualification snapshot was executed as a native Windows
process on the project AMD395 host, using the real model and a separate BF16
correctness authority. The q8192 product row reported:

| Metric | Result | Required bound |
|---|---:|---:|
| TTFT | 3,852.909 ms | <= 4,187.416 ms |
| Prefill throughput | 2,126.186 tok/s | >= 1,506.407 tok/s |
| Decode throughput | 30.551 tok/s | >= 28.168 tok/s |
| TPOT | 32.732 ms | <= 35.502 ms |
| Model + engine load | 19,940.245 ms | <= 30,000 ms |

The complete qualification matrix contains 12 accepted rows covering:

- the q8192 retained peak;
- cold 8k through 128k contexts;
- shared-prefix reuse from 16k through 256k;
- 512-token decode and SSE streaming; and
- first-token correctness anchored to the external BF16 authority.

These numbers are qualification facts, not a promise that this staging tree
can reproduce them. Public reproduction requires the source, generated kernel
inventory, redistributable runtime manifest, compact evidence package, and
clean-machine build instructions that are still being curated.

## Publication gates

- [x] Native Windows real-model product qualification completed.
- [x] External first-token correctness authority attached to the product rows.
- [x] Windows public repository and Apache-2.0 governance created.
- [ ] Remove private deployment paths and addresses from the release history.
- [ ] Publish compact OpenAI API, MMLU-Pro, and product-matrix evidence.
- [ ] Make every required provider reproducible from a clean checkout.
- [ ] Complete the Windows redistributable and third-party license inventory.
- [ ] Publish a checksummed source tag and portable binary package.

The release is ready to be **announced as work in release preparation**, but it
is not ready to be presented as a reproducible open-source engine package.
See [the readiness audit](docs/OPEN_SOURCE_READINESS.md) for the exact boundary.

## Intended public layout

The repository follows the same release organization as the companion Linux
engine while keeping the implementation Windows-native:

- `engine/`: public runtime API and orchestration;
- `native/`: C/C++/HIP providers and generated `gfx1151` kernels;
- `scripts/`: MSVC, Rust, HIP, packaging, and verification entry points;
- `benchmarks/`: compact correctness-attached product evidence;
- `packaging/`: portable Windows archive and license assembly;
- `tests/`: CPU-safe and Windows ABI regression checks; and
- `third_party/licenses/`: verbatim notices for redistributed material.

## Scope

This project is deliberately model- and device-specific. It is not a generic
graph runtime and does not use smoke tests, transport checks, self-hashes, or
proxy benchmarks as proof of real inference. Model weights are never bundled;
users must obtain them separately and comply with their license.

## License

Project-authored material is licensed under the Apache License 2.0. Files that
carry another license or attribution remain under their stated terms. See
[LICENSE](LICENSE), [NOTICE](NOTICE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
