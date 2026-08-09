# Open-source readiness audit

Audit date: 2026-08-09

## Decision

The qualified Windows runtime is functionally mature enough to prepare for an
open-source release, but the private development repository is not safe or
reproducible enough to publish in place. The public repository therefore starts
as a transparent release-staging tree.

## What passed

| Area | Result | Basis |
|---|---|---|
| Product matrix | Pass | 12 accepted real-model Windows rows with external BF16 authority |
| q8192 performance | Pass | 3,852.909 ms TTFT and 2,126.186 tok/s prefill |
| Decode | Pass | 30.551 tok/s and 32.732 ms TPOT |
| Load time | Pass | 19,940.245 ms, below the 30-second bound |
| Context coverage | Pass | cold 8k-128k and prefix 16k-256k |
| Product behavior | Pass in qualification | prefix cache, streaming, 512-token decode, OpenAI-shaped HTTP lifecycle |
| Target build | Pass | clean-source Windows release build and 23/23 Rust/C ABI tests |
| Script regression suite | Pass | 147/147 Python tests |
| MMLU-Pro parity | Pass in qualification | candidate and authority both 7,486 / 12,032 |

Passing qualification does not by itself make a public release reproducible.

## Blocking issues for a supported source release

1. The development repository has no project-level license and governance
   files, and its two project crates do not declare package-license metadata.
2. A public-tree scan found private home-directory strings and private IPv4
   addresses in historical evidence and helper scripts. They must be sanitized
   before any history or snapshot is copied.
3. Compact OpenAI/MMLU acceptance files cited by documentation live in ignored
   qualification artifacts, so a clean clone cannot independently audit them.
4. Several Windows build entry points are tied to the qualification host and
   absolute deployment paths. They need public parameters and clean-machine
   documentation.
5. The full service requires provider DLLs and generated kernel directories
   that are not all reproducible from tracked inputs in a clean checkout.
6. Windows redistributable binaries and generated GPU objects need a complete
   provenance, redistribution, hash, and license inventory.
7. The advertised POSIX local smoke currently fails because a warning is
   promoted to an error, while a retained contract string still reports an old
   bootstrap-only state. Both need correction before source publication.

## Required release bundle

A supported public release must contain or unambiguously generate:

- the model-specific native core and Rust CLI/server source;
- every required provider and generated-kernel input;
- bounded MSVC/Rust/HIP build scripts without private-host assumptions;
- CPU-safe tests plus target-Windows ABI and real-model verification commands;
- compact, redacted, hash-bound qualification evidence;
- a machine-readable runtime and third-party dependency manifest;
- all required upstream license/notice texts; and
- checksummed source and portable Windows artifacts.

No model weights, private reference endpoint, credentials, or private raw
prompt/output corpus may enter the public repository or release assets.
