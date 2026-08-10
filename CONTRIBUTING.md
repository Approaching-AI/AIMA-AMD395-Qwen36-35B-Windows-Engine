# Contributing

Contributions to runtime correctness, Windows packaging, API compatibility,
tests, and performance are welcome. Changes should preserve the model/device
scope and attach evidence proportional to their impact.

## Local checks

Python 3.10+, a C11 compiler, and Rust are sufficient for CPU-safe checks:

```shell
make check
```

Run `cargo fmt --all -- --check` before submitting Rust changes. Provider or
kernel changes additionally require the relevant Windows `gfx1151` build/smoke
script. Performance claims require a real-model run with the matching external
BF16 correctness boundary; stubs, transport checks, estimates, and self-hashes
are diagnostics only.

## Data and licensing

- Never commit model weights, credentials, private endpoints, personal paths,
  or unrestricted prompt/output logs.
- Keep evaluation rows sanitized and identify questions by stable IDs/hashes.
- Preserve compatible upstream SPDX headers, copyright, and license texts.
- By submitting project-authored work, you agree that it may be distributed
  under Apache-2.0.
