# Contributing

The repository is currently in release staging. Documentation, packaging,
licensing, and reproducibility fixes are welcome; runtime feature contributions
will become practical after the qualified source snapshot is published.

## Local checks

The current checks require Python 3.10 or newer and no third-party package:

```powershell
python -m unittest discover -s tests -p "test_*.py"
python tools/public_hygiene.py .
```

The same commands work in a POSIX shell.

## Evidence discipline

- Keep correctness, performance, startup, and API claims separate.
- Never claim real inference from a stub, smoke test, self-hash, estimate, or
  transport-only check.
- Performance evidence must identify the target host class, command, model
  reference, output/token result, source commit, and matching external
  correctness boundary.
- Never commit model weights, private endpoints, credentials, personal paths,
  or unrestricted raw prompt/output logs.
- Third-party code and binaries must retain their original license and
  attribution.

By submitting a contribution, you agree that it may be distributed under the
repository's Apache-2.0 license unless the submitted file clearly carries a
compatible upstream license.
