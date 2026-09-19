# Verify a portable archive's file inventories

Run `scripts/verify_release_archive.py` from a source checkout with Python
3.10 or later. It uses the standard library, reads the ZIP directly and
does not extract or execute its contents. The portable inference runtime
does not require Python.

```powershell
python scripts/verify_release_archive.py engine.zip --checksum-file engine.zip.sha256
```

The sidecar must contain one SHA256 and this exact archive filename. An
explicit `--sha256` can supply the expected archive digest instead. Pass
`--runtime-manifest-sha256` from the accepted native run to bind the check to
that precise runtime inventory as well.

The verifier checks the complete release file set, each file's CRC, SHA256
and byte count, and every runtime artifact's matching release entry. It
checks source/target consistency and the packaged DLLs named by component
records. Duplicate paths, Windows filename aliases, file/directory
collisions, traversal, symbolic links, encrypted entries and ambiguous JSON
inventories fail. The archive is hashed again after reading all contents.
The default compressed and total uncompressed size limit is 8 GiB;
`--maximum-bytes` sets a different explicit inspection limit.

The JSON result distinguishes `release_files` (files listed by
`FILE-SHA256SUMS.json`) from `archive_files` (including that manifest itself).
Component build records are declarations whose packaged artifact hashes
are checked; this tool does not rerun their builds. Successful inventory
verification does not establish native execution, relocation, GB10 output,
performance, soak or release acceptance. Those checks must identify the
same archive or explicitly bound unchanged components.

The [public path redaction record](../benchmarks/correctness/public-home-path-redaction-20260919.json)
separately binds sanitized diagnostic renderings to their original Git
objects. It contains the complete verifier program and before/after hashes.
Only home-directory prefixes and explicit public file references change;
canonical runtime hashes, tokens, numerical results and timings remain intact.
