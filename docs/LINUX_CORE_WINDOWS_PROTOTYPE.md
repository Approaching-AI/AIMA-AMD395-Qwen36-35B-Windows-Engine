# Linux native core Windows experiment

This standalone experiment ports the compute core declared by Linux release
`v1.5.1-native-vl.10`, source `ec9934446911fdf376da8eebcd83e7b137efbb7c`.
It is not enabled in the Windows product. The repaired `3560785` runtime
continues its complete 256k regression independently. There is no Windows
build, model correctness or performance qualification for this prototype yet.

The latest ordinary Windows q8192 control is 23272.0441 ms TTFT. Structural
projection and QK alternatives preserved component bits but increased their
measured execution times. This experiment instead evaluates the Linux release's
complete resident engine, hipBLASLt GEMMs, precompiled HIP/Triton kernels and
native decode. Linux measurements do not qualify Windows or replace GB10.

## Source and adaptation

`third_party/aima_linux/UPSTREAM.json` inventories 327 unchanged upstream files,
29,415,573 bytes, including the Apache license and preserved component notices.
`tools/import_linux_native_core.py --source-repo PATH --verify` compares every
file and the complete inventory against the pinned Git tree. Build preparation
also verifies the fixed inventory SHA and all imported file hashes.

`tools/prepare_linux_core_windows.py --out build/FRESH_DIRECTORY` produces seven
overlays without modifying the imported tree:

- Win32 shard reads retain 64-bit offsets, aligned direct I/O, buffered fallback,
  tensor scatter and GPU payload checksums. The Windows fallback uses a sequential
  access hint; it has no process-local equivalent of `POSIX_FADV_DONTNEED`.
- Windows dynamic-library loading maps to `LoadLibraryW` and `GetProcAddress`.
  The existing q8192 CK ABI is unchanged. Short-owner initialization uses the
  same explicit DLL and its existing dynamic square-attention entry point;
  rectangular support is not fabricated. This first probe admits only q8192.
- An output-only metric records `first.top1_logit`, the existing certified
  greedy LM-head result. It does not change selection or model arithmetic.
- UTF-8 model/report paths cross the loader ABI explicitly. The two upstream
  media enum-name functions are copied intact without media transport code.

The upstream registry generators validate kernel hashes, sizes, metadata and
schedule bindings. LLVM assembles their 72 unique GPU images into read-only
AMD64 COFF data. `tools/linux_core_coff.py` checks every symbol, image extent,
alignment and SHA, and rejects writable/executable or relocated image sections.
Embedded images total 1,319,512 bytes. A separate 51,056-byte qualified vision
image remains a file loaded by the unchanged engine. Its size is validated by
the upstream inventory; its SHA is checked again by the engine.

The prototype preserves the release's language/vision weight topology,
auxiliary prefill owners, visual warmup and resident prefix-cache allocation.
Those costs count toward loading. Its entry point accepts pretokenized text;
there is no image/video, tokenizer, HTTP or media-fetch frontend in this probe.
This is not a claim of Windows visual support.

## Bounded build and product observation

Run `tools/build_linux_core_windows.py --out build/FRESH_DIRECTORY` locally on
baiying through the existing guarded process owner. It requires a clean commit,
uses the installed ROCm 7.1 toolchain and hipBLASLt import library, imposes a
total deadline and per-process timeouts, preserves compiler output, checks the
Win32 host contract, and records source/generated/binary hashes. It never starts
a model. A CPU build or COFF check is not native inference evidence.
`scripts/baiying_linux_core_probe.ps1` binds the completed prior owner, frozen
source manifest and artifacts for separate build/product phases. It keeps the
existing global experiment mutex, memory checks and process cleanup. The build
has a 1500-second internal deadline inside its 1740-second owner; a product run
has a 600-second owner. The executable name matches the existing `qrt*` process
filter. The Win32 host fixture explicitly marks its file sparse before writing
beyond 4 GiB, avoiding a multi-gigabyte zero-filled allocation.

The resulting `qrt-linux-core-q8192-probe.exe` requires explicit model, prompt
u32 file, CK DLL, vision image and fresh load-report paths. It loads the real
model, performs one cold q8192 request, emits all 512 greedy outputs and streaming
callbacks, and records the first raw logit, first-callback TTFT and loading time.
There is no expected-token or oracle-activation input to this executable.

`tools/check_linux_core_q8192.py` binds completed guarded build/run records,
the frozen dispatch plan, source commits and artifacts to the unchanged
`contracts/gb10_cold_token_matrix_20260911_oracle.json` q8192/out512 case. It
checks actual prompt SHA, all 512 outputs and callbacks, and the first logit
within 0.125. It reports the 10000 ms boundary, retained 4187.415605 ms target
and 30000 ms loading bound without promoting a single run to product acceptance.
Broader contexts, prefix continuation, packaging, protocol and soak requirements
remain unchanged.

## Current verification

`python3 tools/test_linux_core_port.py --out build/FRESH_DIRECTORY` passes on
the macOS controller. It checks the pinned import, generated overlays, real COFF
assembly/image bytes, ASan/UBSan host I/O and input parsing, 25 synthetic observer
rejection controls and both exact logit-tolerance edges. The host I/O test reads
a Unicode-named sparse fixture beyond 4 GiB and checks truncation, EOF and invalid
offset handling. Its POSIX branch does not qualify the Win32 branch.

Read-only baiying inspection confirms the ROCm hipBLASLt library and required
headers are installed, and the real model's config/index SHA values equal the
imported layout. Native compilation and correctness-attached product measurement
are still required before this route can be selected.

The [preparation evidence](../benchmarks/correctness/linux-core-windows-preparation-20260922.json)
binds these checks to Windows experiment source
`0a57516e8c7c1dba55a077eba9d38b6155a0620e`, including the actual PowerShell parser
result on baiying. A controller is queued behind the repaired runtime's complete
256k owner. It has not dispatched compilation or inference. Its fixed blueprint
binds 335 build inputs, the original prompt, existing CK DLL and arithmetic
tables; only the completed prior-owner receipt is filled after cleanup. The
subsequent build and q8192/out512 phases must pass their own evidence checks.

The later [dependency inventory](../benchmarks/correctness/linux-core-windows-dependencies-20260922.json)
also records the installed gfx1151 hipBLASLt data and DLL imports. Its96 selected
data files total18,849,143bytes; the DLL adds6,012,312bytes. Actual relocated
execution is still needed to qualify that file set. A non-compiling driver query
confirms automatic MSVC, Windows SDK and linker discovery. The queued source,
environment and product boundary remain unchanged.
