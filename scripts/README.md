# Build and verification scripts

`build-runtime.ps1` is the supported full Windows build. The
`baiying_build_*` names are retained for provenance compatibility, but their
public versions use the current machine name and parameterized toolchain/output
paths; they are not restricted to a private host.

`verify_qrt_openai_server.py` tests the real resident service, including SSE,
tools, FIFO pressure, arbitrary lengths, context limits, prefix-cache logs, and
optional external reference files. Capture/audit scripts produce compact
runtime and shutdown records. MMLU scripts preserve the formal evaluation and
projection-parity workflow.
