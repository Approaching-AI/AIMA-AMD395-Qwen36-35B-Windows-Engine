# Build and verification scripts

`build-runtime.ps1` is the supported full Windows build. The
`baiying_build_*` names are retained for provenance compatibility, but their
public versions use the current machine name and parameterized toolchain/output
paths; they are not restricted to a private host.

`build-product-cli.ps1` builds the direct real-token acceptance executable with
the required 256 MiB Windows stack reserve. The full runtime build and release
archive include it under `product-cli/` so q8192 GB10 and prefix gates can be
reproduced from the packaged artifact rather than an ad-hoc binary.

`verify_qrt_openai_server.py` tests the real resident service, including SSE,
tools, FIFO pressure, arbitrary lengths, context limits, prefix-cache logs, and
optional external reference files. Capture/audit scripts produce compact
runtime and shutdown records. MMLU scripts preserve the formal evaluation and
projection-parity workflow.

`verify_prefill_length_smoothness.py` runs cold, GB10-paired `center-1`,
`center`, and `center+1` triplets from q4096 through q16384 by default. It
requires every local TTFT triplet to stay within `1.10x` and `500 ms`, and the
wide median-throughput envelope to stay within `1.30x`; controlled content and
globally unique first tokens isolate length effects from prefix reuse.

`verify_prefill_random_length_plateaus.py` is the stricter arbitrary-length
gate. It samples at least six random lengths in every configured interval,
brackets each sample with two cold measurements at that interval's upper
anchor, and queries the AMD and GB10 services concurrently with identical real
token IDs. A sample contributes performance evidence only when all three token
outputs match GB10 and the two anchor speeds stay within `1.03x`. Every eligible
random/anchor throughput ratio must remain within `0.97..1.03`; the q8192
anchor must also retain the declared TTFT and throughput bounds. The verifier
does not permit command-line options to relax those retained limits.

After the run, pass its JSON and the matching service log to
`verify_prefill_random_length_route_log.py`. That second gate verifies that
every measured request used the qualified attention, GDN, selected-MoE, and
terminal routes without fallback or prefix-cache contamination. Both reports
must pass before arbitrary-length product performance is accepted.
