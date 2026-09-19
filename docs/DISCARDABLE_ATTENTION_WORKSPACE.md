# Long-attention temporary memory candidate

The original 262144-token owner test stopped at the unchanged 8 GiB available
physical-memory guard after 20 of 32 chunks. That record produced no output
tokens and does not establish numerical failure or a memory leak. A separate
whole-provider KV reservation candidate is being measured against the same
original GB10 case.

This CK candidate reduces avoidable temporary overlap. Long-attention scores,
transposed V, decoded Q/K and suffix Q/K/V staging contain disposable values
that their next producers refresh. Their existing mutexes and completed-stream
boundaries permit releasing the old allocation before allocating a larger one.
Session KV and model tensors do not use this operation. Failed release retains
the owner; failed replacement leaves an empty owner that a subsequent call can
rebuild. Producer and consumer errors keep their existing stream drains.

When the long pipeline needs its independent, larger score allocation, it also
skips initial allocation of the unused ordinary matrix slab. A subsequent call
to the ordinary path can allocate that slab when needed. Previously allocated
ordinary slabs remain reusable. Layouts, capacity rounding, arithmetic,
dispatch flags, limits and host memory guards are unchanged.

Four targeted local tests pass. They exercise the actual ownership helper
under an allocation budget that cannot hold both old and new buffers, release
and allocation failures, retry, actual provider and suffix function bodies,
partial submissions, complete staging refresh, and long-layout bounds. The
provider test also checks unused-slab absence and long-to-ordinary-to-long
transitions at 64k, 128k and 256k starts. The helper and layout tests run with
ASan/UBSan; suffix staging uses UBSan. Mock HIP calls establish ownership and
wiring only. Initial test compilation and mock-counter failures are preserved
in local evidence; their corrections do not change runtime code.

Windows compilation, original-token q8192 and long-context regressions, and a
memory comparison on the real model remain pending. No measured memory or
latency improvement is claimed, and this candidate does not qualify a release.

Source `c5809d7` has 64 local compilation inputs. Relative to qualified CK
`ea6faff`, only the provider implementation changes and the temporary-owner
header is added. All numerical dependencies match. The prepared build checks
those inputs plus its two scripts, with a 240-second native deadline. Its
command file passes the actual Windows PowerShell parser. Local results,
including failed test revisions, and the unexecuted native preparation are in
[`pending-native-candidates-20260919.json`](../benchmarks/correctness/pending-native-candidates-20260919.json).
