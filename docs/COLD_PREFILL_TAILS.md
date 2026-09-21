# Cold prefill final chunks

The cold chunk coordinator previously accepted only multiples of 8192 and
an optional 1024-token tail. This rejected the original q8193 and
q262140/q262142/q262143 native MTP cases before their correctness boundary
could run. It now retains the true final extent from 1 through 8192 inputs.
Every earlier chunk still contains exactly 8192 original tokens; the prompt
is never padded or substituted.

The suffix convolution ring also assumed at least four new inputs. A final
chunk of one, two or three inputs would underflow its source row. The actual
kernel now replaces only the new ring slots and preserves the preceding
prefix slots. Both FP32 and BF16 ring layouts follow this rule.

Finally, the C entry point previously split prompts of 8193 through 9216
tokens into an 8192-token seed and a prefix/decode suffix before the whole
provider saw the request. With native MTP and cold chunking both enabled,
that shortcut now delegates the complete prompt to the chunk owner. This
preserves the original full-prompt shifted inputs and checkpoint provenance.
Other C entry modes retain their existing behavior. Native MTP remains opt-in.

These changes do not expand the public prefix-cache suffix interface or
permit native MTP to start at or beyond its 262144-token drafter limit.
The existing standalone native route already admits q8191; the q7169/q8192
restriction belongs to a separate diagnostic capture mode.

## Local verification

The actual coordinator passes 26 ordinary prompt lengths and 13 native MTP
lengths, including single-token tails and the three original retirement
prompts. Failure checks cover short-tail publication, known and unknown
completion, checkpoint saves and cleanup. The extracted production ring
kernel passes 18 extents, four prefix residues and both formats: 144 shapes
under ASan/UBSan, including unchanged slots and allocation guards. The C
bridge checks the two route flags independently, cancellation, callback
clocks, metadata and ownership.

The following focused runs pass, totaling 23 distinct Python tests:

```sh
PYTHONPATH=tests python3.12 -m unittest test_prefill_chunks test_prefix_batch_suffix test_bounded_prefill_suffix
PYTHONPATH=tests python3.12 -m unittest test_prefix_batch_suffix.PrefixBatchSuffixTests.test_actual_ring_update_keeps_unreplaced_prefix_slots test_attention_suffix test_prefix_fla_suffix test_mtp_chunked_prefill_capture
PYTHONPATH=tests python3.12 -m unittest test_mtp_native_prefill_integration test_native_mtp_decode test_mtp_drafter
make c-smoke
python3.12 scripts/test_q16_transaction_contract.py
```

The C smoke and all seven transaction cases also pass. Three separately
restored old implementations compile successfully and fail their respective
new checks: ring underflow, coordinator tail rejection and C shortcut
misrouting. The initial evidence writer failed to serialize subprocess
bytes after executing these controls; its partial record and fixtures are
preserved, and the corrected writer records all expected outcomes.

## Native qualification remains open

The current 256k ordinary-prefix run uses the previously built 8612387
runtime and continues independently. Its outcome does not qualify this fix.
Both the whole provider and the normal product CLI must be rebuilt, since
the C entry point changed. The prepared native tail and retirement r1 plans
bind the older runtime and must be rebound before execution. Their original
GB10 prompts, expected outputs and numerical tolerances remain unchanged.

Next validation uses the original q8191/out32 and q8193/out32 boundaries,
same-build ordinary and native q8192 controls, then the three original native
retirement cases. Host transport checks establish no native numerical,
performance or release acceptance.
