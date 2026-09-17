# QKV correction through the convolution consumer

The isolated observer preserves every original projection correction and
the production convolution. It tests whether their complete BF16 consumer
could certify omission before any runtime filtering is implemented.

For each four-tap convolution, selected projection inputs use the original
PPB1000 error envelope and both adjacent BF16 hypotheses. BF16-rounded
products and four ordered FP32 additions enclose the sum. The entire sum
must belong to one transition segment of the exact SM121 SiLU table.
Comparing activation endpoints alone would be insufficient near the
negative SiLU minimum. A selected projection may be considered omittable
only if all four affected convolution outputs are constant. The final
three projection rows remain protected for the next convolution window.
Runtime integration must also account for intermediate prefix checkpoints.

Source `f0ed627` passes the retained algorithm4 captured QKV route on
baiying. The test retains all 7169 captured input rows and repeats the
first 1023 rows to reach q8192. Reference tensors are loaded only after GPU
calculation; the extension is an operator shape, not a new captured prompt.

| Captured layer-zero observation | Result |
| --- | ---: |
| Projection output cells | 67108864 |
| Original selected corrections | 4331635 |
| Valid selected input ranges | 3387694 |
| Provisionally omittable corrections | 1221516 (28.1999%) |
| Constant convolution outputs | 63177324 |
| Protected selected halo cells | 1760 |
| Actual projection BF16 changes | 430461 |
| Changes among omittable candidates | 17695 |
| Certificate dispatch with completion | 11.8626 ms |

All selector decisions, valid corrected endpoint intervals and constant
convolution outputs pass. All 67108864 projected BF16 cells match the
independent-row GB10 references; Q/K/V convolution references match for all
original 7169 rows. Guards and immutable inputs pass. Host sanitizers cover
64304 table segments, 64303 boundaries, 397536 four-tap combinations,
131616 halo conditions and complete graphs with actual input substitutions.

The original range helper limits selected inputs to nine BF16 values;
943941 selected cells receive no range under its combined width/tiny/special
rules. Source `11006a1` removes the width cap because each convolution
product is monotone and does not require enumerating interior inputs.
Its host tests additionally enumerate 1414491 BF16 values in wide intervals,
including crossings of zero and both weight signs, with no false certificate.

The second captured run still matches every projection and original
convolution reference, and all declared intervals and constant outputs.
Valid selected ranges increase to 3460386; 72692 are wider than the old
limit. Potential omission rises to 1273093 (29.3906%), including 44512 wide
inputs, and covers 29331 actual BF16 projection changes. Constant outputs
number 63431142. The single certificate dispatch takes 11.3187 ms; this is
not a controlled net performance comparison with the earlier clock.

The remaining 871249 invalid ranges are tiny inputs. A separate inspection
of the fingerprinted operands finds no all-zero input or weight row and no
nonfinite BF16 operand. Keep the tiny-value guard and original empirical
error envelope. Observe all 30 real-model linear layers before deciding
whether the captured layer-zero coverage justifies runtime filtering.

No correction is actually omitted in this observer. Its one dispatch clock
excludes allocation, matrix production, L2 preparation, original correction,
convolution and validation. There is no net speedup or model token result.
The runtime, package and release gates remain unchanged.

[Pinned source, command, references and checks](../benchmarks/correctness/conv-consumer-interval-capture-20260917.json):
46712 bytes, SHA256
`02e32f9f3be1d94a002475c547f48d766c39984ae4336db0a0f5647091872b7d`.

[Wide-interval source and capture](../benchmarks/correctness/conv-consumer-wide-interval-capture-20260917.json):
62676 bytes, SHA256
`e9ae24e12ea26245893095e5dfb348ca80c0997f851b57847f472f8ac4a1d277`.
