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
error envelope. The subsequent real-model observer covers all 30 linear layers.

No correction is actually omitted in this observer. Its one dispatch clock
excludes allocation, matrix production, L2 preparation, original correction,
convolution and validation. Neither captured-data run establishes a net
speedup or a model token result.

[Pinned source, command, references and checks](../benchmarks/correctness/conv-consumer-interval-capture-20260917.json):
46712 bytes, SHA256
`02e32f9f3be1d94a002475c547f48d766c39984ae4336db0a0f5647091872b7d`.

[Wide-interval source and capture](../benchmarks/correctness/conv-consumer-wide-interval-capture-20260917.json):
62676 bytes, SHA256
`e9ae24e12ea26245893095e5dfb348ca80c0997f851b57847f472f8ac4a1d277`.

## Complete real-model observation

Source `0e5bf7d` adds the default-off
`QRT_QWEN36_Q8192_CONV_CONSUMER_AUDIT` option. It copies the original
producer, executes every original correction, invokes the original
convolution into private scratch and reduces all comparison counters on the
GPU. Its 805310592-byte guarded temporary owner includes explicit completion
and cleanup. Native capture checks match all 16 metrics with the separate
full host audit; null-call rejection and cleanup after an injected replay
failure also pass. Prefix suffixes and checkpoint stores are outside this
observer route.

One fresh baiying process uses the new whole DLL with the retained compact
MoE1, adaptive OUT2, exact attention, register PV, FLA and CLI. All original
8192 prompt IDs, 512 GB10 output IDs and actual callbacks pass, with first
logit 10.375 and zero error. Load is 21241.5679 ms; instrumented TTFT is
26121.1392 ms and TPOT is 101.348289 ms. Observation changes the measured
wall, so these values do not replace the 25379.41695 ms experimental baseline.

| Across all 30 linear layers | Count |
| --- | ---: |
| Original selected corrections | 160641307 |
| Valid selected intervals | 157181502 |
| Provisionally omittable corrections | 38931451 (24.2350%) |
| Wide selected intervals | 3823556 |
| Tiny selected inputs | 3459805 |
| Tiny inputs that are nonzero | 3454144 |
| Actual projection BF16 changes | 4144061 |
| Changes among omittable candidates | 876722 |
| Range, output-certificate and unselected-endpoint failures | 0 |

Keep the observer default off and leave omission unimplemented for now.
Whole-model coverage is lower than layer zero, and no net gain has been
measured. With TTFT still above 10 seconds, prioritize a broader attention
replacement. These results remain evidence for revisiting a materially
improved consumer route. Package, retained-performance and release gates
remain open.

[Native owner qualification](../benchmarks/correctness/conv-consumer-runtime-observer-native-20260917.json):
55290 bytes, SHA256
`5aa2b0bd5cc5e5f416da11207a45041244fdbdd974d0b2bae5d05bc887723807`.
[All-layer model boundary and observations](../benchmarks/correctness/conv-consumer-all-layer-observation-20260917.json):
306481 bytes, SHA256
`6e648226d496e9c7019f6484b245a9e6640853de8bfffb4649fd7f89d0a71ec9`.
