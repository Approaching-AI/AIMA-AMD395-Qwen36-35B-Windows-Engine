# Whole-row narrow admission for staged half projection

This isolated component retains the existing lossless36-byte K16 operands,
four-lane reduction and ordered K16 normalization. The candidate preclassifies
both complete original rows. Admission requires K<=8192, signed zero or BF16
exponents95..159, and a supported lossless normal-half encoding for every
group. An unsupported row selects the complete original staged2 dot.

Within this domain nonzero carries are at least2^-89 and absolute growth
through8192 products stays below2^79. The candidate can keep the carry in one
FP32 register without repeated endpoint/range checks. This does not permit
native FP32 summation, reassociation or changing midpoint candidate identities.
Zero groups retain the previous carry. Half-product scales below2^-126
contribute integer zero; all other admitted scales are normal and finite.

The earlier general staged-half-FP32 component retained per-group checks and
whole-dot restart; its measured regression is a prior, not evidence for this
specialization. The sparse-byte tile experiment also changed scheduling and
operand storage. This component keeps the existing four-lane schedule and
compares two- and four-group staging against the original staged2 control.

`tests/test_narrow_half_carry.py` checks all65536 BF16 encodings, independent
admission, lossless roundtrips and ordered original K16 endpoints throughK8192
under ASan/UBSan. The native fixture additionally checks every raw carry,
production/audit parity, actual original-row flags, guards and unchanged inputs,
including unsupported widths, exponent spans, subnormals and nonfinite words.

The real projection fixture charges both lossless preparation passes, both
whole-row classification passes and completed replay. It checks all original
GB10 BF16 outputs, unrounded selected values, inactive cells, encodings and
candidate identities after the warmup and each rotated sample. QKV uses the
retained matrix4 producer with PPB1000; OUT uses matrix0 with PPB10000. This OUT
fixture is not the retained coarse-OUT provider. Full8192 projection rows include
the captured7169 inputs followed by1023 repeated inputs.

Source `0714cea` completes all28 native configurations,2645036 original carry
endpoints,1190108 admitted groups, complete flags/encodings and memory checks.
The host audit admits2112 dots and rejects960, checks322080 ordered endpoints
and14991360 decoded words. All65536 BF16 encodings are classified independently.

| q8192 operator | Original staged2 ms | Narrow staged2 ms | Narrow staged4 ms |
| --- | ---: | ---: | ---: |
| QKV |42.7773|49.2372|48.1558|
| Original midpoint OUT |157.2720|143.0060|142.8770|

Each median includes preparation, candidate classification and completed
replay. Every QKV candidate sample is slower than every control sample;
every OUT candidate sample is faster than every control sample. All67108864
QKV and16777216 OUT GB10 outputs pass, as do4331635/8471989 raw selected
values, inactive outputs and every warmup/timed attempt. QKV admits3458934
selected dots (8080/8192 weight rows, all8192 input rows); OUT admits8469953
selected dots (all2048 weight rows,8190/8192 input rows). Mixing whole-dot
branches may affect QKV scheduling, but these measurements do not establish
that cause. The control reports candidate classification counts from the
completed comparison; it does not run classification in its clock.

Replay kernels use49/48/48 VGPR and no scratch or spills. The classifier uses
6 VGPR and16 shared bytes. These are static resources, not occupancy results.
Additional flags occupy65536 bytes for QKV or40960 for OUT; the retained
lossless operands and candidate list are unchanged. Both native builds and
all tests complete with exit0 and passing host guards. The724-file source
inventory matches the commit; that inventory includes uncompiled files.

Keep this component outside common runtime dispatch because it does not
improve both geometries. No retained coarse-OUT, MoE or model measurement
follows. Runtime dispatch, packaged defaults and the23353.80795ms qualified
model control remain unchanged. The10-second and retained4187.415605ms
mission targets remain open.

[Complete evidence](../benchmarks/correctness/narrow-half-projection-native-components-20260918.json):
364693 bytes, SHA256
`c5d2d32fa080dd86a6f35875e13e5cec85769d38bf4fd5a65d762791a281c420`.
