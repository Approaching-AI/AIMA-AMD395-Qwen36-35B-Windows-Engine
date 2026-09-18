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

Native results are pending. Runtime dispatch, packaged defaults and accepted
model evidence are unchanged; component results alone cannot establish TTFT.
