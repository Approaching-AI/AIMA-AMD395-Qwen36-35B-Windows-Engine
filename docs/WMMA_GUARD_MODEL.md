# Observed WMMA arithmetic and a conditional bound

This investigation concerns native `gfx1151` arithmetic. It does not establish
inference or release acceptance. Production coarse projection retains its
`2^-19` coefficient; the `2^-20` specialization is an explicit diagnostic arm.

The generated integer probes reproduce every observed BF16/FP16 WMMA result
with eight ordered FP16 DOT2 operations. Independent floating fixtures include
normal and subnormal FP16 operands, signed zeros and independent FP32 carries.
The two floating seeds have 2,097,152 captured DOT2 states and 262,144 FP16
WMMA endpoints. Their exactly equivalent BF16 controls cover 196,608 endpoints.
All comparisons pass after the signed-zero and subnormal corrections below.

A separate BF16 probe covers zero or normal operands with raw exponents
80 through 174. It evaluates 1,048,576 masked K2 through K16 prefix outputs.
These are separate zero-padded matrix instructions, not direct observations
of states inside one WMMA instruction. Every result agrees with the model.

## Empirical pair operation

Let `p0=a0*b0`, `p1=a1*b1`, and let `C` be the incoming FP32 value. Let `ep`
be the maximum sum of operand exponents, without normalizing the products.
For a nonzero FP16 subnormal, the observed alignment uses exponent `-14`
without shifting its fraction. For a normal FP32 carry let
`ec=floor(log2(abs(C)))`. Zero products and zero carries use exponent sentinel
`-126` in the diagnostic model. Then the fitted operation uses

```
E = max(ep, ec - 2)
q = 2^(E - 24)
T = trunc(C/q) + 2*signbit(C)
T += signbit(a0)^signbit(b0) ? -(trunc(abs(p0)/q)+1) : trunc(abs(p0)/q)
T += signbit(a1)^signbit(b1) ? -(trunc(abs(p1)/q)+1) : trunc(abs(p1)/q)
D = FP32_RNE(T*q)
```

The negative-carry term uses the sign bit, including negative zero. An ordinary
`C < 0` test fails captured examples. Renormalizing FP16 subnormal operands also
fails. The diagnostic implementation clears subnormal outputs to positive zero;
the general behavior of arbitrary FP32 subnormal carries and outputs has not
been qualified. NaNs, infinities, overflow and alternate denormal modes are
outside this investigation.

## Conditional zero-C error bound

Assume the fitted operation describes all eight pair steps. Restrict the input
to zero or normal BF16 operands with raw exponents 80 through 174 and an initial
positive-zero carry. Nonzero-input steps have `q >= 2^-118`; cancellation occurs
on that grid. Zero-input pairs preserve a normal carry. Overflow is excluded
because the 16-product L1 sum is below `2^100`.

Write `u=2^-24` and `S=abs(p0)+abs(p1)`. Before the final RNE, the alignment
error has magnitude at most `3q`. When the carry determines `E`, `C/q` is
integral, reducing this to `2q <= u*abs(C)/2`. When the products determine `E`,
`q <= u*S`. Combining these cases with the final RNE gives

```
abs(D - (C+p0+p1)) <= alpha*abs(C) + beta*S
alpha = 3*u/2 + u*u/2
beta  = 4*u + 3*u*u
```

The initial positive-zero carry has the smaller coefficient `3u+2u^2`.
Bounding every previous mathematical prefix by its L1 sum gives

```
gamma[1] = 3*u + 2*u*u
gamma[k] = max((1+alpha)*gamma[k-1]+alpha, beta)
gamma[8] = 13.50000502169247... * u < 2^-20
```

The recurrence and final inequality are evaluated with exact rational integers.
The arithmetic result is conditional on the hardware model. Finite native
agreement does not prove that premise for every operand combination.

The wide BF16 capture is also checked independently using arbitrary-size signed
integers in units of `2^-149`, with no violation of the length-specific bound or
`2^-20`. Earlier adversarial fixtures do violate `2^-21`; the smaller maximum
of the new fixture does not justify that coefficient.

The raw commands, commits, capture hashes, counterexamples and independent
analysis source are recorded in
[`wmma-dot-chain-native-characterization-20260916.json`](../benchmarks/correctness/wmma-dot-chain-native-characterization-20260916.json),
[`wmma-dot-float-native-characterization-20260916.json`](../benchmarks/correctness/wmma-dot-float-native-characterization-20260916.json), and
[`wmma-bf16-guard-native-characterization-20260916.json`](../benchmarks/correctness/wmma-bf16-guard-native-characterization-20260916.json).

## Conditional integer residue recovery

The isolated H4 path represents each BF16 operand by an exactly encoded FP16
integer core of magnitude at most4080. Original low exponent terms remain as
explicit exceptions. One zero-C FP16 WMMA estimates the16-product integer sum;
one unsigned IU8 WMMA gives its exact low byte. The nearest congruent integer
is unique if the FP16 estimate has absolute error strictly below128.

Under the observed pair model, let P=4080² and e be the accumulated error before
a pair. The carried magnitude is at most2(j-1)P+e. Product quantum is at most1/4;
carry quantum is2^(floor(log2(carried_bound))-26). The alignment error is bounded
by the larger of three product quanta and two carry quanta; the initial positive
zero needs only two product quanta. Add half an FP32 ulp at the largest possible
pre-round magnitude. Exact rational recurrence gives errors
1.5,4.25,9.25,15.25,25.25,37.25,49.25,61.25 through eight pairs. The executable
derivation is `tests/test_wmma_integer_bound.py`.

This arithmetic bound is conditional on the empirical hardware model. It does
not certify H5/H6 recovery: the same conservative calculation gives245/980,
respectively. Those are inconclusive bounds, not observed counterexamples.
The windowed QK experiment preserves the original exception compensation,
carry order, normalization and fallback. It has no product dispatcher.

## Centered signed16 cross terms

For a signed16 integer write x=256h+l, where h is a signed high byte and l
an unsigned low byte. Set s=h+l-128. Then s is in[-256,254], exactly encoded
in FP16. For sixteen pairs the cross term is
`SS-HH-LL+128*(sum(sA)+sum(sB))+262144`. Thus two IU8 matrices for HH/LL and
one FP16 matrix for SS can reconstruct the complete signed16 dot.

The same conditional error recurrence, now with maximum absolute operand256,
gives85/256<1/2. Nearest-integer conversion therefore recovers SS under the
observed native model. This premise is explicit; the unrounded FP16 result is
not claimed to be an integer. The isolated centered-Karatsuba fixture compares
all partials and reconstructed integers with independent CPU sums and retains
the original four-IU8 control. Provider dispatch remains unchanged.
