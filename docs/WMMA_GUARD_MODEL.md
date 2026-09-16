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
