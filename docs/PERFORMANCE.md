# Real-model performance

These are retained native Windows results for Qwen3.6-35B-A3B BF16 on Ryzen AI
Max+ 395 (`gfx1151`), batch size 1. Startup is measured separately from TTFT.
Every retained product row was accepted only with its matching external BF16
correctness boundary.

## Product matrix

| Mode | Prompt shape | Total tokens | Load ms | TTFT ms | Prefill tok/s | TPOT ms | Decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| cold | q8192 | 8,192 | 19,940.245 | 3,852.909 | 2,126.186 | 32.732 | 30.551 |
| cold | q16384 | 16,384 | 19,948.525 | 8,027.368 | 2,041.018 | 32.457 | 30.810 |
| cold | q32768 | 32,768 | 19,909.139 | 17,274.313 | 1,896.921 | 33.935 | 29.468 |
| cold | q65536 | 65,536 | 19,924.188 | 41,381.599 | 1,583.699 | 39.778 | 25.139 |
| cold control | q130560 | 130,560 | 19,938.847 | 124,594.960 | 1,047.875 | 47.173 | 21.198 |
| cold | q131072 | 131,072 | 20,051.963 | 108,563.050 | 1,207.335 | 42.762 | 23.385 |
| prefix | 16,384 + 1,024 | 17,408 | 19,919.603 | 1,529.093 | 11,384.526 | 36.488 | 27.406 |
| prefix | 32,768 + 1,024 | 33,792 | 19,920.588 | 1,930.620 | 17,503.185 | 38.509 | 25.968 |
| prefix | 65,536 + 1,024 | 66,560 | 19,933.010 | 3,065.768 | 21,710.708 | 36.922 | 27.084 |
| prefix | 129,536 + 1,024 | 130,560 | 20,035.182 | 7,899.540 | 16,527.544 | 42.745 | 23.394 |
| prefix | 131,072 + 1,024 | 132,096 | 20,023.448 | 8,592.158 | 15,374.019 | 42.890 | 23.315 |
| prefix | 262,144 + 1,024 | 263,168 | 20,045.888 | 9,278.300 | 28,363.817 | 55.555 | 18.000 |

The last historical prefix row used a 263,680-token qualification service
limit. The public HTTP profile defaults to a stricter 262,144-token total
context; operators must not infer that the larger row changes the released
default contract.

## q8192 retained target

The confirmed retained target is 1,506.407 prefill tok/s and 4,187.416 ms TTFT.
The accepted row reached 2,126.186 tok/s and 3,852.909 ms. Model plus engine
load remained below the 30-second product bound.

## q8192-neighbor continuity gate

The v1.0.1 repair replaces fixed-q8192-only CK-FMHA and fused-GDN calls with
dynamic q8191/q8193 entries while leaving the retained q8192 entry points
unchanged. The isolated gfx1151 provider smoke produced:

| Provider | q8191 | q8192 fixed | q8193 | Neighbor/fixed ratio | Numerical gate |
|---|---:|---:|---:|---:|---|
| CK-FMHA | 19.888 ms | 19.983 ms | 20.067 ms | 0.995 / 1.004 | no value above `1e-5`; no nonfinite |
| fused GDN, decay | 8.546 ms | 8.773 ms | 8.709 ms | 0.974 / 0.993 | max output error `7.63e-6`; async exact |
| fused GDN, log-g | 8.731 ms | 8.727 ms | 8.703 ms | 1.000 / 0.997 | max output error `7.63e-6`; async exact |

The q8191 CK comparison had 267 bit-level differences at the final compared
token, with maximum absolute error `2.98e-8`; none exceeded the declared
`1e-5` component tolerance. q8193's q8192 prefix was bitwise exact. The packed,
fixed q8192, and q262144 tile-regression checks were also exact. These numbers
remain synthetic component evidence rather than product inference evidence.

The native Windows real-model publication gate passed on `baiying` at source
commit `212bd8159a92b33e46f7647b00957871b25fb639`. It used the model at
`D:\models\Qwen3.6-35B-A3B`, the public GB10 oracle, `max_tokens=1` and `2`,
and three cold-prefix repetitions per shape. All 18 returned token sequences
matched GB10 exactly.

| Prompt | max_tokens=1 median TTFT | max_tokens=2 median TTFT | All-six range |
|---|---:|---:|---:|
| q8191 | 4,185.807 ms | 4,253.678 ms | 4,099.691–4,272.996 ms |
| q8192 | 3,883.923 ms | 3,887.042 ms | 3,872.773–3,928.488 ms |
| q8193 | 5,483.337 ms | 5,529.695 ms | 5,482.172–5,541.596 ms |

The worst neighbor/q8192 median ratio was `1.422597`; the worst positive
residual was `1,642.653 ms`. Both pass the declared `2x` and `5,000 ms` limits,
and every q8192 sample beats the retained `4,187.416 ms` TTFT target. The
machine-readable product record, provider smoke, verifier, and GB10 oracle are
under `benchmarks/` and `scripts/`.

## Correctness attachment

At q8192 the external BF16 authority and native runtime selected token 16. The
authority first-token logit was 24.750 and native was 24.875, an absolute
difference of 0.125 (the declared tolerance boundary).

Long continuation checks generated 512/512 matching tokens:

| Case | Input digest | Output digest | Match |
|---|---|---|---:|
| cold q131072 | `ffef18d340fb4fe8` | `4813300b562d057b` | 512 / 512 |
| prefix q131072 + 1024 | `9a681295dde809d4` | `fd69ca7327c912bf` | 512 / 512 |
| prefix q262144 + 1024 | `6aa7d9579bb8427c` | `7a8c5d8ec398878b` | 512 / 512 |

Digests are compact publication identifiers, not standalone correctness
authority. Acceptance was token-for-token against the external BF16 service.

## Interpretation

TTFT excludes model/engine startup but includes real prompt prefill and the
first generated token. Prefix throughput divides the full effective prompt by
the measured reused-prefix request time and should not be compared as cold
compute throughput. Results are specific to the stated model, hardware,
toolchain, runtime profile, and batch size.

Machine-readable rows and hashes are under `benchmarks/performance/`.
