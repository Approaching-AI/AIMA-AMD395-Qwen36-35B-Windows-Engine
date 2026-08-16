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
commit `34264b47c468b52f626ef350ba7a3cd746550b5e`. It used the model at
`D:\models\Qwen3.6-35B-A3B`, the public GB10 oracle, `max_tokens=1` and `2`,
and three cold-prefix repetitions per shape. All 18 returned token sequences
matched GB10 exactly.

| Prompt | max_tokens=1 median TTFT | max_tokens=2 median TTFT | All-six range |
|---|---:|---:|---:|
| q8191 | 3,956.174 ms | 4,030.357 ms | 3,858.328–4,031.200 ms |
| q8192 | 3,881.627 ms | 3,884.071 ms | 3,852.914–3,891.808 ms |
| q8193 | 3,894.552 ms | 3,893.829 ms | 3,863.929–3,912.984 ms |

The worst neighbor/q8192 median ratio is now `1.037663`; the worst positive
residual is `146.286 ms`. Both pass the tightened `1.10x` and `500 ms`
limits, and every q8192 sample beats the retained `4,187.416 ms` TTFT target.
The former q8193 5.5-second fallback is no longer present.

## Wide prompt-length continuity gate

The neighbor check above is supplemented by a 72-request cold-prefill sweep
covering eight independent length boundaries from q4096 through q16384. Each
boundary tests `q-1`, `q`, and `q+1` three times with a unique leading token
to prevent prefix-cache reuse. Every AMD-generated token ID matched the GB10
BF16 authority.

| Center | q-1 median TTFT | q median TTFT | q+1 median TTFT | Local max/min |
|---:|---:|---:|---:|---:|
| 4,096 | 2,266.678 ms | 2,194.764 ms | 2,235.846 ms | 1.032766 |
| 6,144 | 3,542.818 ms | 3,475.555 ms | 3,367.777 ms | 1.051975 |
| 8,192 | 3,970.170 ms | 3,867.689 ms | 3,925.276 ms | 1.026497 |
| 9,216 | 4,616.410 ms | 4,564.467 ms | 4,579.720 ms | 1.011380 |
| 10,240 | 5,111.462 ms | 5,069.652 ms | 5,091.924 ms | 1.008247 |
| 12,288 | 6,025.344 ms | 5,972.787 ms | 6,029.855 ms | 1.009555 |
| 14,336 | 7,293.122 ms | 7,296.177 ms | 7,603.634 ms | 1.042576 |
| 16,384 | 8,314.751 ms | 8,037.034 ms | 8,083.771 ms | 1.034555 |

All eight local gates remain below `1.10x` and `500 ms`; the worst observed
values are `1.051975x` and `307.457 ms`. Median throughput ranges from
`1,733.930` to `2,118.061 tok/s` across all 24 cohorts, a global max/min ratio
of `1.221537`. This distinguishes normal shape-efficiency variation from a
length-rounding cliff: no tested exact boundary or adjacent non-boundary length
takes a separate multi-second fallback.

The machine-readable q8192 product record, wide continuity record, provider
smoke, verifiers, and GB10 oracle are under `benchmarks/` and `scripts/`.

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
