# Four-lane exact GDN candidate

This isolated candidate distributes each original K16 dot over four adjacent
lanes in W/U, recurrent state and output. Each lane holds four products. The
retained scalar schedule holds sixteen, which contributes to register pressure
in the complete GDN chain. The completed q8192 profile attributes 3327.918 ms
to GDN recurrence; an actual product measurement is required to decide whether
the new schedule helps.

The candidate reuses the existing strong row certificate: every nonzero BF16
operand must have exponent 84 through 174. Both complete operand rows are
checked, including freshly rounded recurrent checkpoints and residuals. K64
and K128 fall within the existing K4096 proof. Product multiplication, unsigned
modulo reduction, integer carry, ordered K16 normalization, BF16 endpoints,
exponential lookup and final FMA retain their original definitions. Excluded
rows use the retained scalar operation in lane zero.

Shared packed operands and nonoverlapping arena lifetimes follow the retained
scalar layout. A quad publishes each result once. W/U captures every input V
column before writing U, including when U and V share storage. Eight complete
state rows stay with their original CTA. Output retains each Q/H result until
all readers cross the barrier that permits reusing the shared arenas.

The local test runs the actual candidate kernel bodies under ASan/UBSan with
threaded CTAs and host-emulated subgroup transport. Its independent wide
integer accumulator checks original tensor values, one-token and 65-token
tails, the first and last heads/column tiles, U=V, all recurrent checkpoints,
untouched output cells and immutable inputs. Both certified and fallback rows
execute. The test passes after correcting a test-helper name collision with
the macOS math library; the initial failed compile is retained in local evidence.
This host check does not execute GPU DPP instructions.

The native fixture compares the unchanged complete chain with quad W/U plus
output, then with quad state added. It checks seven input families and eight
boundary sizes, both U ownership modes, all intermediates and independent CPU
dots. Original GB10 q7169 replay and a separately labelled q8192 capture
extension are prepared. The extension repeats captured rows and is a component
diagnostic; only a subsequent original-token q8192 model run can establish
product correctness or performance.

The provider dispatch and packaged profile are unchanged. Native compilation,
GPU numerical checks, register/resource measurements and product comparison
remain pending. No performance improvement or release acceptance is claimed.

Source `50cc22d`, the local arithmetic and host checks, and the prepared native
commands are recorded in
[`pending-native-candidates-20260919.json`](../benchmarks/correctness/pending-native-candidates-20260919.json).
The command file passes the actual Windows PowerShell parser. Native dispatch
requires the active original 256k owner's completed host cleanup record;
parsing has not executed the fixture or loaded the model.
