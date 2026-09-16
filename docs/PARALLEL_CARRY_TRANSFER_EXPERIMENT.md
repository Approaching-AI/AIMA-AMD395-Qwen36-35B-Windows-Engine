# Parallel exact carry windows across adjacent lane teams

Source `01111cc926391acac23711d67d1bfa1e48d47b1e` assigns adjacent
four-lane teams to separate K16 groups of one output. Subwaves of8/16/32
lanes scan two/four/eight exact integer increments. Every interior carry must
retain the certified sign and exponent; the final group uses original
normalization. Rejected windows reuse aligned sums only at their original
exponent and otherwise use unchanged staged replay. This changes component
geometry only; all original candidates and36-byte prepared rows remain.

Local54 Rust/470 Python(two skipped), C/q16 ABI, clippy and hygiene pass in
219.514235 s. Native21 generated cases cover2866566 ordered raw K16 states
with no mismatch,215446 accepted windows,860592 reused groups and227280
transition replays. Normal/audit parity, full prepared words, original inputs,
partial windows and guards pass. Across both captures,16 attempts per shape
verify1342177280 original raw/BF16 outputs and204825488 unrounded selected
outputs without a mismatch. QKV repeats1023 captured rows; OUT uses a
separate full8192 GB10 reference. No new prompt or token loop is run.

| Full preparation and replay | Current4 lanes ms | Parallel8 lanes ms | Parallel16 lanes ms | Parallel32 lanes ms |
| --- | ---: | ---: | ---: | ---: |
| q8192 QKV | 43.0226 | 101.390 | 154.322 | 219.964 |
| q8192 OUT | 159.468 | 361.363 | 539.188 | 727.783 |

Each variant includes its own full preparation and limits dispatches to4096
blocks. Candidate quanta are262144/131072/65536/32768. One warmup and three
rotating samples check every result. Generated production kernels declare
56/56/57 VGPRs, no LDS and no private allocation; these are compiler metadata,
not measured occupancy. Native small build/test/full build/QKV/OUT walls
2886.189/786.986/113905.841/5877.818/9995.103 ms finish normally on baiying
with host guards. No variant earns integration. Keep the original runtime.
Evidence: `parallel-transfer-projection-native-components-20260916.json`,
168357 bytes, SHA256
`ca4531d4929819cda3db21cc8ca2c5d336886c9ce491def193046a7307771df2`.
No product dispatch, retained performance or release change follows.
