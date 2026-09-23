# Optional persistent GDN prefill

`AIMA_PORT_NATIVE_GDN_PERSISTENT=1` requires
`AIMA_PORT_NATIVE_GDN_PREFILL=1` and applies only to cold q8192. Both settings
default off. The runtime uses five original unsigned64 upstream images and one
explicit-layout recurrence image, selected from the prepared Windows numerical
trial. Their identities are in
`native/linux_core_port/gb10_gdn_persistent_compile.json`.

The recurrence keeps each head/value tile's K128 state in shared memory through
all 128 chunks. It writes the complete BF16 core and final resident FP32 state.
The original cumsum plus six module launches replace the existing 390 AOT
launches per linear layer. This is a dispatch-count reduction, not a measured
speedup. The original preparation, cold-state clear and pointer-span checks
still apply. This opt-in rejects seeded prefill; decode arithmetic is unchanged.

The reference-side checks cover q7169, a partial 65-token continuation with a
nonzero state, and all 30 original q8192 GDN layers. Host checks cover actual AOT
argument widths, scratch ownership, six failed-dispatch positions and the
generated activation marker. Offline rebuilding checks executable content and
launch metadata without replacing the exact pending native images.

All 30 original q8192 GDN layers now match their complete BF16 core and FP32 final state on the reference host. Windows component execution and a complete q8192/out512 run are still required.
No Windows model result, load time or TTFT is qualified by these preparation checks.
The generated marker records `persistent_recurrence: true`, `stages: 6` and
`aot_launches: 7` when the new route actually executes.

The compact recurrence uses16-row shared tiles and looped K16 reductions.
Its instruction section is65,408 bytes, compared with615,296 in the parent
prototype; VGPR use is158 rather than225. The host ABI,6144 shared bytes,
allocation, dispatch count and numerical boundaries are unchanged. These are
compiler resource observations; actual Windows model performance is pending.
