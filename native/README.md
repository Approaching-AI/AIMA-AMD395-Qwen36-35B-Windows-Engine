# Native runtime and providers

- `src/` contains the stable C ABI and model-specific transaction core.
- `providers/` contains the whole-model provider and CK, Triton, AITER/FLA,
  and host-BF16 components, with source-adjacent upstream licenses.
- `generators/` contains fixed-shape Triton AOT generators.
- `aot/gfx1151/` contains the qualified generated inventory and metadata.

Generated objects are convenience/reproducibility artifacts, not numerical
authority. Rebuilt objects must pass their provider smoke and real-model
external BF16 boundary.
