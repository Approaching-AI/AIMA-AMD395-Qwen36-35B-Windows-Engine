# Engine

`qrt-server` builds the `qrt` resident lifecycle and OpenAI HTTP executable.
`qrt-cli` is the thinner native product/ABI command surface. Both link the C
core from `native/src` through their build scripts and share Apache-2.0 package
metadata from the workspace.

`runtime.env` is the qualified, path-free performance profile. Provider paths
must be supplied by `--provider`, the arbitrary-MoE options, and repeatable
`--set-env` values documented in the root README.
