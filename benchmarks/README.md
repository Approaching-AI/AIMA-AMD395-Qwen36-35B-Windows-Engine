# Benchmarks and correctness evidence

The evidence bundle is deliberately compact and reviewable:

- `performance/`: all retained Windows product rows and acceptance bounds;
- `correctness/`: external BF16 first-token and continuation bindings;
- `openai/`: API, lifecycle, queue, context, and prefix-cache acceptance; and
- `eval/`: MMLU-Pro aggregate acceptance plus 12,032 sanitized parity rows.

Evidence never includes model weights, evaluation question text, private
endpoints, credentials, or personal deployment paths. SHA256 values bind each
published artifact; diagnostic self-hashes are not correctness authority.
