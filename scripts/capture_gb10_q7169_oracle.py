"""Diagnostic reuse of the frozen q8192 capturers with the q7169 fixture.

Only fixture constants and record labels change. The raw-logit capture still
requires an independently matching BF16 full-vocabulary replay of the service.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

import capture_gb10_q8192_oracle as raw
import capture_gb10_q8192_continuation as continuation

raw.torch.set_num_threads(8)
modules = (raw, continuation)
for module in modules:
    module.PROMPT_TOKENS = 7169
    module.PROMPT_SEED = 50536152
    module.PROMPT_FIRST_TOKEN = 30199
    module.PROMPT_SHA256 = 'b4ba753492f994e7c64ca7805c1b48523993ff24f6e75b44fa5705067d1de87d'
    module.PROMPT_FNV1A64 = 'c900c18703532d6a'

records = []
for module, label in zip(modules, ('raw_logit', 'continuation')):
    sys.argv = [module.__file__, '--timeout-seconds', '40']
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = module.main()
    if result != 0:
        raise RuntimeError(f'{label} capture did not complete')
    record = json.loads(output.getvalue())
    record['record_type'] = 'gb10_q7169_' + label + '_capture'
    record['capture_source_sha256'] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    records.append(record)
print(json.dumps({'schema_version': 1, 'record_type': 'q7169_gb10_refreshed_oracle',
                  'fixture_override_only': True, 'captures': records}, sort_keys=True))
