"""Observe the original BF16 model with unchanged frozen reference kernels."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

B = Path(__file__).resolve().parent
G = Path('/Users/jiawei-macmini/projects/AIMA-explicit-gdn-layout-candidate')
old_name = 'qrt-gb10-ordered-gdn-layers-20260923-r1'
name = 'qrt-gb10-explicit-gdn-layers-20260923-r1'
old_stage = B / (old_name + '-stage')
stage = B / (name + '-stage')
stage.mkdir()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
manifest = read(old_stage / 'source-inputs.json')
files = {}
for relative, expected in manifest['files'].items():
    if relative.startswith('candidate/u32/'):
        continue
    src = old_stage / relative
    assert sha(src) == expected
    dst = stage / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    files[relative] = expected
for file in ('ordered_pipeline.py', 'group16.py', 'exp2.py'):
    src = G / 'native/providers/gdn_explicit_layout' / file
    dst = stage / 'candidate/explicit_layout' / file
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    files[str(dst.relative_to(stage))] = sha(dst)
assert sha(stage / 'candidate/u64/ordered_inverse.py') == sha(G / 'native/providers/gdn/ordered_inverse.py')
observer = stage / 'scripts/capture_ordered_gdn_layers.py'
s = observer.read_text()
start = s.index("        previous = sys.modules.get('integer_u')")
end = s.index('        self.inverse = ', start)
s = s[:start] + '''        previous = {key: sys.modules.get(key) for key in ('integer_u', 'group16', 'exp2')}
        try:
            directory = ROOT / 'candidate/u64'
            arithmetic = module('ordered_u64_arithmetic', directory / 'integer_u.py')
            sys.modules['integer_u'] = arithmetic
            pipeline = module('ordered_u64_pipeline', directory / 'ordered_pipeline.py')
            self.variants['u64'] = (pipeline, arithmetic.integer_u_kernel)
            directory = ROOT / 'candidate/explicit_layout'
            sys.modules['group16'] = module('ordered_explicit_group16', directory / 'group16.py')
            sys.modules['exp2'] = module('ordered_explicit_exp2', directory / 'exp2.py')
            pipeline = module('ordered_explicit_pipeline', directory / 'ordered_pipeline.py')
            self.variants['explicit_layout'] = (pipeline, pipeline.integer_u_kernel)
        finally:
            for key, prior in previous.items():
                if prior is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = prior
''' + s[end:]
s = s.replace("('u64', 'u32')", "('u64', 'explicit_layout')").replace("['u64', 'u32']", "['u64', 'explicit_layout']")
assert 'u32' not in s
compile(s, str(observer), 'exec')
observer.write_text(s)
files[str(observer.relative_to(stage))] = sha(observer)
commit = subprocess.check_output(['git', '-C', str(G), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
assert commit == 'b07bf58e5140ea6e256c477f4aacfd39fc89695d'
manifest.update(candidate_base_commit=commit, files=files,
                variants=dict(u64='Original e15 unsigned64 numerical control',
                              explicit_layout='Exact committed explicit-layout kernel sources and unchanged inverse'),
                candidate_runtime_report_sha256=sha(G / 'benchmarks/correctness/ordered-gdn-explicit-runtime-preparation-20260923.json'),
                previous_all_layer_manifest_sha256=sha(old_stage / 'source-inputs.json'),
                generator_sha256=sha(Path(__file__)))
manifest.pop('u32_equivalence_report_sha256')
manifest.pop('u32_continuous_state_report_sha256')
(stage / 'source-inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
archive = B / (name + '.tar')
with tarfile.open(archive, 'x', dereference=True) as tar:
    for relative in sorted([*files, 'source-inputs.json']):
        tar.add(stage / relative, arcname=relative, recursive=False)
old_record = read(B / (old_name + '-stage.json'))
wrapper = (B / old_record['script']).read_text()
assert sha(B / old_record['script']) == old_record['script_sha256']
wrapper = wrapper.replace(old_name, name).replace(old_record['archive_sha256'], sha(archive))
script = B / ('run-' + name + '.py')
compile(wrapper, str(script), 'exec')
script.write_text(wrapper)
dispatcher = B / 'dispatch_gb10_explicit_gdn_layers_r1.py'
dispatch = (B / 'dispatch_gb10_ordered_gdn_layers_r1.py').read_text().replace(old_name, name)
compile(dispatch, str(dispatcher), 'exec')
dispatcher.write_text(dispatch)
record = dict(name=name, archive=archive.name, archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
              script=script.name, script_sha256=sha(script), dispatcher=dispatcher.name,
              dispatcher_sha256=sha(dispatcher), source_inputs_sha256=sha(stage / 'source-inputs.json'),
              ssh_timeout_seconds=1950, candidate_base_commit=commit,
              original_model_sources_unchanged=True, original_576_outputs_required=True,
              observer_source_base_commit=manifest['source_commit'], generator_sha256=sha(Path(__file__)))
(B / (name + '-stage.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
