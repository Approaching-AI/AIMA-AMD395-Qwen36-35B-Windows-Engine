"""Prepare an original-model all-layer comparison of the committed recurrence."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess
import tarfile

B = Path(__file__).resolve().parent
R = B.parents[1]
old_name = 'qrt-gb10-explicit-gdn-layers-20260923-r1'
name = 'qrt-gb10-compact-gdn-layers-20260923-r1'
old_stage = B/(old_name+'-stage')
stage = B/(name+'-stage'); stage.mkdir()
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
read = lambda p: json.loads(p.read_text())
manifest = read(old_stage/'source-inputs.json')
files = {}
for relative, digest in manifest['files'].items():
    source = old_stage/relative
    assert sha(source) == digest
    target = stage/relative; target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target); files[relative] = digest
arithmetic = R/'tools/experiments/ordered_q8192/persistent_gdn/compact'
for file in ('group16.py','exp2.py','ordered_pipeline.py'):
    assert sha(arithmetic/file) == files['candidate/explicit_layout/'+file]
target = stage/'candidate/explicit_layout/persistent.py'
shutil.copyfile(arithmetic/'persistent.py', target)
files[str(target.relative_to(stage))] = sha(target)
observer = stage/'scripts/capture_ordered_gdn_layers.py'
original = observer.read_text()
s = original
s = s.replace("for key in ('integer_u', 'group16', 'exp2')", "for key in ('integer_u', 'group16', 'exp2', 'ordered_pipeline')")
first = s.index("            directory = ROOT / 'candidate/u64'")
last = s.index("            directory = ROOT / 'candidate/explicit_layout'", first)
s = s[:first] + s[last:]
old = "            self.variants['explicit_layout'] = (pipeline, pipeline.integer_u_kernel)"
assert s.count(old) == 1
s = s.replace(old, """            self.variants['persistent'] = (pipeline, pipeline.integer_u_kernel)
            sys.modules['ordered_pipeline'] = pipeline
            self.persistent = module('ordered_persistent_recurrence', directory / 'persistent.py').persistent_kernel""")
first = s.index('        current = states[0]\n')
last = s.index('        torch.cuda.synchronize()\n', first)
s = s[:first] + """        current = states[1]
        self.persistent[(32, 16)](q, k, w, u, g, scores, table, states[0], current,
            core, None, None, T, CAPTURE=False, num_warps=4, num_stages=2, enable_fp_fusion=False)
""" + s[last:]
assert s.count("('u64', 'explicit_layout')") == 1
assert s.count("['u64', 'explicit_layout']") == 1
s = s.replace("('u64', 'explicit_layout')", "('persistent',)")
s = s.replace("['u64', 'explicit_layout']", "['persistent']")
compile(s, str(observer), 'exec')
observer.write_text(s)
files[str(observer.relative_to(stage))] = sha(observer)
# The original-model hooks, boundary capture and comparisons are unchanged;
# only variant declarations are normalized for this structural source check.
old_tree, new_tree = ast.parse(original), ast.parse(s)
old_hooks = next(x for x in old_tree.body if isinstance(x,ast.ClassDef) and x.name=='OrderedGdnCapture')
new_hooks = next(x for x in new_tree.body if isinstance(x,ast.ClassDef) and x.name=='OrderedGdnCapture')
normalized = s.replace("['persistent']", "['u64', 'explicit_layout']")
normalized_hooks = next(x for x in ast.parse(normalized).body if isinstance(x,ast.ClassDef) and x.name=='OrderedGdnCapture')
assert ast.dump(old_hooks, include_attributes=False) == ast.dump(normalized_hooks, include_attributes=False)
commit = subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True,timeout=10).strip()
assert subprocess.check_output(['git','-C',str(R),'show',commit+':tools/experiments/ordered_q8192/persistent_gdn/compact/persistent.py'],timeout=15) == target.read_bytes()
manifest.update(candidate_base_commit=commit, files=files,
    variants=dict(persistent='One persistent block per head/value tile; exact committed explicit recurrence'),
    persistent_component_manifest_sha256=sha(arithmetic/'manifest.json'),
    persistent_component_evidence_sha256=sha(R/'benchmarks/correctness/persistent-gdn-compact-controls-20260923.json'),
    previous_all_layer_manifest_sha256=sha(old_stage/'source-inputs.json'),
    original_model_hooks_ast_unchanged_except_variant_label=True,
    model_compute_sources_modified=False, candidate_outputs_fed_to_model=False,
    generator_sha256=sha(Path(__file__)))
(stage/'source-inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive = B/(name+'.tar')
with tarfile.open(archive,'x',dereference=True) as tar:
    for relative in sorted([*files,'source-inputs.json']): tar.add(stage/relative,arcname=relative,recursive=False)
old_record = read(B/(old_name+'-stage.json'))
wrapper = (B/old_record['script']).read_text()
assert sha(B/old_record['script']) == old_record['script_sha256']
wrapper = wrapper.replace(old_name,name).replace(old_record['archive_sha256'],sha(archive))
wrapper = wrapper.replace("('.json','.log','.txt','.py','.bin')", "('.json','.log','.txt','.py','.bin','.best_config')")
script = B/('run-'+name+'.py'); compile(wrapper,str(script),'exec'); script.write_text(wrapper)
dispatcher = B/'dispatch_gb10_compact_gdn_layers_r1.py'
# This collector already retains and verifies the frozen .best_config input.
dispatch = (B/'dispatch_gb10_explicit_q8192_layers_r1.py').read_text().replace(
    'qrt-gb10-explicit-q8192-layers-20260923-r1',name)
compile(dispatch,str(dispatcher),'exec'); dispatcher.write_text(dispatch)
record = dict(name=name,archive=archive.name,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
    script=script.name,script_sha256=sha(script),dispatcher=dispatcher.name,dispatcher_sha256=sha(dispatcher),
    source_inputs_sha256=sha(stage/'source-inputs.json'),ssh_timeout_seconds=1950,candidate_base_commit=commit,
    original_model_sources_unchanged=True,original_576_outputs_required=True,
    observer_source_base_commit=manifest['source_commit'],generator_sha256=sha(Path(__file__)))
(B/(name+'-stage.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
