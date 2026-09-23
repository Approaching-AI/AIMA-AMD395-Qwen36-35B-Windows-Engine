"""Freeze a product trial after the selected native component boundaries pass."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

from linux_core_build_inventory_r1 import inventory
from qualify_explicit_q8192_components_r1 import COMMIT, PLANS, qualify

B = Path(__file__).resolve().parent
P = B.parent.parent.parent / 'AIMA-explicit-q8192-candidate'
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check', action='store_true')
parser.add_argument('--dense-tile', type=int, choices=(0, 32, 64, 128), default=64)
parser.add_argument('--routed-tile', type=int, choices=(0, 32, 64, 128), default=64)
parser.add_argument('--without-attention', action='store_true')
parser.add_argument('--without-gdn', action='store_true')
args = parser.parse_args()
options = dict(dense_batch=args.dense_tile, routed_batch=args.routed_tile,
               attention=not args.without_attention, gdn=not args.without_gdn)
selection = qualify(B, P, **options)
if args.check:
    print(json.dumps(dict(selection=selection, remote_calls=0, staged=False), indent=2))
    raise SystemExit(0)
assert selection['ready'], 'Selected native numerical boundaries are pending; no stage files or remote calls.'
old = read(B / 'linux-core-windows-ordered-gdn-u64-source-r1.json')
repo = 'P:/projects/AIMA-public-explicit-q8192-20260923-r1'
plan = json.loads(json.dumps(old).replace(old['repo'], repo).replace('linux-core-batch-replay', 'explicit-q8192'))
plan['source_commit'] = plan['run_spec']['repo_commit'] = COMMIT
plan['environment'].update(selection['environment'])
assert plan['environment']['AIMA_PORT_PREFILL_PROJECTION_PROFILE'] == '0'
assert plan['environment']['AIMA_PORT_PREFILL_BATCH_REPLAY'] == '1'
assert plan['environment']['AIMA_PORT_PREFILL_TERMINAL_ONLY'] == '1'
assert plan['environment'].get('AIMA_PORT_PREFILL_GROUP_MAJOR_WEIGHTS', '0') == '0'
plan['build_source_inputs'] = inventory(P)
assert len(plan['build_source_inputs']) == 438
compiled = read(P / 'native/linux_core_port/explicit_q8192_compile.json')
gdn = read(P / 'native/linux_core_port/gb10_gdn_ordered_compile.json')
generator_paths = sorted({*compiled['source_files'], *gdn['source_files'],
    'tools/compile_linux_core_gdn_explicit.py', 'tools/compile_linux_core_q8192_explicit.py'})
plan['generator_source_inputs'] = [dict(path=name, bytes=(P / name).stat().st_size, sha256=sha(P / name)) for name in generator_paths]
plan['native_component_selection'] = selection
plan['component_selection_options'] = options
plan['local_gdn_host_contract_sha256'] = sha(P / 'build/explicit-gdn-host-r1/result.json')
plan['local_prefill_projection_host_contract_sha256'] = sha(P / 'build/explicit-prefill-host-r1/result.json')
plan['local_ordered_attention_host_contract_sha256'] = sha(P / 'build/explicit-attention-host-r1/result.json')
plan['local_gdn_preparation_sha256'] = sha(P / 'build/explicit-q8192-prepare-r1/prepare.json')
plan['local_prefill_projection_preparation_sha256'] = plan['local_gdn_preparation_sha256']
plan.pop('ordered_gdn_trial')
plan['explicit_q8192_trial'] = dict(candidate_commit=COMMIT,
    compiled_manifest_sha256=sha(P / 'native/linux_core_port/explicit_q8192_compile.json'),
    preparation_sha256=sha(P / 'native/linux_core_port/explicit_q8192_preparation.json'),
    q8192_embedded_images=13, q8192_image_bytes=521376, gdn_images=8, gdn_image_bytes=743344,
    generator_source_inputs_are_separate_from_actual_cxx_build_inputs=True,
    requested_environment=selection['environment'], activation_requires_same_run_markers=True,
    original_q8192_out512_tokens_and_first_logit_required=True, inference_acceptance=False,
    performance_acceptance=False, release_qualified=False)
family = next(name for name in ('routed', 'attention', 'dense', 'gdn') if name in selection['selections'])
if family == 'gdn':
    prior_root = B / 'native-gdn-layout-controls-windows-r2'
    remote_root = 'P:/projects/native-gdn-layout-controls-20260923-r2'
else:
    prior_root = B / PLANS[family][0]
    remote_root = read(prior_root / 'manifest.json')['remote_directory']
prior = prior_root / 'run/run-record.json'
assert sha(prior) == selection['selections'][family]['run_sha256']
plan['prior_owner_record'] = remote_root + '/run/run-record.json'
plan['prior_owner_record_sha256'] = sha(prior)
plan['stage_generator_sha256'] = sha(Path(__file__))
plan['build_inventory_helper_sha256'] = sha(B / 'linux_core_build_inventory_r1.py')
manifest = B / 'linux-core-windows-explicit-q8192-source-r1.json'
with manifest.open('x') as file:
    json.dump(plan, file, indent=2)
    file.write('\n')
dispatcher = B / 'dispatch_explicit_q8192_product_r1.py'
compile(dispatcher.read_bytes(), str(dispatcher), 'exec')
bundle = B / 'linux-core-windows-explicit-q8192-source-r1.bundle'
assert not bundle.exists()
subprocess.run(['git', '-C', str(P), 'bundle', 'create', str(bundle), old['source_commit'] + '..HEAD'],
               check=True, timeout=60)
stage = dict(source_commit=COMMIT, seed_repo=old['repo'], manifest=manifest.name, manifest_sha256=sha(manifest),
    dispatcher=dispatcher.name, dispatcher_sha256=sha(dispatcher), bundle=bundle.name, bundle_sha256=sha(bundle),
    qualifier_sha256=sha(B / 'qualify_explicit_q8192_components_r1.py'),
    gdn_qualifier_sha256=sha(B / 'qualify_native_gdn_explicit_r1.py'),
    stage_generator_sha256=sha(Path(__file__)), inventory_helper_sha256=sha(B / 'linux_core_build_inventory_r1.py'))
with (B / 'linux-core-windows-explicit-q8192-stage-r1.json').open('x') as file:
    json.dump(stage, file, indent=2)
    file.write('\n')
print(json.dumps(stage))
