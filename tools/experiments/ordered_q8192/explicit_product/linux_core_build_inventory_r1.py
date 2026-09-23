"""Evaluate only the actual builder's source-inventory block on the controller."""
from pathlib import Path
import ast
import hashlib
import importlib.util
from types import SimpleNamespace

sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()


def inventory(checkout):
    preparation = checkout / 'tools/prepare_linux_core_windows.py'
    spec = importlib.util.spec_from_file_location('inventory_preparation_' + sha(preparation), preparation)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imported = module.verify_import()
    builder = checkout / 'tools/build_linux_core_windows.py'
    tree = ast.parse(builder.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    block = next(n for n in main.body if isinstance(n, ast.Try))
    first = next(i for i, n in enumerate(block.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'source_paths' for t in n.targets))
    last = next(i for i, n in enumerate(block.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == 'record'
                        and isinstance(t.slice, ast.Constant) and t.slice.value == 'source_inputs' for t in n.targets))
    nodes = block.body[first:last + 1]
    assert nodes and all(isinstance(n, (ast.Assign, ast.AugAssign, ast.If)) for n in nodes)
    flags = ('gb10_convolution', 'gb10_gdn', 'gb10_projections', 'gb10_prefill_projections',
             'gb10_normalization', 'gb10_moe')
    scope = dict(ROOT=checkout, inventory=imported, args=SimpleNamespace(**{name: True for name in flags}),
                 sha=sha, record={})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(builder), 'exec'), scope)
    result = scope['record']['source_inputs']
    assert result and all((checkout / item['path']).is_file() for item in result)
    assert len({item['path'] for item in result}) == len(result)
    return result
