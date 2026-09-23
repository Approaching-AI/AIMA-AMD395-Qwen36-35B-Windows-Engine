"""Compare actual embedded attention code, without rebuilding or running it."""
from pathlib import Path
import hashlib
import json
import struct

B = Path(__file__).resolve().parent
D = B / 'runtime-attention-codeobjects-r1'
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
prefixes = ('_ZN22qrt_sm121_q1_attention6append', '_ZN22qrt_sm121_q1_attention6scores',
            '_ZN32qrt_sm121_q1_segmented_attention14segment_kernel',
            '_ZN32qrt_sm121_q1_segmented_attention12merge_kernel')


def extract(path):
    raw = path.read_bytes()
    magic = b'__CLANG_OFFLOAD_BUNDLE__'
    start = raw.find(magic)
    assert start >= 0 and raw.find(magic, start + 1) < 0
    pos = start + len(magic)
    count, = struct.unpack_from('<Q', raw, pos)
    pos += 8
    assert count == 2
    bundles = []
    for _ in range(count):
        offset, size, length = struct.unpack_from('<QQQ', raw, pos)
        pos += 24
        assert length < 256 and pos + length <= len(raw)
        name = raw[pos:pos + length].decode()
        pos += length
        assert start + offset + size <= len(raw)
        if size:
            assert name == 'hipv4-amdgcn-amd-amdhsa--gfx1151'
            data = raw[start + offset:start + offset + size]
            assert data[:6] == b'\x7fELF\x02\x01'
            bundles.append((dict(bundle_offset=start, image_offset=start + offset,
                                 target=name, image_bytes=size,
                                 image_sha256=hashlib.sha256(data).hexdigest()), data))
    assert len(bundles) == 1
    return bundles[0]


def inspect(raw):
    h = struct.unpack_from('<16sHHIQQQIHHHHHH', raw)
    assert h[2] == 224 and h[11] == 64 and h[12] < 100
    sections = [struct.unpack_from('<IIQQQQIIQQ', raw, h[6] + i * h[11]) for i in range(h[12])]
    string_section = sections[h[13]]
    names = raw[string_section[4]:string_section[4] + string_section[5]]
    named = {names[s[0]:].split(b'\0', 1)[0].decode(): s for s in sections}
    symbols = named['.symtab']
    strings = sections[symbols[6]]
    text = raw[strings[4]:strings[4] + strings[5]]
    assert symbols[9] == 24
    result = {}
    for offset in range(symbols[4], symbols[4] + symbols[5], 24):
        name_offset, info, other, section_index, address, size = struct.unpack_from('<IBBHQQ', raw, offset)
        name = text[name_offset:].split(b'\0', 1)[0].decode()
        if not name.startswith(prefixes) or (info & 15) not in (1, 2) or not size:
            continue
        section = sections[section_index]
        file_offset = section[4] + address - section[3]
        data = raw[file_offset:file_offset + size]
        assert len(data) == size
        item = dict(symbol=name, bytes=size, type=info & 15, sha256=hashlib.sha256(data).hexdigest())
        if name.endswith('.kd'):
            assert size == 64
            normalized = bytearray(data)
            normalized[16:24] = b'\0' * 8
            item.update(descriptor_sha256_excluding_relative_entry_offset=hashlib.sha256(normalized).hexdigest(),
                        group_segment_fixed_bytes=struct.unpack_from('<I', data, 0)[0],
                        private_segment_fixed_bytes=struct.unpack_from('<I', data, 4)[0],
                        kernarg_segment_bytes=struct.unpack_from('<I', data, 8)[0])
        result[name] = item
    assert len(result) == 8
    return result


def main():
    sources = [
        ('observed', B / 'q1-cache-capture-whole-build-r1/qrt_qwen36_whole_provider.dll',
         '066299c0f40d81601d84e623cd3328f6faaf26a638212e0dbfdac1cd18f226ba',
         'b2b7c0bea3f3fe41ada0fc2d99abb48079e64478'),
        ('prior', B / 'packed-gate32-whole-build-r1/qrt_qwen36_whole_provider.dll',
         '983868b6413fdbe2fd7b0b0cf4cb8ea8d8620542b11995c0a441ba09b07d06de',
         'c90feccdfc01661b582c5a1db101f2660dc8642c'),
        ('qualified_probe', D / 'qualified-segmented-attention.exe',
         'd7758f48f263e9a909b11eff4cbb4e7c69ec353b134b1713654aadf4f177ea20',
         '9428e9eed89f0e38851e226a0f3220fb43acfffc')]
    records = {}
    for name, source, expected, commit in sources:
        assert sha(source) == expected
        bundle, raw = extract(source)
        image = D / (name + '-verified.hsaco')
        with image.open('xb') as stream:
            stream.write(raw)
        records[name] = dict(source_file=str(source), source_sha256=expected, source_commit=commit,
                             bundle=bundle, functions=inspect(raw), image_file=str(image))
    rows = []
    for symbol, item in records['observed']['functions'].items():
        original = records['prior']['functions'][symbol]
        probe = records['qualified_probe']['functions'][symbol]
        key = 'descriptor_sha256_excluding_relative_entry_offset' if symbol.endswith('.kd') else 'sha256'
        rows.append(dict(symbol=symbol, bytes=item['bytes'],
            observed_matches_prior=item[key] == original[key] and item['bytes'] == original['bytes'],
            observed_matches_qualified_probe=item[key] == probe[key] and item['bytes'] == probe['bytes'],
            comparison=key))
    report = dict(comparison_script_sha256=sha(Path(__file__)), sources=records, comparisons=rows,
        all_attention_instructions_and_descriptor_resources_match_prior=all(r['observed_matches_prior'] for r in rows),
        all_attention_instructions_and_descriptor_resources_match_probe=all(r['observed_matches_qualified_probe'] for r in rows),
        native_execution=False, inference_acceptance=False, performance_acceptance=False)
    with (D / 'comparison.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(report_sha256=sha(D / 'comparison.json'), comparisons=rows)))


if __name__ == '__main__':
    main()
