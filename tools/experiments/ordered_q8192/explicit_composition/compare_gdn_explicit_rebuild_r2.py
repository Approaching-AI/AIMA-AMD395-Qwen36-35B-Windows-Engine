"""Recheck existing code objects; preserve the earlier SHT_NOBITS misread."""
from pathlib import Path
import hashlib
import json
import struct
import subprocess

B = Path(__file__).resolve().parent
D = B / "qrt-gdn-explicit-runtime-rebuild-20260923-r1"
C = D / "collected"
G = Path("/Users/jiawei-macmini/projects/AIMA-explicit-gdn-layout-candidate")
sha = lambda p: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()


def inspect(raw):
    assert raw[:6] == b"\x7fELF\x02\x01"
    h = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
    offset, step, count, string_index = h[6], h[11], h[12], h[13]
    assert step == 64 and 0 < count < 100 and offset + count * step <= len(raw)
    sections = [struct.unpack_from("<IIQQQQIIQQ", raw, offset + i * step) for i in range(count)]
    names = sections[string_index]
    strings = raw[names[4]:names[4] + names[5]]
    allocated = {}
    for s in sections:
        if not (s[2] & 2):
            continue
        name = strings[s[0]:].split(b"\0", 1)[0].decode()
        assert name not in allocated
        item = dict(type=s[1], flags=s[2], address=s[3], bytes=s[5],
                    link=s[6], info=s[7], alignment=s[8], entry_bytes=s[9])
        if s[1] == 8:  # SHT_NOBITS occupies memory, not bytes at sh_offset.
            item.update(storage="zero_fill", file_bytes=0)
        else:
            assert s[4] + s[5] <= len(raw)
            item.update(storage="file", file_bytes=s[5],
                        sha256=hashlib.sha256(raw[s[4]:s[4] + s[5]]).hexdigest())
        allocated[name] = item
    assert ".text" in allocated and h[9] == 56 and h[5] + h[9] * h[10] <= len(raw)
    programs = [struct.unpack_from("<IIQQQQQQ", raw, h[5] + i * h[9]) for i in range(h[10])]
    loaded = []
    for p in programs:
        if p[0] != 1:
            continue
        data = bytearray(raw[p[2]:p[2] + p[5]])
        assert len(data) == p[5] and p[6] >= p[5]
        # The sole permitted file-backed difference is ELF e_shoff, which
        # locates the nonloaded section-header table after variable debug paths.
        if p[2] <= 40 and p[2] + p[5] >= 48:
            data[40 - p[2]:48 - p[2]] = b"\0" * 8
        loaded.append(dict(program_header=p, file_sha256_excluding_e_shoff=hashlib.sha256(data).hexdigest(),
                           zero_fill_bytes=p[6] - p[5]))
    header = list(h)
    header[0] = header[0].hex()
    header[6] = "excluded_section_header_offset"
    return dict(header_excluding_e_shoff=header, program_headers=programs,
                allocated_sections=allocated, load_segments_excluding_e_shoff=loaded)


def main():
    selected = json.loads((G / "native/linux_core_port/gb10_gdn_ordered_compile.json").read_text())
    rebuilt = json.loads((C / "aot/result.json").read_text())
    rows = []
    checks = []
    for name, entry in rebuilt["compiled"].items():
        old = selected["compiled"][name]
        for key in ("signature", "constants", "options"):
            assert entry[key] == old[key]
        for key in ("name", "shared", "num_warps", "warp_size"):
            assert entry["metadata"][key] == old["metadata"][key]
        a, b = C / "aot" / entry["file"], B / "linux-core-gdn-explicit-asset-import-r1" / old["file"]
        left, right = inspect(a.read_bytes()), inspect(b.read_bytes())
        command = ["/usr/bin/objdump", "--section-headers", str(a)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0
        bss = [line for line in result.stdout.splitlines() if ".relro_padding" in line]
        assert len(bss) == 1 and "BSS" in bss[0]
        checks.append(dict(name=name, command=command, returncode=0, bss_line=bss[0]))
        rows.append(dict(name=name, rebuilt_sha256=sha(a), selected_sha256=sha(b),
                         all_file_bytes_match=sha(a) == sha(b), rebuilt=left, selected=right,
                         executable_layout_and_content_match=left == right))
    # A NOBITS offset need not identify readable file data. Changing only its
    # offset must preserve memory semantics, while any code-byte change must fail.
    original = (C / "aot/kkt.hsaco").read_bytes()
    h = struct.unpack_from("<16sHHIQQQIHHHHHH", original)
    variant = bytearray(original)
    found = False
    for i in range(h[12]):
        pos = h[6] + i * h[11]
        s = struct.unpack_from("<IIQQQQIIQQ", original, pos)
        if s[1] == 8 and s[2] & 2:
            struct.pack_into("<Q", variant, pos + 24, len(original) + 4096)
            found = True
    assert found and inspect(variant) == inspect(original)
    for i in range(h[12]):
        s = struct.unpack_from("<IIQQQQIIQQ", original, h[6] + i * h[11])
        if s[2] & 4 and s[2] & 2:
            variant = bytearray(original)
            variant[s[4]] ^= 1
            assert inspect(variant) != inspect(original)
            break
    else:
        raise AssertionError("No executable section")
    report = dict(original_failed_report_sha256=sha(D / "image-comparison.json"),
                  original_report_preserved=True,
                  correction="SHT_NOBITS has zero-filled memory semantics and no file content at sh_offset",
                  source_files_match=rebuilt["source_files"] == selected["source_files"],
                  compiler_sha256=sha(G / "tools/compile_linux_core_gdn_explicit.py"),
                  comparison_script_sha256=sha(Path(__file__)),
                  dispatch_sha256=sha(C / "dispatch.json"),
                  independent_objdump=checks, nobits_out_of_file_control_pass=True,
                  code_mutation_detected=True, signatures_constants_launches_match=True,
                  rows=rows, executable_layout_and_content_match=all(
                      row["executable_layout_and_content_match"] for row in rows),
                  ignored_loaded_bytes="Only eight-byte ELF e_shoff, at file offset 40",
                  product_images_changed=False, native_execution=False,
                  inference_acceptance=False, performance_acceptance=False)
    path = D / "image-comparison-r2.json"
    with path.open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")
    assert report["source_files_match"] and report["executable_layout_and_content_match"]
    print(json.dumps(dict(report=str(path), sha256=sha(path), images=len(rows),
                         executable_layout_and_content_match=True,
                         nobits_out_of_file_control_pass=True, code_mutation_detected=True)))


if __name__ == "__main__":
    main()
