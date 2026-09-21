#!/usr/bin/env python3
"""Verify embedded GPU image bytes and symbols in a Windows AMD64 COFF object."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def verify_coff(data, images):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(len(data) >= 20, "Truncated COFF header")
    machine, count, _, symtab, symbols, optional, _ = struct.unpack_from("<HHIIIHH", data)
    require(machine == 0x8664 and optional == 0 and 1 <= count <= 16, "Not a plain AMD64 COFF object")
    require(20 + count * 40 <= len(data), "Truncated section table")
    sections = []
    for i in range(count):
        header = struct.unpack_from("<8sIIIIIIHHI", data, 20 + i * 40)
        name, _, _, size, start, _, _, relocations, _, flags = header
        require(start + size <= len(data), "Section exceeds object extent")
        sections.append(dict(name=name.rstrip(b"\0"), size=size, start=start,
                             flags=flags, relocations=relocations))
    strings = symtab + symbols * 18
    require(symtab >= 20 + count * 40 and strings + 4 <= len(data), "Invalid symbol table")
    string_size, = struct.unpack_from("<I", data, strings)
    require(string_size >= 4 and strings + string_size <= len(data), "Invalid string table")
    found = {}
    i = 0
    while i < symbols:
        offset = symtab + i * 18
        name = data[offset:offset + 8]
        if name[:4] == b"\0" * 4:
            position, = struct.unpack_from("<I", name, 4)
            require(4 <= position < string_size, "Invalid symbol name offset")
            end = data.find(b"\0", strings + position, strings + string_size)
            require(end >= 0, "Unterminated symbol name")
            name = data[strings + position:end]
        else:
            name = name.rstrip(b"\0")
        value, section, _, storage, aux = struct.unpack_from("<IhHBB", data, offset + 8)
        require(i + aux < symbols, "Invalid auxiliary symbol extent")
        if name.startswith(b"_binary_"):
            require(storage == 2 and aux == 0 and 1 <= section <= count, "Invalid image symbol")
            label = name.decode("ascii")
            require(label not in found, "Duplicate image symbol")
            found[label] = (value, sections[section - 1])
        i += 1 + aux
    require(set(found) == {x["symbol"] for x in images}, "Embedded image symbol set differs")
    spans = []
    for image in images:
        position, section = found[image["symbol"]]
        require(section["name"] == b".rdata" and section["flags"] & 0x40000040 == 0x40000040,
                "Image is not in readable initialized data")
        require(section["flags"] & 0xA0000020 == 0 and section["relocations"] == 0,
                "Image section is writable, executable or relocated")
        require(position % 256 == 0 and position + image["bytes"] <= section["size"],
                "Image alignment or extent differs")
        start = section["start"] + position
        require(hashlib.sha256(data[start:start + image["bytes"]]).hexdigest() == image["sha256"],
                "Embedded GPU image hash differs")
        spans.append((start, start + image["bytes"]))
    spans.sort()
    require(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "Embedded GPU images overlap")
    return dict(machine="AMD64", images=len(images), image_bytes=sum(x["bytes"] for x in images),
                object_bytes=len(data), object_sha256=hashlib.sha256(data).hexdigest(),
                all_image_bytes_and_symbols_match=True, gpu_executed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_coff(args.object.read_bytes(), json.loads(args.prepare.read_text())["images"])))


if __name__ == "__main__":
    main()
