from pathlib import Path
import struct

base = Path("/root/.julia/juliaup")
for path in sorted(base.glob("julia-*/lib/julia/*.so")):
    data = path.read_bytes()
    if data[:4] != b"\x7fELF":
        continue

    ei_class = data[4]
    ei_data = data[5]
    endian = "<" if ei_data == 1 else ">"

    if ei_class == 1:
        phoff = struct.unpack(endian + "I", data[28:32])[0]
        phentsize = struct.unpack(endian + "H", data[42:44])[0]
        phnum = struct.unpack(endian + "H", data[44:46])[0]
        flags_offset = 24
    elif ei_class == 2:
        phoff = struct.unpack(endian + "Q", data[32:40])[0]
        phentsize = struct.unpack(endian + "H", data[54:56])[0]
        phnum = struct.unpack(endian + "H", data[56:58])[0]
        flags_offset = 4
    else:
        continue

    changed = False
    for i in range(phnum):
        off = phoff + i * phentsize
        if struct.unpack(endian + "I", data[off:off+4])[0] == 0x6474e551:
            flags = struct.unpack(endian + "I", data[off+flags_offset:off+flags_offset+4])[0]
            if flags & 1:
                data = data[:off+flags_offset] + struct.pack(endian + "I", flags & ~1) + data[off+flags_offset+4:]
                changed = True

    if changed:
        path.write_bytes(data)
        print("patched", path)
