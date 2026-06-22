#!/usr/bin/env python3
"""Bake the standalone DeepSeek-V4-Flash MTP head (arch deepseek4_mtp_support,
tensors mtp.0.*) INTO the single-file FP4 model, producing one combined GGUF that
our TurboQuant deepseek4 loader can serve with --spec-type draft-mtp.

The FP4 model is a single file (no split metadata), so we can't in-place-patch a
shard set; we write a new combined file = all FP4 tensors (byte-identical, nsparks
names kept) + the MTP head renamed to blk.43.* / blk.43.nextn.* (our names — the
loader's name-alias is a *fallback*, so our-named block-43 tensors resolve directly)
+ nextn_predict_layers=1. file_type stays 41 so the FP4 loader adapter still applies.

Usage:
  python3 ds4_fp4_bake_mtp.py <fp4.gguf> <mtp.gguf> <out-combined.gguf> [--dry-run]
  (--dry-run writes header+KV+tensor-info only, skips the ~150GB tensor data copy)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gguf-py'))
import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

MTP_LAYER = 43  # FP4 has blocks 0..42; the MTP head becomes block 43 (the NextN layer)

# mtp.0.<base>.<suffix> -> blk.43.<base or renamed>.<suffix>
NEXTN_RENAME = {
    'e_proj':        'nextn.e_proj',
    'h_proj':        'nextn.h_proj',
    'enorm':         'nextn.enorm',
    'hnorm':         'nextn.hnorm',
    'norm':          'nextn.shared_head_norm',
    'hc_head_base':  'nextn.hc_head_base',
    'hc_head_fn':    'nextn.hc_head_fn',
    'hc_head_scale': 'nextn.hc_head_scale',
}

def map_mtp_name(name: str) -> str:
    assert name.startswith('mtp.0.'), f'unexpected MTP tensor name: {name}'
    rest = name[len('mtp.0.'):]
    base, suffix = rest.rsplit('.', 1)
    base = NEXTN_RENAME.get(base, base)
    return f'blk.{MTP_LAYER}.{base}.{suffix}'

def field_to_value(f):
    """Reconstruct (value, GGUFValueType, is_array, elem_type) from a reader field."""
    t = f.types[0]
    if t == GGUFValueType.ARRAY:
        elem_t = f.types[1]
        if elem_t == GGUFValueType.STRING:
            vals = [bytes(f.parts[di]).decode('utf-8') for di in f.data]
        else:
            vals = [f.parts[di][0].item() if hasattr(f.parts[di][0], 'item') else f.parts[di][0] for di in f.data]
        return vals, GGUFValueType.ARRAY, True, elem_t
    if t == GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode('utf-8'), t, False, None
    v = f.parts[f.data[0]][0]
    return (v.item() if hasattr(v, 'item') else v), t, False, None

def main():
    fp4, mtp, out = sys.argv[1], sys.argv[2], sys.argv[3]
    dry = '--dry-run' in sys.argv
    rf = GGUFReader(fp4)
    rm = GGUFReader(mtp)

    w = GGUFWriter(out, 'deepseek4')

    # ---- copy ALL FP4 KV (except the arch key the writer already set), set nextn=1 ----
    skip = {'general.architecture', 'general.alignment'}
    has_nextn = False
    for f in rf.fields.values():
        if f.name in skip or f.name.startswith('GGUF.'):  # GGUF.* are structural header pseudo-fields
            continue
        if f.name == 'deepseek4.nextn_predict_layers':
            w.add_uint32(f.name, 1); has_nextn = True; continue
        val, vt, is_arr, et = field_to_value(f)
        if is_arr:
            w.add_array(f.name, val)
        elif vt == GGUFValueType.STRING:
            w.add_string(f.name, val)
        elif vt == GGUFValueType.BOOL:
            w.add_bool(f.name, bool(val))
        else:
            w.add_key_value(f.name, val, vt)
    if not has_nextn:
        w.add_uint32('deepseek4.nextn_predict_layers', 1)

    # ---- tensor info: FP4 tensors (names kept) then MTP head (renamed to blk.43.*) ----
    for t in rf.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.n_bytes, t.tensor_type)
    for t in rm.tensors:
        nm = map_mtp_name(t.name)
        print(f'  {t.name:34s} -> {nm}')
        w.add_tensor_info(nm, t.data.shape, t.data.dtype, t.n_bytes, t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    if not dry:
        n = len(rf.tensors) + len(rm.tensors); i = 0
        for t in rf.tensors:
            w.write_tensor_data(t.data, tensor_endianess=rf.endianess); i += 1
            if i % 200 == 0: print(f'  data {i}/{n}', flush=True)
        for t in rm.tensors:
            w.write_tensor_data(t.data, tensor_endianess=rm.endianess); i += 1
    w.close()
    print(f'{"DRY-RUN " if dry else ""}wrote {out}: {len(rf.tensors)} FP4 + {len(rm.tensors)} MTP tensors, nextn=1')

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(1)
    main()
