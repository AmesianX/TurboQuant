#!/usr/bin/env python3
"""Convert antirez's DeepSeek-V4-Flash MTP side GGUF (arch deepseek4_mtp_support,
tensors mtp.0.*) into a third split shard for our IQ2_XS-XL main model, renaming
tensors to the blk.43.* / blk.43.nextn.* names the TurboQuant deepseek4 loader
expects. Raw tensor data is copied byte-identical (no requantization).

Also patches shard 1's split bookkeeping in place (split.count 2->3,
split.tensors.count 1328->1360) — both fixed-width values, so no rewrite of the
82 GB shards. --revert undoes the patch.

Usage:
  python3 ds4_mtp_to_shard.py convert <mtp.gguf> <out-00003-of-00003.gguf>
  python3 ds4_mtp_to_shard.py patch   <shard1.gguf> [--revert]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gguf-py'))
import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter

MTP_LAYER = 43  # main model has blocks 0..42; MTP head is the extra layer

# mtp.0.<x> -> blk.43.nextn.<y> for the MTP-specific tensors; everything else
# maps to the standard per-layer name blk.43.<x> (matches blk.0..42 naming).
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

N_TENSORS_MAIN = 1328   # shard1 (752) + shard2 (576)
SPLIT_COUNT_OLD, SPLIT_COUNT_NEW = 1, 2


def map_name(name: str) -> str:
    assert name.startswith('mtp.0.'), f'unexpected tensor name: {name}'
    rest = name[len('mtp.0.'):]                      # e.g. 'attn_q_a.weight'
    base, suffix = rest.rsplit('.', 1)               # ('attn_q_a', 'weight')
    base = NEXTN_RENAME.get(base, base)
    return f'blk.{MTP_LAYER}.{base}.{suffix}'


def convert(src: str, dst: str) -> None:
    reader = GGUFReader(src)
    n = len(reader.tensors)
    writer = GGUFWriter(dst, 'deepseek4')
    writer.add_uint16('split.no', 1)
    writer.add_uint16('split.count', SPLIT_COUNT_NEW)
    writer.add_int32('split.tensors.count', N_TENSORS_MAIN + n)

    for t in reader.tensors:
        new_name = map_name(t.name)
        print(f'  {t.name:34s} -> {new_name:44s} {t.tensor_type.name}')
        writer.add_tensor_info(new_name, t.data.shape, t.data.dtype, t.n_bytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    print(f'wrote {dst}: {n} tensors')


def _field_value_offset(reader: GGUFReader, key: str) -> tuple[int, np.ndarray]:
    byte_bounds = np.lib.array_utils.byte_bounds if hasattr(np.lib, 'array_utils') else np.byte_bounds
    f = reader.fields[key]
    part = f.parts[f.data[0]]
    base = byte_bounds(reader.data)[0]
    return byte_bounds(part)[0] - base, part


def patch(shard1: str, revert: bool) -> None:
    reader = GGUFReader(shard1)  # read-only mmap, used just to locate offsets
    off_cnt, part_cnt = _field_value_offset(reader, 'split.count')
    off_ten, part_ten = _field_value_offset(reader, 'split.tensors.count')
    cur_cnt, cur_ten = int(part_cnt[0]), int(part_ten[0])
    dt_cnt, dt_ten = part_cnt.dtype, part_ten.dtype
    del reader

    if revert:
        want_cnt_from, want_cnt_to = SPLIT_COUNT_NEW, SPLIT_COUNT_OLD
        want_ten_to = N_TENSORS_MAIN
    else:
        want_cnt_from, want_cnt_to = SPLIT_COUNT_OLD, SPLIT_COUNT_NEW
        want_ten_to = N_TENSORS_MAIN + 32
    if cur_cnt != want_cnt_from:
        print(f'split.count is {cur_cnt}, expected {want_cnt_from} — nothing to do')
        return

    with open(shard1, 'r+b') as fh:
        fh.seek(off_cnt); fh.write(np.array([want_cnt_to], dtype=dt_cnt).tobytes())
        fh.seek(off_ten); fh.write(np.array([want_ten_to], dtype=dt_ten).tobytes())
    print(f'patched {Path(shard1).name}: split.count {cur_cnt}->{want_cnt_to}, '
          f'split.tensors.count {cur_ten}->{want_ten_to}')


if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == 'convert':
        convert(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[1] == 'patch':
        patch(sys.argv[2], revert='--revert' in sys.argv)
    else:
        sys.exit(__doc__)
