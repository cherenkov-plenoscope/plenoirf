from . import structure

import sparse_numeric_table as snt
import dynamicsizerecarray


def add_empty_level(evttab, level_key):
    level_structure = getattr(
        structure, "init_" + level_key + "_level_structure"
    )()
    level_dtype = structure.level_structure_to_dtype(
        level_structure=level_structure
    )
    evttab[level_key] = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=level_dtype
    )
    return evttab


def add_levels_from_path(evttab, path):
    add = snt.read(path=path)
    evttab.update(add)
    return evttab


def append_to_levels_from_path(evttab, path):
    add = snt.read(path=path)
    for level_key in add:
        evttab[level_key].append_recarray(add[level_key].to_recarray())
    return evttab


def write_certain_levels_to_path(evttab, path, level_keys):
    out = {}
    for level_key in level_keys:
        out[level_key] = evttab[level_key]
    write_all_levels_to_path(evttab=out, path=path)


def write_all_levels_to_path(evttab, path):
    snt.write(path=path, table=evttab)
