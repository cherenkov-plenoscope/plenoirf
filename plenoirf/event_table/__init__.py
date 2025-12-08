from . import structure
from . import binned_by_pointing_zenith_and_primary_energy

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
    with snt.open(file=path, mode="r") as arc:
        add = arc.query()
    evttab.update(add)
    return evttab


def append_to_levels_from_path(evttab, path):
    with snt.open(file=path, mode="r") as arc:
        add = arc.query()
    evttab.append(add)
    return evttab


def write_certain_levels_to_path(evttab, path, level_keys):
    out = snt.SparseNumericTable(index_key=evttab.index_key)
    for level_key in level_keys:
        out[level_key] = evttab[level_key]
    write_all_levels_to_path(evttab=out, path=path)


def write_all_levels_to_path(evttab, path):
    with snt.open(
        path, mode="w", dtypes_and_index_key_from=evttab, compress=True
    ) as arc:
        arc.append_table(evttab)
