import os
import zipfile

import rename_after_writing as rnw
import sparse_numeric_table as snt
import sequential_tar

from .. import utils
from . import memory


def list_topic_filenames():
    return [
        "event_table.snt.zip",
        "reconstructed_cherenkov.loph.tar",
        "ground_grid_intensity.zip",
        "ground_grid_intensity_roi.zip",
        "benchmark.snt.zip",
        "event_uids_for_debugging.txt",
    ]


def merge_topic(
    topic_key,
    out_path,
    in_paths,
    memory_config=None,
):
    args = {
        "out_path": out_path,
        "in_paths": in_paths,
        "memory_config": memory.make_config_if_None(memory_config),
    }

    if topic_key == "event_table.snt.zip":
        merge_event_table(**args)
    elif topic_key == "reconstructed_cherenkov.loph.tar":
        merge_reconstructed_cherenkov(**args)
    elif topic_key == "ground_grid_intensity.zip":
        merge_ground_grid_intensity(**args)
    elif topic_key == "ground_grid_intensity_roi.zip":
        merge_ground_grid_intensity(**args)
    elif topic_key == "benchmark.snt.zip":
        merge_benchmark(**args)
    elif topic_key == "event_uids_for_debugging.txt":
        merge_event_uids_for_debugging(**args)
    else:
        raise KeyError(f"No such topic '{topic_key:s}'.")


def merge_event_table(out_path, in_paths, memory_config=None):
    mem = memory.make_config_if_None(memory_config)

    open_file_function = (
        lambda path, mode: utils.open_and_read_into_memory_when_small_enough(
            path=path,
            mode=mode,
            size=mem["read_buffer_size"],
        )
    )

    with rnw.Path(out_path, use_tmp_dir=mem["use_tmp_dir"]) as tmp_path:
        snt.files.merge(
            out_path=tmp_path,
            in_paths=in_paths,
            open_file_function=open_file_function,
        )


def merge_reconstructed_cherenkov(out_path, in_paths, memory_config=None):
    mem = memory.make_config_if_None(memory_config)

    with rnw.Path(out_path, use_tmp_dir=mem["use_tmp_dir"]) as tmp_path:
        with sequential_tar.open(tmp_path, mode="w") as tarout:
            for in_path in in_paths:
                with sequential_tar.open(in_path, mode="r") as tarin:
                    for item in tarin:
                        tarout.write(
                            name=item.name,
                            payload=item.read(mode="rb"),
                            mode="wb",
                        )


def merge_ground_grid_intensity(out_path, in_paths, memory_config=None):
    mem = memory.make_config_if_None(memory_config)
    open_mem = utils.open_and_read_into_memory_when_small_enough

    with rnw.Path(out_path, use_tmp_dir=mem["use_tmp_dir"]) as tmp_path:
        with zipfile.ZipFile(file=tmp_path, mode="w") as zout:
            for in_path in in_paths:
                with open_mem(
                    path=in_path, mode="rb", size=mem["read_buffer_size"]
                ) as fin:
                    with zipfile.ZipFile(file=fin, mode="r") as zin:
                        for item in zin.filelist:
                            with zout.open(item.filename, mode="w") as fo:
                                with zin.open(item.filename, mode="r") as fi:
                                    fo.write(fi.read())


def merge_benchmark(out_path, in_paths, memory_config=None):
    mem = memory.make_config_if_None(memory_config)

    open_file_function = (
        lambda path, mode: utils.open_and_read_into_memory_when_small_enough(
            path=path,
            mode=mode,
            size=mem["read_buffer_size"],
        )
    )

    with rnw.Path(out_path, use_tmp_dir=mem["use_tmp_dir"]) as tmp_path:
        snt.files.merge(
            out_path=tmp_path,
            in_paths=in_paths,
            open_file_function=open_file_function,
        )


def merge_event_uids_for_debugging(out_path, in_paths, memory_config=None):
    mem = memory.make_config_if_None(memory_config)

    with rnw.Path(out_path, use_tmp_dir=mem["use_tmp_dir"]) as tmp_path:
        with open(tmp_path, "wt") as fout:
            for in_path in in_paths:
                with open(in_path, "rt") as fin:
                    fout.write(fin.read())
