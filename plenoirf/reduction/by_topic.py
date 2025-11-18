import os
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


def make_paths(reduction_dirs):
    item_paths = {}
    for item in list_topic_filenames():
        item_paths[item] = []

    valid = {}
    for reduction_dir in reduction_dirs:
        valid[reduction_dir] = True

    for reduction_dir in reduction_dirs:
        for item in list_topic_filenames():
            item_path = os.path.join(reduction_dir, item)
            if not os.path.exists(item_path):
                valid[reduction_dir] = False

    for item in list_topic_filenames():
        for reduction_dir in reduction_dirs:
            if valid[reduction_dir]:
                item_path = os.path.join(reduction_dir, item)
                item_paths[item].append(item_path)

    return item_paths


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
                            payload=item.read(mode="b"),
                            mode="b",
                        )


def merge_ground_grid_intensity(out_path, in_paths, memory_config=None):
    mem = memory.make_config_if_None(memory_config)
    open_mem = utils.open_and_read_into_memory_when_small_enough

    with rnw.Path(out_path, use_tmp_dir=mem["use_tmp_dir"]) as tmp_path:
        with zipfile.open(file=tmp_path, mode="w") as zout:
            for in_path in in_paths:
                with open_mem(
                    path=in_path, mode="r", size=mem["read_buffer_size"]
                ) as fin:
                    with zipfile.open(file=fin, mode="r") as zin:
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
                with open(path=in_path, mode="rt") as fin:
                    fout.write(fin.read())
