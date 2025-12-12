import numpy as np
import os
import sparse_numeric_table as snt
import binning_utils
import tempfile
import rename_after_writing

from . import utils as search_index_utils

from .. import structure


def init(
    path,
    event_table_path,
    zenith_bin_edges,
    energy_bin_edges,
):
    zenith_bin_edges = np.asarray(zenith_bin_edges, dtype=np.float64)
    energy_bin_edges = np.asarray(energy_bin_edges, dtype=np.float64)

    assert binning_utils.is_strictly_monotonic_increasing(zenith_bin_edges)
    assert np.all(zenith_bin_edges >= 0)

    assert binning_utils.is_strictly_monotonic_increasing(energy_bin_edges)
    assert np.all(energy_bin_edges > 0)

    os.makedirs(path, exist_ok=True)

    search_index_utils.write_config(
        work_dir=path,
        zenith_bin_edges=zenith_bin_edges,
        energy_bin_edges=energy_bin_edges,
    )

    _populate(work_dir=path, event_table_path=event_table_path)


def _populate(work_dir, event_table_path, config=None):
    config = search_index_utils.read_config_if_None(
        work_dir=work_dir, config=config
    )
    bins_path = os.path.join(work_dir, "bins")

    with tempfile.TemporaryDirectory(prefix="plenoirf_") as tmp_dir:
        bin_assignment_path = os.path.join(
            tmp_dir, "uid_zenith_energy_bin.uid-u8.zd-u1.en-u1"
        )
        make_bin_assignment(
            event_table_path=event_table_path,
            zenith_bin_edges=config["zenith_bin"]["edges"],
            energy_bin_edges=config["energy_bin"]["edges"],
            bin_assignment_path=bin_assignment_path,
        )

        tmp_stage_path = os.path.join(tmp_dir, "stage")
        _populated_bins_step_one(
            event_table_path=event_table_path,
            bin_assignment_path=bin_assignment_path,
            out_path=tmp_stage_path,
            num_zenith_bins=config["zenith_bin"]["num"],
            num_energy_bins=config["energy_bin"]["num"],
        )

        _populated_bins_step_two(
            out_path=bins_path,
            stage_path=tmp_stage_path,
            num_zenith_bins=config["zenith_bin"]["num"],
            num_energy_bins=config["energy_bin"]["num"],
            dtypes=structure.dtypes(),
        )


def _assign_event_table_to_bins(
    event_table_path,
    level_key,
    column_key,
    bin_edges,
    bin_assignment_path,
):
    num_bins = len(bin_edges) - 1
    assert num_bins < np.iinfo("u1").max

    lvlcol_key = level_key + "/" + column_key

    with rename_after_writing.open(
        bin_assignment_path, mode="wb"
    ) as fout, snt.open(event_table_path, mode="r") as reader:
        for block in snt._file_io.LevelBlockLooper(
            reader=reader, level_key=level_key
        ):
            tmp = np.recarray(
                shape=block.shape[0],
                dtype=[("uid", "u8"), (lvlcol_key, "u1")],
            )
            tmp["uid"] = block["uid"]
            tmp[lvlcol_key] = binning_utils.assign_to_bins(
                bin_edges=bin_edges,
                x=block[column_key],
            )
            assert np.all(
                tmp[lvlcol_key] >= 0
            ), "Did not expect under- or overflow."
            fout.write(tmp.tobytes())


def _merge_bin_assignments(
    zenith_bin_assignment_path,
    energy_bin_assignment_path,
    out_path,
):
    with open(zenith_bin_assignment_path, "rb") as f:
        uid_zd = np.frombuffer(f.read(), dtype=[("uid", "u8"), ("zd", "u1")])
        _sort = np.argsort(uid_zd["uid"])
        uid_zd = uid_zd[_sort]
        del _sort

    with open(energy_bin_assignment_path, "rb") as f:
        uid_en = np.frombuffer(f.read(), dtype=[("uid", "u8"), ("en", "u1")])
        _sort = np.argsort(uid_en["uid"])
        uid_en = uid_en[_sort]
        del _sort

    np.testing.assert_array_equal(uid_zd["uid"], uid_en["uid"])
    uid_zd_en = np.recarray(
        shape=uid_zd.shape[0],
        dtype=get_bin_assignment_dtype(),
    )
    uid_zd_en["uid"] = uid_zd["uid"]
    uid_zd_en["zd"] = uid_zd["zd"]
    uid_zd_en["en"] = uid_en["en"]

    with rename_after_writing.open(out_path, "wb") as f:
        f.write(uid_zd_en.tobytes())


def make_bin_assignment(
    event_table_path,
    zenith_bin_edges,
    energy_bin_edges,
    bin_assignment_path,
):
    with tempfile.TemporaryDirectory(prefix="plenoirf_") as tmp_dir:
        zenith_bin_assignment_path = os.path.join(
            tmp_dir, "uid_zenith_bin.recarray"
        )
        _assign_event_table_to_bins(
            event_table_path=event_table_path,
            level_key="instrument_pointing",
            column_key="zenith_rad",
            bin_edges=zenith_bin_edges,
            bin_assignment_path=zenith_bin_assignment_path,
        )
        energy_bin_assignment_path = os.path.join(
            tmp_dir, "uid_energy_bin.recarray"
        )
        _assign_event_table_to_bins(
            event_table_path=event_table_path,
            level_key="primary",
            column_key="energy_GeV",
            bin_edges=energy_bin_edges,
            bin_assignment_path=energy_bin_assignment_path,
        )
        _merge_bin_assignments(
            zenith_bin_assignment_path=zenith_bin_assignment_path,
            energy_bin_assignment_path=energy_bin_assignment_path,
            out_path=bin_assignment_path,
        )


class FileBins:
    def __init__(self, path, num_zenith_bins, num_energy_bins):
        self.path = path
        self.num_zenith_bins = num_zenith_bins
        self.num_energy_bins = num_energy_bins
        assert self.num_zenith_bins > 0
        assert self.num_energy_bins > 0

    def __enter__(self):
        os.makedirs(self.path, exist_ok=True)
        self.fbins = {}
        for zd in range(self.num_zenith_bins):
            self.fbins[zd] = {}
            for en in range(self.num_energy_bins):
                _, tmp_fpath = self._fpath_and_tmp_fpath(zd=zd, en=en)
                self.fbins[zd][en] = open(tmp_fpath, mode="wb")
        return self.fbins

    def _fpath_and_tmp_fpath(self, zd, en):
        fpath = os.path.join(self.path, f"zd{zd:06d}_en{en:06d}.recarray")
        return fpath, fpath + ".incomplete"

    def __exit__(self, type, value, traceback):
        for zd in self.fbins:
            for en in self.fbins[zd]:
                self.fbins[zd][en].close()
                fpath, tmp_fpath = self._fpath_and_tmp_fpath(zd=zd, en=en)
                os.rename(src=tmp_fpath, dst=fpath)

    def __repr__(self):
        return f"{self.__class__.__name__:s}()"


def _populated_bins_step_one(
    event_table_path,
    bin_assignment_path,
    out_path,
    num_zenith_bins,
    num_energy_bins,
):
    assert num_energy_bins > 0
    assert num_zenith_bins > 0

    os.makedirs(out_path, exist_ok=True)

    with open(bin_assignment_path, "rb") as f:
        bin_assignment = np.frombuffer(
            f.read(),
            dtype=get_bin_assignment_dtype(),
        )

    with snt.open(event_table_path, mode="r") as event_table_reader:
        level_keys = list(event_table_reader.dtypes.keys())

    for level_key in level_keys:
        level_dir = os.path.join(out_path, level_key)

        if os.path.exists(level_dir):
            continue

        with FileBins(
            path=level_dir,
            num_zenith_bins=num_zenith_bins,
            num_energy_bins=num_energy_bins,
        ) as fbins:
            with snt.open(event_table_path, mode="r") as event_table_reader:
                for level_block in snt._file_io.LevelBlockLooper(
                    reader=event_table_reader, level_key=level_key
                ):
                    iii = np.searchsorted(
                        a=bin_assignment["uid"], v=level_block["uid"]
                    )
                    addr = bin_assignment[iii]

                    for zd in range(num_zenith_bins):
                        zd_mask = addr["zd"] == zd
                        for en in range(num_energy_bins):
                            ene_mask = addr["en"] == en
                            bin_mask = np.logical_and(zd_mask, ene_mask)
                            bin_part = level_block[bin_mask]
                            fbins[zd][en].write(bin_part.tobytes())


def _populated_bins_step_two(
    out_path,
    stage_path,
    num_zenith_bins,
    num_energy_bins,
    dtypes,
):
    assert num_energy_bins > 0
    assert num_zenith_bins > 0
    os.makedirs(out_path, exist_ok=True)

    for zd in range(num_zenith_bins):
        for en in range(num_energy_bins):

            binname = f"zd{zd:06d}_en{en:06d}"
            bin_outpath = os.path.join(out_path, binname + ".snt.zip")

            if not os.path.exists(bin_outpath):
                with rename_after_writing.Path(
                    bin_outpath
                ) as tmp_bin_outpath, snt.open(
                    tmp_bin_outpath,
                    mode="w",
                    dtypes=dtypes,
                    index_key="uid",
                    compress=False,
                ) as event_table_writer:
                    for level_key in dtypes:

                        bin_level_inpath = os.path.join(
                            stage_path,
                            level_key,
                            binname + ".recarray",
                        )

                        with open(bin_level_inpath, mode="rb") as fin:
                            level_recarray = np.frombuffer(
                                fin.read(),
                                dtype=dtypes[level_key],
                            )
                        _table = snt.SparseNumericTable(
                            dtypes=dtypes,
                            index_key="uid",
                        )
                        _table[level_key].append(level_recarray)
                        event_table_writer.append_table(_table)


def get_bin_assignment_dtype():
    return [
        ("uid", "u8"),
        ("zd", "u1"),
        ("en", "u1"),
    ]
