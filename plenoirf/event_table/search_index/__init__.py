import numpy as np
import os
import sparse_numeric_table as snt
import binning_utils
import json_utils
import tempfile
import rename_after_writing

from ... import utils as plenoirf_utils
from .. import structure


class EventTable:
    def __init__(self, path):
        self.path = path
        self.config = read_config_if_None(work_dir=path, config=None)

    def query(
        self,
        energy_bin_indices=None,
        energy_start_GeV=None,
        energy_stop_GeV=None,
        zenith_bin_indices=None,
        zenith_start_rad=None,
        zenith_stop_rad=None,
        indices=None,
        levels_and_columns=None,
        sort=False,
        return_tasks=False,
    ):
        tasks = _make_list_of_zenith_energy_bins_to_be_read(
            energy_bin_edges_GeV=self.config["energy_bin"]["edges"],
            zenith_bin_edges_rad=self.config["zenith_bin"]["edges"],
            energy_bin_indices=energy_bin_indices,
            energy_start_GeV=energy_start_GeV,
            energy_stop_GeV=energy_stop_GeV,
            zenith_bin_indices=zenith_bin_indices,
            zenith_start_rad=zenith_start_rad,
            zenith_stop_rad=zenith_stop_rad,
        )

        out = snt.SparseNumericTable(index_key="uid")

        for task in tasks:
            zd = task["zenith"]["bin_index"]
            en = task["energy"]["bin_index"]
            bin_path = os.path.join(
                self.path, "bins", f"zd{zd:06d}_en{en:06d}.snt.zip"
            )
            with snt.open(bin_path, "r") as reader:
                event_table_zd_en_bin = reader.query(
                    levels_and_columns=levels_and_columns,
                )
                if task["energy"]["cut"] is not None:
                    uid_energy = _get_uid_in_energy_range(
                        reader=reader,
                        energy_start_GeV=task["energy"]["cut"]["start"],
                        energy_stop_GeV=task["energy"]["cut"]["stop"],
                    )
                else:
                    uid_energy = None

                if task["zenith"]["cut"] is not None:
                    uid_zenith = _get_uid_in_zenith_range(
                        reader=reader,
                        zenith_start_rad=zenith_start_rad,
                        zenith_stop_rad=zenith_stop_rad,
                    )
                else:
                    uid_zenith = None

            uid_common = _get_uid_common(
                uid_zenith=uid_zenith, uid_energy=uid_energy
            )

            if uid_common is not None:
                event_table_zd_en_bin = snt.logic.cut_table_on_indices(
                    table=event_table_zd_en_bin,
                    common_indices=uid_common,
                    inplace=True,
                )

            out.append(event_table_zd_en_bin)

        if sort:
            out = snt.logic.cut_and_sort_table_on_indices(
                table=out,
                common_indices=indices,
                inplace=True,
            )

        if return_tasks:
            return out, tasks
        else:
            return out

    def __repr__(self):
        return f"{self.__class__.__name__:s}()"

    def assert_valid(self):
        num_en = self.config["energy_bin"]["num"]
        num_zd = self.config["zenith_bin"]["num"]

        for en in range(num_en):
            energy_start_GeV = self.config["energy_bin"]["edges"][en]
            energy_stop_GeV = self.config["energy_bin"]["edges"][en + 1]

            for zd in range(num_zd):
                zenith_start_GeV = self.config["zenith_bin"]["edges"][zd]
                zenith_stop_GeV = self.config["zenith_bin"]["edges"][zd + 1]

                print(f"zd: {zd:d}/{num_zd:d}, en: {en:d}/{num_en:d}")
                part = self.query(
                    energy_bin_indices=en,
                    zenith_bin_indices=zd,
                    levels_and_columns={
                        "primary": ("uid", "energy_GeV"),
                        "instrument_pointing": ("uid", "zenith_rad"),
                    },
                )

                assert np.all(
                    np.logical_and(
                        energy_start_GeV <= part["primary"]["energy_GeV"],
                        part["primary"]["energy_GeV"] < energy_stop_GeV,
                    )
                )
                assert np.all(
                    np.logical_and(
                        zenith_start_GeV
                        <= part["instrument_pointing"]["zenith_rad"],
                        part["instrument_pointing"]["zenith_rad"]
                        < zenith_stop_GeV,
                    )
                )
                for level_key in structure.dtypes():
                    other = self.query(
                        energy_bin_indices=en,
                        zenith_bin_indices=zd,
                        levels_and_columns={level_key: "__all__"},
                    )
                    assert np.all(
                        np.isin(
                            element=other[level_key]["uid"],
                            test_elements=part["primary"]["uid"],
                            assume_unique=True,
                        )
                    ), (
                        f"Expected all uid from level '{level_key:s}' "
                        "are also in level 'primary'."
                    )


def _get_uid_common(uid_zenith, uid_energy):
    if uid_energy is None and uid_zenith is None:
        uid_common = None
    elif uid_zenith is not None:
        uid_common = uid_zenith
    elif uid_energy is not None:
        uid_common = uid_energy
    else:
        uid_common = snt.logic.intersection(uid_zenith, uid_energy)
    return uid_common


def _get_uid_in_energy_range(reader, energy_start_GeV, energy_stop_GeV):
    tab = reader.query(levels_and_columns={"primary": ("uid", "energy_GeV")})
    level = tab["primary"]
    greater_equal_start = energy_start_GeV <= level["energy_GeV"]
    less_stop = level["energy_GeV"] < energy_stop_GeV
    mask = np.logical_and(greater_equal_start, less_stop)
    return level["uid"][mask]


def _get_uid_in_zenith_range(reader, zenith_start_rad, zenith_stop_rad):
    tab = reader.query(
        levels_and_columns={"instrument_pointing": ("uid", "zenith_rad")}
    )
    level = tab["instrument_pointing"]
    greater_equal_start = zenith_start_rad <= level["zenith_rad"]
    less_stop = level["zenith_rad"] < zenith_stop_rad
    mask = np.logical_and(greater_equal_start, less_stop)
    return level["uid"][mask]


def _make_list_of_zenith_energy_bins_to_be_read(
    energy_bin_edges_GeV,
    zenith_bin_edges_rad,
    energy_bin_indices=None,
    energy_start_GeV=None,
    energy_stop_GeV=None,
    zenith_bin_indices=None,
    zenith_start_rad=None,
    zenith_stop_rad=None,
):
    zenith_bins, zenith_fully = _make_list_of_bins_to_be_read(
        bin_edges=zenith_bin_edges_rad,
        bin_indices=zenith_bin_indices,
        start=zenith_start_rad,
        stop=zenith_stop_rad,
    )

    energy_bins, energy_fully = _make_list_of_bins_to_be_read(
        bin_edges=energy_bin_edges_GeV,
        bin_indices=energy_bin_indices,
        start=energy_start_GeV,
        stop=energy_stop_GeV,
    )

    tasks = []
    for zzz in range(len(zenith_bins)):
        for eee in range(len(energy_bins)):
            task = {}
            task["zenith"] = {}
            task["zenith"]["bin_index"] = zenith_bins[zzz]
            if zenith_fully[zzz]:
                task["zenith"]["cut"] = None
            else:
                task["zenith"]["cut"] = {
                    "start": zenith_start_rad,
                    "stop": zenith_stop_rad,
                }
            task["energy"] = {}
            task["energy"]["bin_index"] = energy_bins[eee]
            if energy_fully[eee]:
                task["energy"]["cut"] = None
            else:
                task["energy"]["cut"] = {
                    "start": energy_start_GeV,
                    "stop": energy_stop_GeV,
                }
            tasks.append(task)

    return tasks


def _make_list_of_bins_to_be_read(
    bin_edges,
    bin_indices=None,
    start=None,
    stop=None,
):
    if bin_indices is not None:
        assert start is None and stop is None, (
            f"When 'bin_indices' is given, "
            f"there must be no 'start' or 'stop'."
        )
        num_bins = len(bin_edges) - 1

        if hasattr(bin_indices, "__len__"):
            out = []
            fully_contained = []
            for bin_index in bin_indices:
                assert plenoirf_utils.can_be_interpreted_as_int(bin_index)
                assert 0 <= bin_index < num_bins
                out.append(bin_index)
                fully_contained.append(True)
            return out, fully_contained
        elif plenoirf_utils.can_be_interpreted_as_int(bin_indices):
            assert 0 <= bin_indices < num_bins
            return [bin_indices], [True]
        else:
            raise AttributeError(f"Can not interpret '{key:s}_bin_index'.")

    else:
        assert start is not None and stop is not None

        return binning_utils.find_bin_indices_in_start_stop_range(
            bin_edges=bin_edges,
            start=start,
            stop=stop,
            return_if_bin_is_fully_contained=True,
        )


def get_bin_assignment_dtype():
    return [
        ("uid", "u8"),
        ("instrument_pointing/zenith_rad", "u1"),
        ("primary/energy_GeV", "u1"),
    ]


def init(
    work_dir,
    zenith_bin_edges,
    energy_bin_edges,
):
    zenith_bin_edges = np.asarray(zenith_bin_edges, dtype=np.float64)
    energy_bin_edges = np.asarray(energy_bin_edges, dtype=np.float64)

    assert binning_utils.is_strictly_monotonic_increasing(zenith_bin_edges)
    assert np.all(zenith_bin_edges >= 0)

    assert binning_utils.is_strictly_monotonic_increasing(energy_bin_edges)
    assert np.all(energy_bin_edges > 0)

    os.makedirs(work_dir, exist_ok=True)
    config_dir = os.path.join(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    with rename_after_writing.open(
        os.path.join(config_dir, "zenith_bin_edges.numpy"), "wb"
    ) as f:
        np.save(f, zenith_bin_edges)

    with rename_after_writing.open(
        os.path.join(config_dir, "energy_bin_edges.numpy"), "wb"
    ) as f:
        np.save(f, energy_bin_edges)


def read_config_if_None(work_dir, config):
    if config is None:
        config_dir = os.path.join(work_dir, "config")
        out = {}
        out["energy_bin"] = {}
        out["zenith_bin"] = {}
        with open(
            os.path.join(config_dir, "energy_bin_edges.numpy"), "rb"
        ) as f:
            out["energy_bin"]["edges"] = np.load(f)
        with open(
            os.path.join(config_dir, "zenith_bin_edges.numpy"), "rb"
        ) as f:
            out["zenith_bin"]["edges"] = np.load(f)

        for key in out:
            out[key]["num"] = len(out[key]["edges"]) - 1
        return out
    else:
        return config


def populate(work_dir, event_table_path, config=None):
    config = read_config_if_None(work_dir=work_dir, config=config)
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
        dtype=[("uid", "u8"), ("zd", "u1"), ("en", "u1")],
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


def _write_json(path, obj):
    with rename_after_writing.open(path, mode="wt") as f:
        f.write(json_utils.dumps(obj))


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
            f.read(), dtype=[("uid", "u8"), ("zd", "u1"), ("en", "u1")]
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
):
    assert num_energy_bins > 0
    assert num_zenith_bins > 0
    os.makedirs(out_path, exist_ok=True)

    EVENT_TABLE_DTYPES = structure.dtypes()

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
                    dtypes=EVENT_TABLE_DTYPES,
                    index_key="uid",
                    compress=False,
                ) as event_table_writer:
                    for level_key in EVENT_TABLE_DTYPES:

                        bin_level_inpath = os.path.join(
                            stage_path,
                            level_key,
                            binname + ".recarray",
                        )

                        with open(bin_level_inpath, mode="rb") as fin:
                            level_recarray = np.frombuffer(
                                fin.read(),
                                dtype=EVENT_TABLE_DTYPES[level_key],
                            )
                        _table = snt.SparseNumericTable(
                            dtypes=EVENT_TABLE_DTYPES,
                            index_key="uid",
                        )
                        _table[level_key].append(level_recarray)
                        event_table_writer.append_table(_table)
