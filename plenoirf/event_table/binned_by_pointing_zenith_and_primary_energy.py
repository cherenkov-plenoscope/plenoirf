import numpy as np
import os
import sparse_numeric_table as snt
import binning_utils
import json_utils
import rename_after_writing

from .. import utils as plenoirf_utils
from . import structure


class EventTableByPointingZenithAndPrimaryEnergy:
    def __init__(self, path):
        self.path = path

        with open(
            os.path.join(self.path, "primary_energy_bin_edges_GeV.json"), "rt"
        ) as fin:
            self.energy_bin_edges_GeV = json_utils.loads(fin.read())

        with open(
            os.path.join(
                self.path, "instrument_pointing_zenith_bin_edges_rad.json"
            ),
            "rt",
        ) as fin:
            self.zenith_bin_edges_rad = json_utils.loads(fin.read())

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
    ):
        tasks = _make_list_of_zenith_energy_bins_to_be_read(
            energy_bin_edges_GeV=self.energy_bin_edges_GeV,
            zenith_bin_edges_rad=self.zenith_bin_edges_rad,
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
                self.path, f"zd{zd:06d}_en{en:06d}.snt.zip"
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

        return out

    def __repr__(self):
        return f"{self.__class__.__name__:s}()"


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
    tab = arcin.query(level_and_columns={"primary": ("uid", "energy_GeV")})
    mask = energy_start_GeV <= tab["primary"]["energy_GeV"] < energy_stop_GeV
    return tab["primary"]["uid"][mask]


def _get_uid_in_zenith_range(reader, zenith_start_rad, zenith_stop_rad):
    tab = arcin.query(
        level_and_columns={"instrument_pointing": ("uid", "zenith_rad")}
    )
    mask = (
        zenith_start_rad
        <= tab["instrument_pointing"]["zenith_rad"]
        < zenith_stop_rad
    )
    return tab["instrument_pointing"]["uid"][mask]


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
            task["energy"]["bin_index"] = energy_bins[zzz]
            if energy_fully[zzz]:
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


def make_bin_assignment(
    zenith_assignment, zenith_bin, energy_assignment, energy_bin
):
    assert zenith_bin["num"] < np.iinfo("u1").max
    assert energy_bin["num"] < np.iinfo("u1").max

    bin_assignment = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=BIN_ASSIGNMENT_DTYPE
    )

    for zd in range(zenith_bin["num"]):
        zk = f"zd{zd:d}"
        for en in range(energy_bin["num"]):
            ek = f"{en:d}"

            uid_zenith_energy = snt.logic.intersection(
                zenith_assignment[zk][pk],
                energy_assignment[pk][ek],
            )
            _blk = np.recarray(
                shape=uid_zenith_energy.shape[0],
                dtype=bin_assignment.dtype,
            )
            _blk["uid"] = uid_zenith_energy
            _blk["instrument_pointing/zenith_rad"] = zd
            _blk["primary/energy_GeV"] = en
            bin_assignment.append(_blk)

    argsort = np.argsort(bin_assignment["uid"])
    bin_assignment = bin_assignment[argsort]
    return bin_assignment


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


def _populated_bins_step_zero(
    zden_dir, zenith_binning_key, energy_binning_key
):
    os.makedirs(zden_dir, exist_ok=True)

    binning_dir = os.path.join(zden_dir, "binning")
    os.makedirs(binning_dir, exist_ok=True)

    _write_json(
        os.path.join(
            binning_dir, "instrument_pointing_zenith_bin_edges_rad.json"
        ),
        zenith_bin["edges"],
    )
    _write_json(
        os.path.join(binning_dir, "primary_energy_bin_edges_GeV.json"),
        energy_bin["edges"],
    )


def _write_json(path, obj):
    with rename_after_writing.open(path, mode="wt") as f:
        f.write(json_utils.dumps(obj))


def _populated_bins_step_one(
    event_table_path, bin_assignment, tmp_path, zenith_bin, energy_bin
):
    os.makedirs(tmp_path, exist_ok=True)

    with snt.open(event_table_path, mode="r") as event_table_reader:
        level_keys = list(event_table_reader.dtypes.keys())

    for level_key in level_keys:
        level_dir = opj(tmp_path, level_key)

        if os.path.exists(level_dir):
            continue

        with FileBins(
            path=level_dir,
            num_zenith_bins=zenith_bin["num"],
            num_energy_bins=energy_bin["num"],
        ) as fbins:
            with snt.open(event_table_path, mode="r") as event_table_reader:
                for level_block in snt._file_io.LevelBlockLooper(
                    reader=event_table_reader, level_key=level_key
                ):
                    iii = np.searchsorted(
                        a=bin_assignment["uid"], v=level_block["uid"]
                    )
                    addr = bin_assignment[iii]

                    for zd in range(zenith_bin["num"]):
                        zd_mask = addr["instrument_pointing/zenith_rad"] == zd
                        for en in range(energy_bin["num"]):
                            ene_mask = addr["primary/energy_GeV"] == en
                            bin_mask = np.logical_and(zd_mask, ene_mask)
                            bin_part = level_block[bin_mask]
                            fbins[zd][en].write(bin_part.tobytes())


def _populated_bins_step_two(
    out_path,
    zenith_bin,
    energy_bin,
):
    EVENT_TABLE_DTYPES = structure.dtypes()

    for zd in range(zenith_bin["num"]):
        for en in range(energy_bin["num"]):

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
                    for level_key in level_keys:

                        bin_level_inpath = os.path.join(
                            pk_binned_cache_dir,
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
