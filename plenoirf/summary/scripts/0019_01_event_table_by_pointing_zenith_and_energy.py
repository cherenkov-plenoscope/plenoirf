#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import rename_after_writing as rnw
import dynamicsizerecarray
import os
from os.path import join as opj
import binning_utils
import json_utils
import numpy as np

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

energy_bin_key = "trigger_acceptance_onregion"
energy_bin = res.energy_binning(key=energy_bin_key)

zenith_bin_key = "once"
zenith_bin = res.zenith_binning(key=zenith_bin_key)

zenith_assignment = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0019_zenith_bin_assignment")
)
energy_assignment = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0018_energy_bin_assignment")
)[energy_bin_key]

level_dtypes = irf.event_table.structure.dtypes()


BIN_ASSIGNMENT_DTYPE = [
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


pk = "helium"

# MAKE BIN ASSIGNMENT
pk_cache_dir = opj(res.paths["cache_dir"], pk)
bin_assignment_cache_path = opj(pk_cache_dir, "bin_assignment.recarray")


if not os.path.exists(bin_assignment_cache_path):
    os.makedirs(pk_cache_dir, exist_ok=True)
    bin_assignment = make_bin_assignment(
        zenith_assignment=zenith_assignment,
        zenith_bin=zenith_bin,
        energy_assignment=energy_assignment,
        energy_bin=energy_bin,
    )
    with rnw.open(bin_assignment_cache_path, mode="wb") as fout:
        fout.write(bin_assignment.tobytes())

else:
    with rnw.open(bin_assignment_cache_path, mode="rb") as fin:
        bin_assignment = np.frombuffer(fin.read(), dtype=BIN_ASSIGNMENT_DTYPE)

print(bin_assignment.shape)


ipath = opj(
    res.paths["plenoirf_dir"],
    "response",
    res.instrument_key,
    res.site_key,
    pk,
    "reduce",
    "event_table.snt.zip",
)
pk_binned_cache_dir = opj(
    res.paths["cache_dir"], pk, "binned_in_zenith_and_energy"
)

with snt.open(ipath, mode="r") as event_table_reader:
    level_keys = list(event_table_reader.dtypes.keys())

for level_key in level_keys:
    cache_pk_level_dir = opj(pk_binned_cache_dir, level_key)

    if not os.path.exists(cache_pk_level_dir):
        with FileBins(
            path=cache_pk_level_dir,
            num_zenith_bins=zenith_bin["num"],
            num_energy_bins=energy_bin["num"],
        ) as fbins:
            with snt.open(ipath, mode="r") as event_table_reader:
                for level_block in LevelBlockLooper(
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


EVENT_TABLE_DTYPES = irf.event_table.structure.dtypes()
pk_dir = opj(res.paths["out_dir"], pk)
os.makedirs(pk_dir, exist_ok=True)

with rnw.open(
    opj(pk_dir, "instrument_pointing_zenith_bin_edges_rad.json"), mode="wt"
) as f:
    f.write(json_utils.dumps(zenith_bin["edges"]))

with rnw.open(
    opj(pk_dir, "primary_energy_bin_edges_GeV.json"), mode="wt"
) as f:
    f.write(json_utils.dumps(energy_bin["edges"]))

for zd in range(zenith_bin["num"]):
    for en in range(energy_bin["num"]):

        binname = f"zd{zd:06d}_en{en:06d}"
        bin_outpath = opj(pk_dir, binname + ".snt.zip")

        if not os.path.exists(bin_outpath):
            with rnw.Path(bin_outpath) as tmp_bin_outpath, snt.open(
                tmp_bin_outpath,
                mode="w",
                dtypes=EVENT_TABLE_DTYPES,
                index_key="uid",
                compress=False,
            ) as event_table_writer:
                for level_key in level_keys:

                    bin_level_inpath = opj(
                        pk_binned_cache_dir, level_key, binname + ".recarray"
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


# test and assert
# ---------------


class EventTableByPointingZenithAndPrimaryEnergy:
    def __init__(self, path):
        self.path = path

        with open(
            os.path.join(self.path, "primary_energy_bin_edges_GeV.json"), "rt"
        ) as fin:
            self.primary_energy_bin_edges_GeV = json_utils.loads(fin.read())

        with open(
            os.path.join(
                self.path, "instrument_pointing_zenith_bin_edges_rad.json"
            ),
            "rt",
        ) as fin:
            self.instrument_pointing_zenith_bin_edges_rad = json_utils.loads(
                fin.read()
            )

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
        tasks = make_list_of_bins_to_be_read(
            energy_bin_edges_GeV=self.primary_energy_bin_edges_GeV,
            zenith_bin_edges_rad=self.instrument_pointing_zenith_bin_edges_rad,
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
            with snt.open(bin_path, "r") as arcin:
                full = arcin.query(
                    levels_and_columns=levels_and_columns,
                )
                if task["energy"]["cut"] is not None:
                    uid_energy = _get_uid_in_energy_range(
                        reader=arcin,
                        energy_start_GeV=task["energy"]["cut"]["start"],
                        energy_stop_GeV=task["energy"]["cut"]["stop"],
                    )
                else:
                    uid_energy = None

                if task["zenith"]["cut"] is not None:
                    uid_zenith = _get_uid_in_zenith_range(
                        reader=arcin,
                        zenith_start_rad=zenith_start_rad,
                        zenith_stop_rad=zenith_stop_rad,
                    )
                else:
                    uid_zenith = None

            if uid_energy is None and uid_zenith is None:
                uid_common = None
            elif uid_zenith is not None:
                uid_common = uid_zenith
            elif uid_energy is not None:
                uid_common = uid_energy
            else:
                uid_common = snt.logic.intersection(uid_zenith, uid_energy)

            if uid_common is None:
                out.append(full)
            else:
                out.append(full.query(indices=uid_common))

        if sort:
            out = snt.logic.cut_and_sort_table_on_indices(
                table=out,
                common_indices=indices,
                inplace=True,
            )

        return out

    def __repr__(self):
        return f"{self.__class__.__name__:s}()"


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


def make_list_of_bins_to_be_read(
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
                assert irf.utils.can_be_interpreted_as_int(bin_index)
                assert 0 <= bin_index < num_bins
                out.append(bin_index)
                fully_contained.append(True)
            return out, fully_contained
        elif irf.utils.can_be_interpreted_as_int(bin_indices):
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


res.stop()
