import numpy as np
import os
import sparse_numeric_table as snt
import binning_utils

from . import utils as search_index_utils

from ... import utils as plenoirf_utils
from .. import structure


class EventTable:
    """
    Open an Event Table with search index for reading.
    The index is build in:
        - primary/energy_GeV
        - instrument_pointing/zenith_rad
    """

    def __init__(self, path):
        self.path = path

    @property
    def config(self):
        if not hasattr(self, "_config"):
            self._config = search_index_utils.read_config_if_None(self.path)
        return self._config

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
        return f"{self.__class__.__name__:s}(path={self.path:s})"

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

    elif bin_indices is None and (start is not None or stop is not None):
        assert start is not None and stop is not None

        return binning_utils.find_bin_indices_in_start_stop_range(
            bin_edges=bin_edges,
            start=start,
            stop=stop,
            return_if_bin_is_fully_contained=True,
        )
    else:
        assert bin_indices is None
        assert start is None
        assert stop is None
        num_bins = len(bin_edges) - 1

        out = np.arange(num_bins)
        fully_contained = [True for b in range(num_bins)]
        return out, fully_contained
