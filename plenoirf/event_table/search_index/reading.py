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
        energy_start_GeV=None,
        energy_stop_GeV=None,
        zenith_start_rad=None,
        zenith_stop_rad=None,
        indices=None,
        levels_and_columns=None,
        sort=False,
        bin_by_bin=False,
    ):
        tasks = _make_list_of_zenith_energy_bins_to_be_read(
            energy_bin_edges_GeV=self.config["energy_bin"]["edges"],
            zenith_bin_edges_rad=self.config["zenith_bin"]["edges"],
            energy_start_GeV=energy_start_GeV,
            energy_stop_GeV=energy_stop_GeV,
            zenith_start_rad=zenith_start_rad,
            zenith_stop_rad=zenith_stop_rad,
        )

        task_looper = EventTableTaskLooper(
            event_table=self,
            tasks=tasks,
            indices=indices,
            levels_and_columns=levels_and_columns,
        )

        if bin_by_bin:
            return task_looper
        else:
            out = snt.SparseNumericTable(index_key="uid")
            for event_table_zd_en_bin, _zd_en_bin in task_looper:
                _ = _zd_en_bin
                out.append(event_table_zd_en_bin)

            if sort:
                out = snt.logic.cut_and_sort_table_on_indices(
                    table=out,
                    common_indices=indices,
                    inplace=True,
                )

            return out

    def population(
        self,
        energy_start_GeV=None,
        energy_stop_GeV=None,
        zenith_start_rad=None,
        zenith_stop_rad=None,
        level_key="primary",
        column_key="uid",
    ):
        looper = self.query(
            energy_start_GeV=energy_start_GeV,
            energy_stop_GeV=energy_stop_GeV,
            zenith_start_rad=zenith_start_rad,
            zenith_stop_rad=zenith_stop_rad,
            levels_and_columns={level_key: [column_key]},
            bin_by_bin=True,
        )
        population_count = 0
        for tmp, _zd_en_bin in looper:
            _ = _zd_en_bin
            population_count += tmp[level_key][column_key].shape[0]
        return population_count

    def __repr__(self):
        return f"{self.__class__.__name__:s}(path={self.path:s})"

    def bin_path(self, zd, en):
        return os.path.join(
            self.path, "bins", f"zd{zd:06d}_en{en:06d}.snt.zip"
        )

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
                    energy_start_GeV=energy_bin["edges"][en],
                    energy_stop_GeV=energy_bin["edges"][en + 1],
                    zenith_start_rad=zenith_bin["edges"][zd],
                    zenith_stop_rad=zenith_bin["edges"][zd + 1],
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
                        energy_start_GeV=energy_bin["edges"][en],
                        energy_stop_GeV=energy_bin["edges"][en + 1],
                        zenith_start_rad=zenith_bin["edges"][zd],
                        zenith_stop_rad=zenith_bin["edges"][zd + 1],
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


class EventTableTaskLooper:
    def __init__(
        self, event_table, tasks, indices=None, levels_and_columns=None
    ):
        self.event_table = event_table
        self.indices = indices
        self.levels_and_columns = levels_and_columns
        self.tasks = tasks
        self.itask = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.itask == len(self.tasks):
            raise StopIteration
        task = self.tasks[self.itask]
        self.itask += 1

        zd = task["zenith"]["bin_index"]
        en = task["energy"]["bin_index"]
        with snt.open(self.event_table.bin_path(zd=zd, en=en), "r") as reader:
            event_table_zd_en_bin = reader.query(
                indices=self.indices,
                levels_and_columns=self.levels_and_columns,
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

        uid_common = _intersection_or_None(uid_zenith, uid_energy)

        if uid_common is not None:
            event_table_zd_en_bin = snt.logic.cut_table_on_indices(
                table=event_table_zd_en_bin,
                common_indices=uid_common,
                inplace=True,
            )

        return event_table_zd_en_bin, (zd, en)

    def __repr__(self):
        return f"{self.__class__.__name__:s}(path={self.reader.path:s})"


def _intersection_or_None(a, b):
    if b is None and a is None:
        c = None
    elif a is not None:
        c = a
    elif uid_b is not None:
        c = b
    else:
        c = snt.logic.intersection(a, b)
    return c


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
    energy_start_GeV=None,
    energy_stop_GeV=None,
    zenith_start_rad=None,
    zenith_stop_rad=None,
):
    zenith_bins, zenith_fully = _make_list_of_bins_to_be_read(
        bin_edges=zenith_bin_edges_rad,
        start=zenith_start_rad,
        stop=zenith_stop_rad,
    )

    energy_bins, energy_fully = _make_list_of_bins_to_be_read(
        bin_edges=energy_bin_edges_GeV,
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
    start=None,
    stop=None,
):
    if (start is not None or stop is not None):
        assert start is not None and stop is not None

        return binning_utils.find_bin_indices_in_start_stop_range(
            bin_edges=bin_edges,
            start=start,
            stop=stop,
            return_if_bin_is_fully_contained=True,
        )
    else:
        assert start is None
        assert stop is None
        num_bins = len(bin_edges) - 1

        out = np.arange(num_bins)
        fully_contained = [True for b in range(num_bins)]
        return out, fully_contained
