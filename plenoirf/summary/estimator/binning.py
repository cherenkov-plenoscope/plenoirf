import numpy as np
import binning_utils as bu
import sparse_numeric_table as snt


class Bins:
    def __init__(
        self, zenith_bin_edges_rad, energy_bin_edges_GeV, altitude_bin_edges_m
    ):
        self.zenith = bu.Binning(bin_edges=zenith_bin_edges_rad)
        self.energy = bu.Binning(bin_edges=energy_bin_edges_GeV)
        self.altitude = bu.Binning(bin_edges=altitude_bin_edges_m)
        self.shape = (
            self.zenith["num"],
            self.energy["num"],
            self.altitude["num"],
        )

    def lists(self, default=None):
        return make_cube_of_lists(shape=self.shape, default=default)

    def array(self, default=np.nan, dtype=float):
        return default * np.ones(shape=self.shape, dtype=dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def assign_uids_to_zenith_energy_altitude(event_table, bins):
    assignment = make_cube_of_lists(shape=bins.shape)

    uid_zd = assign_uids_zenith(event_table=event_table, bins=bins)
    uid_en = assign_uids_energy(event_table=event_table, bins=bins)
    uid_al = assign_uids_altitude(event_table=event_table, bins=bins)

    for zd in range(bins.zenith["num"]):
        for en in range(bins.energy["num"]):
            for al in range(bins.altitude["num"]):
                assignment[zd][en][al] = snt.logic.intersection(
                    uid_zd[zd], uid_en[en], uid_al[al]
                )
                assignment[zd][en][al] = np.sort(assignment[zd][en][al])
    return assignment


def assign_uids_zenith(event_table, bins):
    uids = [None for zd in range(bins.zenith["num"])]
    for zd in range(bins.zenith["num"]):
        uids[zd] = _cut_uids(
            level=event_table["instrument_pointing"],
            column_key="zenith_rad",
            start=bins.zenith["edges"][zd],
            stop=bins.zenith["edges"][zd + 1],
        )
    return uids


def assign_uids_energy(event_table, bins):
    uids = [None for zd in range(bins.energy["num"])]
    for en in range(bins.energy["num"]):
        uids[en] = _cut_uids(
            level=event_table["primary"],
            column_key="energy_GeV",
            start=bins.energy["edges"][en],
            stop=bins.energy["edges"][en + 1],
        )
    return uids


def assign_uids_altitude(event_table, bins):
    uids = [None for zd in range(bins.altitude["num"])]
    for al in range(bins.altitude["num"]):
        uids[al] = _cut_uids(
            level=event_table["cherenkovpool"],
            column_key="z_emission_p50_m",
            start=bins.altitude["edges"][al],
            stop=bins.altitude["edges"][al + 1],
        )
    return uids


def make_cube_of_lists(shape, default=None):
    """
    Returns a 3D cube of lists with 'shape' and filled with 'default'.
    """
    x, y, z = shape
    cube = []
    for _ in range(x):
        _x = []
        for _ in range(y):
            _y = []
            for _ in range(z):
                _y.append(default)
            _x.append(_y)
        cube.append(_x)
    return cube


def _cut_uids(level, column_key, start, stop):
    mask = np.logical_and(level[column_key] >= start, level[column_key] < stop)
    return level["uid"][mask]
