import numpy as np
import binning_utils as bu
import sparse_numeric_table as snt
import os
import json_utils


class Bins:
    def __init__(self, zenith_rad, energy_GeV, altitude_m):
        self.zenith = bu.Binning(bin_edges=zenith_rad)
        self.energy = bu.Binning(bin_edges=energy_GeV)
        self.altitude = bu.Binning(bin_edges=altitude_m)
        self.shape = (
            self.zenith["num"],
            self.energy["num"],
            self.altitude["num"],
        )
        self.zenith_overlaps = make_overlaps(size=self.zenith["num"])
        self.energy_overlaps = make_overlaps(size=self.energy["num"])
        self.altitude_overlaps = make_overlaps(size=self.altitude["num"])

    @classmethod
    def from_path(cls, path):
        jr = json_utils.read
        opj = os.path.join

        return cls(
            zenith_rad=jr(opj(path, "zenith_rad.json")),
            energy_GeV=jr(opj(path, "energy_GeV.json")),
            altitude_m=jr(opj(path, "altitude_m.json")),
        )

    def to_path(self, path):
        opj = os.path.join
        jw = json_utils.write
        os.makedirs(path, exist_ok=True)
        jw(opj(path, "energy_GeV.json"), self.energy["edges"])
        jw(opj(path, "zenith_rad.json"), self.zenith["edges"])
        jw(opj(path, "altitude_m.json"), self.altitude["edges"])

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


def len_cube(cube):
    shape = (len(cube), len(cube), len(cube))
    out = make_cube_of_lists(shape=shape, default=0)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                out[x][y][z] = len(cube[x][y][z])
    return np.asarray(out)


def _cut_uids(level, column_key, start, stop):
    mask = np.logical_and(level[column_key] >= start, level[column_key] < stop)
    return level["uid"][mask]


def make_overlaps(size):
    overlaps = []
    for ipivot in range(size):
        indices = []
        weights = []
        for ii in range(ipivot - 1, ipivot + 2):
            if ii >= 0 and ii < size:
                indices.append(ii)
                weights.append(1.0 if ii == ipivot else 0.5)
        ipi = {"indices": indices, "weights": weights}
        ipi["weights"] = np.asarray(ipi["weights"]) / np.sum(ipi["weights"])
        overlaps.append(ipi)
    return overlaps


def smoothen_uid_assign_zenith_energy_altitude(assignment, bins):
    smo = bins.lists(default=None)
    over_zenith = make_overlaps(size=bins.zenith["num"])

    for zdp in range(bins.zenith["num"]):
        for enp in range(bins.energy["num"]):
            for alp in range(bins.altitude["num"]):

                smo[zdp][enp][alp] = set()
                for zd in bins.zenith_overlaps[zdp]["indices"]:
                    for en in bins.energy_overlaps[enp]["indices"]:
                        for al in bins.altitude_overlaps[alp]["indices"]:
                            to_add = set(assignment[zd][en][al])
                            smo[zdp][enp][alp] = set.union(
                                smo[zdp][enp][alp],
                                to_add,
                            )
                smo[zdp][enp][alp] = np.asarray(
                    list(smo[zdp][enp][alp]), dtype=int
                )
                smo[zdp][enp][alp] = np.sort(smo[zdp][enp][alp])
    return smo


def smoothen_uid_assign_zenith(assignment, bins):
    smo = [None for zd in range(bins.zenith["num"])]
    for zdp in range(bins.zenith["num"]):

        smo[zdp] = set()
        for zd in bins.zenith_overlaps[zdp]["indices"]:
            to_add = set(assignment[zd])
            smo[zdp] = set.union(smo[zdp], to_add)

        smo[zdp] = np.asarray(list(smo[zdp]), dtype=int)
        smo[zdp] = np.sort(smo[zdp])
    return smo
