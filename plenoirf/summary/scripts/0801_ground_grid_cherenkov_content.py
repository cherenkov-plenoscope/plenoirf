#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import json_utils
import zipfile
import gzip
import binning_utils
import pickle
import sebastians_matplotlib_addons as sebplt


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

record_dtype = [("x_bin", "i4"), ("y_bin", "i4"), ("size", "f8")]


class Reader:
    def __init__(self, path):
        self.path = path
        self.zip = zipfile.ZipFile(self.path, "r")
        self.uids = []
        for zipitem in self.zip.infolist():
            if zipitem.filename.endswith(".i4_i4_f8.gz"):
                run_id = int(os.path.dirname(zipitem.filename))
                event_id = int(os.path.basename(zipitem.filename)[0:6])
                self.uids.append(
                    irf.bookkeeping.uid.make_uid(
                        run_id=run_id, event_id=event_id
                    )
                )

    def close(self):
        self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _read_by_filename(self, filename):
        with self.zip.open(filename) as f:
            payload_gz = f.read()
        payload = gzip.decompress(payload_gz)
        return np.fromstring(payload, dtype=record_dtype)

    def _read_by_uid(self, uid):
        run_id, event_id = irf.bookkeeping.uid.split_uid(uid)
        return self._read_by_filename(
            filename=f"{run_id:06d}/{event_id:06d}.i4_i4_f8.gz"
        )

    def __getitem__(self, uid):
        return self._read_by_uid(uid=uid)

    def __iter__(self):
        return iter(self.uids)

    def __repr__(self):
        return f"{self.__class__.__name__:s}(path='{self.path:s}')"


size_bin = binning_utils.Binning(bin_edges=np.geomspace(1, 1e6, 13))

energy_bin = res.energy_binning(key="trigger_acceptance_onregion")
zenith_bin = res.zenith_binning(key="once")

cache_dir = opj(res.paths["out_dir"], "__cache__")
hist_cache_path = opj(cache_dir, "hist.pkl")
expo_cache_path = opj(cache_dir, "expo.pkl")

if os.path.exists(hist_cache_path) and os.path.exists(expo_cache_path):
    with open(hist_cache_path, "rb") as f:
        hist = pickle.loads(f.read())
    with open(expo_cache_path, "rb") as f:
        expo = pickle.loads(f.read())
else:
    os.makedirs(cache_dir, exist_ok=True)
    hist = {}
    expo = {}
    for pk in res.PARTICLES:
        gpath = opj(
            res.response_path(particle_key=pk), "ground_grid_intensity.zip"
        )

        hist[pk] = np.zeros(
            shape=(zenith_bin["num"], energy_bin["num"], size_bin["num"])
        )
        expo[pk] = np.zeros(shape=(zenith_bin["num"], energy_bin["num"]))

        with res.open_event_table(particle_key=pk) as arc:
            event_table = arc.query(
                levels_and_columns={
                    "primary": (
                        "uid",
                        "energy_GeV",
                        "zenith_rad",
                    ),
                }
            )
        primary_by_uid = {}
        for entry in event_table["primary"]:
            primary_by_uid[entry["uid"]] = (
                entry["energy_GeV"],
                entry["zenith_rad"],
            )

        with Reader(gpath) as rrr:
            for uid in rrr:
                a = rrr[uid]
                print(pk, uid, a.shape)

                eunder, ebin, eover = binning_utils.find_bin_in_edges(
                    bin_edges=energy_bin["edges"], value=primary_by_uid[uid][0]
                )
                if eunder or eover:
                    continue

                zunder, zbin, zover = binning_utils.find_bin_in_edges(
                    bin_edges=zenith_bin["edges"], value=primary_by_uid[uid][1]
                )
                if zunder or zover:
                    continue

                expo[pk][zbin][ebin] += 1
                hist[pk][zbin][ebin] += np.histogram(
                    a["size"],
                    bins=size_bin["edges"],
                )[0]

    with open(hist_cache_path, "wb") as f:
        f.write(pickle.dumps(hist))
    with open(expo_cache_path, "wb") as f:
        f.write(pickle.dumps(expo))


max_expo = 0
for pk in res.PARTICLES:
    for zd in range(zenith_bin["num"]):
        _max_expo = max(expo[pk][zd])
        if _max_expo > max_expo:
            max_expo = _max_expo

for pk in res.PARTICLES:
    for zd in range(zenith_bin["num"]):

        vals = hist[pk][zd]
        vals_norm_in_energy = np.sum(vals, axis=1)
        for ebin in range(energy_bin["num"]):
            if vals_norm_in_energy[ebin] > 0:
                vals[ebin, :] /= vals_norm_in_energy[ebin]

        fig = sebplt.figure(style=sebplt.FIGURE_1_1)
        ax_c = sebplt.add_axes(fig=fig, span=[0.25, 0.27, 0.55, 0.65])
        ax_h = sebplt.add_axes(fig=fig, span=[0.25, 0.11, 0.55, 0.1])
        ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
        ax_zd = sebplt.add_axes_zenith_range_indicator(
            fig=fig,
            span=[0.85, 0.11, 0.1, 0.1],
            zenith_bin_edges_rad=zenith_bin["edges"],
            zenith_bin=zd,
            fontsize=5,
        )
        ax_zd.text(
            s=f"{res.site_key:s}",
            x=0.0,
            y=-0.65,
            transform=ax_zd.transAxes,
        )
        ax_zd.text(
            s=f"{pk:s}",
            x=0.0,
            y=-1.0,
            transform=ax_zd.transAxes,
        )
        _pcm_confusion = ax_c.pcolormesh(
            energy_bin["edges"],
            size_bin["edges"],
            np.transpose(vals),
            cmap=res.PARTICLE_COLORMAPS[pk],
            norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
        )
        sebplt.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")

        ax_c.set_ylabel("Cherenkov size in bin / 1")
        ax_c.loglog()
        ax_c.axhline(
            res.config["ground_grid"]["threshold_num_photons"],
            linestyle=":",
            color="k",
        )
        ax_c.text(
            s=f"grid bin threshold",
            x=1.3 * energy_bin["limits"][0],
            y=1.3 * res.config["ground_grid"]["threshold_num_photons"],
        )

        ax_c.set_xticklabels([])
        sebplt.ax_add_grid(ax_c)

        ax_h.semilogx()
        ax_h.set_ylim([0, 1.1 * max_expo])
        ax_h.set_xlim(energy_bin["limits"])
        ax_h.set_xlabel("energy / GeV")
        ax_h.set_ylabel("num. events")

        sebplt.ax_add_histogram(
            ax=ax_h,
            bin_edges=energy_bin["edges"],
            bincounts=expo[pk][zd],
            linestyle="-",
            linecolor="k",
        )
        fig.savefig(
            opj(
                res.paths["out_dir"],
                f"{pk:s}_zd{zd:d}_ground_grid_cherenkov_content.jpg",
            )
        )
        sebplt.close(fig)
