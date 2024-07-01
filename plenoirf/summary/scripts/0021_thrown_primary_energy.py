#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
import cosmic_fluxes
import json_utils
import spherical_coordinates
import solid_angle_utils
import binning_utils
import spherical_histogram
import sebastians_matplotlib_addons as seb


paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)
seb.matplotlib.rcParams.update(res.analysis["plot"]["matplotlib"])

energy_bin = json_utils.read(
    os.path.join(paths["analysis_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance"]

POINTNIG_ZENITH_BIN = res.ZenithBinning("once")

eee = {}
for pk in res.PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        table = arc.read_table(
            levels_and_columns={
                "primary": [snt.IDX, "energy_GeV", "azimuth_rad", "zenith_rad"]
            }
        )

    eee[pk] = {}
    eee[pk]["bin_counts"] = np.zeros(
        shape=(POINTNIG_ZENITH_BIN.num, energy_bin["num_bins"])
    )

    for zdbin in range(POINTNIG_ZENITH_BIN.num):
        zenith_start_rad = POINTNIG_ZENITH_BIN.edges[zdbin]
        zenith_stop_rad = POINTNIG_ZENITH_BIN.edges[zdbin + 1]

        mask = np.logical_and(
            table["primary"]["zenith_rad"] >= zenith_start_rad,
            table["primary"]["zenith_rad"] < zenith_stop_rad,
        )

        eee[pk]["bin_counts"][zdbin, :] = np.histogram(
            table["primary"]["energy_GeV"][mask], bins=energy_bin["edges"]
        )[0]


# differential SED style histogram
# --------------------------------
linestyles = ["-", "--", ":"]
fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
for pk in res.PARTICLES:
    for zdbin in range(POINTNIG_ZENITH_BIN.num):
        ax.plot(
            energy_bin["centers"],
            (eee[pk]["bin_counts"][zdbin, :] / energy_bin["widths"])
            * energy_bin["centers"] ** (1.5),
            linestyle=linestyles[zdbin],
            color=res.PARTICLE_COLORS[pk],
        )
res.ax_add_site_marker(ax)
ax.loglog()
ax.set_xlim(energy_bin["limits"])
ax.set_xlabel("energy / GeV")
ax.set_ylabel(
    r"(energy)$^{1.5}$ differential intensity /"
    + "\n"
    + r"(GeV)$^{1.5}$ (GeV)$^{-1}$"
)
fig.savefig(
    os.path.join(
        paths["out_dir"],
        "thrown_primary_energy_differential_spectral_energy_distribution.jpg",
    )
)
seb.close(fig)

# simple histogram
# ----------------
fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
for pk in res.PARTICLES:
    for zdbin in range(POINTNIG_ZENITH_BIN.num):
        seb.ax_add_histogram(
            ax=ax,
            bin_edges=energy_bin["edges"],
            bincounts=eee[pk]["bin_counts"][zdbin, :],
            linestyle=linestyles[zdbin],
            linecolor=res.PARTICLE_COLORS[pk],
            linealpha=1.0,
            bincounts_upper=None,
            bincounts_lower=None,
            face_color=res.PARTICLE_COLORS[pk],
            face_alpha=0.3,
            draw_bin_walls=True,
        )
res.ax_add_site_marker(ax)
ax.loglog()
ax.set_xlim(energy_bin["limits"])
ax.set_xlabel("energy / GeV")
ax.set_ylabel(r"intensity / 1")
fig.savefig(os.path.join(paths["out_dir"], "thrown_primary_energy.jpg"))
seb.close(fig)
