#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import cosmic_fluxes
import json_utils
import spherical_coordinates
import solid_angle_utils
import binning_utils
import spherical_histogram
import sebastians_matplotlib_addons as sebplt


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

energy_bin = res.energy_binning(key="trigger_acceptance")
zenith_bin = res.zenith_binning(key="once")

eee = {}
for pk in res.PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        table = arc.query(
            levels_and_columns={
                "primary": ["uid", "energy_GeV", "azimuth_rad", "zenith_rad"]
            }
        )

    eee[pk] = {}
    eee[pk]["bin_counts"] = np.zeros(
        shape=(zenith_bin["num"], energy_bin["num"])
    )

    for zdbin in range(zenith_bin["num"]):
        zenith_start_rad = zenith_bin["edges"][zdbin]
        zenith_stop_rad = zenith_bin["edges"][zdbin + 1]

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
fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
for pk in res.PARTICLES:
    for zdbin in range(zenith_bin["num"]):
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
    opj(
        res.paths["out_dir"],
        "thrown_primary_energy_differential_spectral_energy_distribution.jpg",
    )
)
sebplt.close(fig)

# simple histogram
# ----------------
fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
for pk in res.PARTICLES:
    for zdbin in range(zenith_bin["num"]):
        sebplt.ax_add_histogram(
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
fig.savefig(opj(res.paths["out_dir"], "thrown_primary_energy.jpg"))
sebplt.close(fig)


fig = sebplt.figure({"rows": 64, "cols": 1280, "fontsize": 1.0})
ax = sebplt.add_axes(
    fig=fig, span=[0.0, 0.0, 1.0, 1.0], style=sebplt.AXES_BLANK
)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
xpos = 0.0
for pk in res.PARTICLES:
    ax.plot(
        [xpos, xpos + 0.05],
        [0.5, 0.5],
        linestyle="-",
        color=res.PARTICLE_COLORS[pk],
    )
    ax.text(
        x=xpos + 0.075,
        y=0.5,
        s=pk,
        fontsize=12,
        verticalalignment="center",
    )
    xpos += 0.25
fig.savefig(opj(res.paths["out_dir"], "particle_labels.jpg"))
sebplt.close(fig)

fig = sebplt.figure({"rows": 64, "cols": 1280, "fontsize": 1.0})
ax = sebplt.add_axes(
    fig=fig, span=[0.0, 0.0, 1.0, 1.0], style=sebplt.AXES_BLANK
)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
xpos = 0.0
for zdbin in range(zenith_bin["num"]):
    ax.plot(
        [xpos, xpos + 0.05],
        [0.5, 0.5],
        linestyle=linestyles[zdbin],
        color="grey",
    )
    zd_str = irf.summary.make_angle_range_str(
        start_rad=zenith_bin["edges"][zdbin],
        stop_rad=zenith_bin["edges"][zdbin + 1],
    )
    ax.text(
        x=xpos + 0.075,
        y=0.5,
        s=zd_str,
        fontsize=12,
        verticalalignment="center",
    )
    xpos += 0.33
fig.savefig(opj(res.paths["out_dir"], "zenith_range_labels.jpg"))
sebplt.close(fig)

res.stop()
