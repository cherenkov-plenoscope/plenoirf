#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils
import copy

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

energy_bin = res.energy_binning(key="trigger_acceptance_onregion")
zenith_bin = res.zenith_binning("once")

acceptance = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"],
        "0102_trigger_acceptance_for_cosmic_particles_vs_max_scatter_angle",
    )
)

source_key = "diffuse"

# find limits for axis
# --------------------
MAX_SCATTER_SOLID_ANGLE_SR = 0.0
for pk in res.PARTICLES:
    scatter_bin = res.scatter_binning(particle_key=pk)
    MAX_SCATTER_SOLID_ANGLE_SR = np.max(
        [MAX_SCATTER_SOLID_ANGLE_SR, scatter_bin["stop"]]
    )

AXSPAN = copy.deepcopy(irf.summary.figure.AX_SPAN)
AXSPAN = [AXSPAN[0], AXSPAN[1], AXSPAN[2], AXSPAN[3]]

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    for pk in res.PARTICLES:
        print("plot 2D", pk)
        scatter_bin = res.scatter_binning(particle_key=pk)

        acc = acceptance[zk][pk][source_key]

        Q = acc["mean"]
        Q_au = acc["absolute_uncertainty"]

        dQdScatter = np.zeros(shape=(Q.shape[0] - 1, Q.shape[1]))
        for isc in range(scatter_bin["num"] - 1):
            dQ = Q[isc + 1, :] - Q[isc, :]
            Qmean = 0.5 * (Q[isc + 1, :] + Q[isc, :])
            dS = 1e3 * scatter_bin["widths"][isc]

            with np.errstate(divide="ignore", invalid="ignore"):
                dQdScatter[isc, :] = (dQ / dS) / Qmean

            dQdScatter[np.isnan(dQdScatter)] = 0.0

        fig = sebplt.figure(style=irf.summary.figure.FIGURE_STYLE)
        ax = sebplt.add_axes(fig=fig, span=[AXSPAN[0], AXSPAN[1], 0.55, 0.7])

        sebplt.add_axes_zenith_range_indicator(
            fig=fig,
            span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
            zenith_bin_edges_rad=zenith_bin["edges"],
            zenith_bin=zd,
            fontsize=6,
        )

        ax_cb = sebplt.add_axes(
            fig=fig,
            span=[0.8, AXSPAN[1], 0.02, 0.7],
            # style=sebplt.AXES_BLANK,
        )

        ax.set_xlim(energy_bin["limits"])
        ax.set_ylim(
            [
                0,
                1e3 * MAX_SCATTER_SOLID_ANGLE_SR,
            ]
        )
        ax.semilogx()

        ax.set_xlabel("energy / GeV")
        ax.set_ylabel("scatter solid angle / msr")

        fig.text(
            x=0.8,
            y=0.05,
            s=r"1msr = 3.3(1$^\circ)^2$",
            color="grey",
        )
        pcm_ratio = ax.pcolormesh(
            energy_bin["edges"],
            1e3 * scatter_bin["edges"][0:-1],
            dQdScatter,
            norm=sebplt.plt_colors.LogNorm(vmin=1e-4, vmax=1e0),
            cmap="terrain_r",
        )

        sebplt.plt.colorbar(
            pcm_ratio,
            cax=ax_cb,
            label=(
                "Q: acceptance / m$^2$ sr\n"
                "S: scatter solid angle / msr\n"
                "dQ/dS Q$^{-1}$ / (msr)$^{-1}$"
            ),
        )
        sebplt.ax_add_grid(ax=ax)

        fig.savefig(
            opj(
                res.paths["out_dir"],
                f"{pk:s}_{zk:s}_acceptance_vs_scatter_vs_energy.jpg",
            )
        )
        sebplt.close(fig)

res.stop()
