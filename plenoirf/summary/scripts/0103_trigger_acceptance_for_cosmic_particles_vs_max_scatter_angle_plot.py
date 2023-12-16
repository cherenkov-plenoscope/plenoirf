#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import copy
import sebastians_matplotlib_addons as seb
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

PARTICLES = irf_config["config"]["particles"]
SITES = irf_config["config"]["sites"]
PLT = sum_config["plot"]

energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

scatter_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "scatter.json")
)

acceptance = json_utils.tree.read(
    os.path.join(
        pa["summary_dir"],
        "0102_trigger_acceptance_for_cosmic_particles_vs_max_scatter_angle",
    )
)

source_key = "diffuse"

# find limits for axis
# --------------------
MAX_SCATTER_SOLID_ANGLE_SR = 0.0
for pk in PARTICLES:
    MAX_SCATTER_SOLID_ANGLE_SR = np.max(
        [MAX_SCATTER_SOLID_ANGLE_SR, scatter_bin[pk]["stop"]]
    )

AXSPAN = copy.deepcopy(irf.summary.figure.AX_SPAN)
AXSPAN = [AXSPAN[0], AXSPAN[1], AXSPAN[2], AXSPAN[3]]

for sk in SITES:
    for pk in PARTICLES:
        print("plot 2D", sk, pk)

        acc = acceptance[sk][pk][source_key]

        Q = acc["mean"]
        Q_au = acc["absolute_uncertainty"]

        dQdScatter = np.zeros(shape=(Q.shape[0] - 1, Q.shape[1]))
        for isc in range(scatter_bin[pk]["num_bins"] - 1):
            dQ = Q[isc + 1, :] - Q[isc, :]
            Qmean = 0.5 * (Q[isc + 1, :] + Q[isc, :])
            dS = 1e3 * scatter_bin[pk]["widths"][isc]

            with np.errstate(divide="ignore", invalid="ignore"):
                dQdScatter[isc, :] = (dQ / dS) / Qmean

            dQdScatter[np.isnan(dQdScatter)] = 0.0

        fig = seb.figure(style=irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=[AXSPAN[0], AXSPAN[1], 0.55, 0.7])

        ax_cb = seb.add_axes(
            fig=fig,
            span=[0.8, AXSPAN[1], 0.02, 0.7],
            # style=seb.AXES_BLANK,
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
            1e3 * scatter_bin[pk]["edges"][0:-1],
            dQdScatter,
            norm=seb.plt_colors.LogNorm(),
            cmap="terrain_r",
            vmin=1e-4,
            vmax=1e0,
        )

        seb.plt.colorbar(
            pcm_ratio,
            cax=ax_cb,
            label=(
                "Q: acceptance / m$^2$ sr\n"
                "S: scatter solid angle / msr\n"
                "dQ/dS Q$^{-1}$ / (msr)$^{-1}$"
            ),
        )
        seb.ax_add_grid(ax=ax)

        fig.savefig(
            os.path.join(
                pa["out_dir"],
                "{:s}_{:s}_acceptance_vs_scatter_vs_energy.jpg".format(
                    sk,
                    pk,
                ),
            )
        )
        seb.close(fig)
