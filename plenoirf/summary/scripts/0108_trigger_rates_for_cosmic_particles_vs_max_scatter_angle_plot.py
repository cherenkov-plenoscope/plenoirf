#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import copy
import propagate_uncertainties as pu
import sebastians_matplotlib_addons as seb
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

PARTICLES = irf_config["config"]["particles"]
COSMIC_RAYS = irf.utils.filter_particles_with_electric_charge(PARTICLES)
SITES = irf_config["config"]["sites"]
PLT = sum_config["plot"]

energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

scatter_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "scatter.json")
)

rates = json_utils.tree.read(
    os.path.join(
        pa["summary_dir"],
        "0107_trigger_rates_for_cosmic_particles_vs_max_scatter_angle",
    )
)

MAX_SCATTER_SOLID_ANGLE_SR = 0.0
for pk in COSMIC_RAYS:
    MAX_SCATTER_SOLID_ANGLE_SR = np.max(
        [MAX_SCATTER_SOLID_ANGLE_SR, scatter_bin[pk]["stop"]]
    )

for sk in SITES:
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in COSMIC_RAYS:
        R = rates[sk][pk]["integral"]["R"]
        R_au = rates[sk][pk]["integral"]["R_au"]

        seb.ax_add_histogram(
            ax=ax,
            bin_edges=1e3 * scatter_bin[pk]["edges"],
            bincounts=1e-3 * (R),
            bincounts_lower=1e-3 * (R - R_au),
            bincounts_upper=1e-3 * (R + R_au),
            linecolor=PLT["particle_colors"][pk],
            face_color=PLT["particle_colors"][pk],
            face_alpha=0.2,
        )
    fig.text(
        x=0.8,
        y=0.05,
        s=r"1msr = 3.3(1$^\circ)^2$",
        color="grey",
    )
    ax.set_ylabel("trigger-rate / 1k s$^{-1}$")
    ax.set_xlabel("scatter solid angle / msr")
    ax.set_xlim(1e3 * scatter_bin[pk]["limits"])
    ax.set_ylim([0, 1e2])
    fig.savefig(
        os.path.join(pa["out_dir"], sk + "_trigger-rate_vs_scatter.jpg")
    )
    seb.close(fig)


for sk in SITES:
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in COSMIC_RAYS:
        R = rates[sk][pk]["integral"]["R"]
        R_au = rates[sk][pk]["integral"]["R_au"]

        dRdS = np.zeros(scatter_bin[pk]["num_bins"] - 1)
        dRdS_au = np.zeros(dRdS.shape)
        for sc in range(scatter_bin[pk]["num_bins"] - 1):
            dR, dR_au = pu.add(
                x=R[sc + 1],
                x_au=R_au[sc + 1],
                y=-R[sc],
                y_au=R_au[sc],
            )
            _Rmean, _Rmean_au = pu.add(
                x=R[sc + 1],
                x_au=R_au[sc + 1],
                y=R[sc],
                y_au=R_au[sc],
            )
            Rmean, Rmean_au = pu.multiply(
                x=0.5,
                x_au=0.0,
                y=_Rmean,
                y_au=_Rmean_au,
            )
            dS = 1e3 * scatter_bin[pk]["widths"][sc]
            _dRdS, _dRdS_au = pu.divide(x=dR, x_au=dR_au, y=dS, y_au=0.0)
            dRdS[sc], dRdS_au[sc] = pu.divide(
                x=_dRdS,
                x_au=_dRdS_au,
                y=Rmean,
                y_au=Rmean_au,
            )

        seb.ax_add_histogram(
            ax=ax,
            bin_edges=1e3 * scatter_bin[pk]["edges"][0:-1],
            bincounts=dRdS,
            bincounts_lower=(dRdS - dRdS_au),
            bincounts_upper=(dRdS + dRdS_au),
            linecolor=PLT["particle_colors"][pk],
            face_color=PLT["particle_colors"][pk],
            face_alpha=0.2,
        )

    fig.text(
        x=0.8,
        y=0.05,
        s=r"1msr = 3.3(1$^\circ)^2$",
        color="grey",
    )
    ax.set_ylabel(
        (
            "R: trigger-rate / s$^{-1}$\n"
            "S: scatter solid angle / msr\n"
            "dR/dS R$^{-1}$ / (msr)$^{-1}$"
        )
    )
    ax.set_xlabel("scatter solid angle / msr")
    ax.set_xlim(1e3 * scatter_bin[pk]["limits"])
    ax.set_ylim([1e-4, 1e0])
    ax.semilogy()
    fig.savefig(
        os.path.join(pa["out_dir"], sk + "_diff-trigger-rate_vs_scatter.jpg")
    )
    seb.close(fig)


AXSPAN = copy.deepcopy(irf.summary.figure.AX_SPAN)
AXSPAN = [AXSPAN[0], AXSPAN[1], AXSPAN[2], AXSPAN[3]]

for sk in SITES:
    for pk in COSMIC_RAYS:
        print("plot 2D", sk, pk)

        dRdE = rates[sk][pk]["differential"]["dRdE"]
        dRdE_au = rates[sk][pk]["differential"]["dRdE_au"]

        # integrate along energy to get rate
        # ----------------------------------
        R = np.zeros(shape=dRdE.shape)
        for sc in range(scatter_bin[pk]["num_bins"]):
            R[sc, :] = dRdE[sc, :] * energy_bin["widths"]

        # differentiate w.r.t. scatter
        # ----------------------------
        dRdS = np.zeros(shape=(R.shape[0] - 1, R.shape[1]))
        for eb in range(energy_bin["num_bins"]):
            for sc in range(scatter_bin[pk]["num_bins"] - 1):
                dR = R[sc + 1, eb] - R[sc, eb]
                Rmean = 0.5 * (R[sc + 1, eb] + R[sc, eb])
                dS = 1e3 * scatter_bin[pk]["widths"][sc]

                with np.errstate(divide="ignore", invalid="ignore"):
                    dRdS[sc, eb] = (dR / dS) / Rmean

        dRdS[np.isnan(dRdS)] = 0.0

        fig = seb.figure(style=irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=[AXSPAN[0], AXSPAN[1], 0.55, 0.7])

        ax_cb = seb.add_axes(
            fig=fig,
            span=[0.8, AXSPAN[1], 0.02, 0.7],
            # style=seb.AXES_BLANK,
        )

        ax.set_xlim(energy_bin["limits"])
        ax.set_ylim([0, 1e3 * MAX_SCATTER_SOLID_ANGLE_SR])
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
            dRdS,
            norm=seb.plt_colors.LogNorm(),
            cmap="terrain_r",
            vmin=1e-4,
            vmax=1e0,
        )

        seb.plt.colorbar(
            pcm_ratio,
            cax=ax_cb,
            label=(
                "R: trigger-rate / s$^{-1}$\n"
                "S: scatter solid angle / msr\n"
                "dR/dS R$^{-1}$ / (msr)$^{-1}$"
            ),
        )
        seb.ax_add_grid(ax=ax)

        fig.savefig(
            os.path.join(
                pa["out_dir"],
                "{:s}_{:s}_diff-trigger-rate_vs_scatter_vs_energy.jpg".format(
                    sk,
                    pk,
                ),
            )
        )
        seb.close(fig)
