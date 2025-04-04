#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils
import propagate_uncertainties as pru
import copy


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

energy_bin = json_utils.read(
    opj(res.paths["analysis_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

scatter_bin = json_utils.read(
    opj(res.paths["analysis_dir"], "0005_common_binning", "scatter.json")
)

rates = json_utils.tree.read(
    opj(
        res.paths["analysis_dir"],
        "0107_trigger_rates_for_cosmic_particles_vs_max_scatter_angle",
    )
)

MAX_SCATTER_SOLID_ANGLE_SR = 0.0
for pk in res.COSMIC_RAYS:
    MAX_SCATTER_SOLID_ANGLE_SR = np.max(
        [MAX_SCATTER_SOLID_ANGLE_SR, scatter_bin[pk]["stop"]]
    )

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
for pk in res.COSMIC_RAYS:
    R = rates[pk]["integral"]["R"]
    R_au = rates[pk]["integral"]["R_au"]

    sebplt.ax_add_histogram(
        ax=ax,
        bin_edges=1e3 * scatter_bin[pk]["edges"],
        bincounts=1e-3 * (R),
        bincounts_lower=1e-3 * (R - R_au),
        bincounts_upper=1e-3 * (R + R_au),
        linecolor=res.PARTICLE_COLORS[pk],
        face_color=res.PARTICLE_COLORS[pk],
        face_alpha=0.2,
    )
fig.text(
    x=0.8,
    y=0.05,
    s=r"1msr = 3.3(1$^\circ)^2$",
    color="grey",
)
ax.set_ylabel("trigger rate / 1k s$^{-1}$")
ax.set_xlabel("scatter solid angle / msr")
ax.set_xlim(1e3 * scatter_bin[pk]["limits"])
ax.set_ylim([0, 1e2])
fig.savefig(opj(res.paths["out_dir"], "trigger-rate_vs_scatter.jpg"))
sebplt.close(fig)


fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
for pk in res.COSMIC_RAYS:
    R = rates[pk]["integral"]["R"]
    R_au = rates[pk]["integral"]["R_au"]

    dRdS = np.zeros(scatter_bin[pk]["num"] - 1)
    dRdS_au = np.zeros(dRdS.shape)
    for sc in range(scatter_bin[pk]["num"] - 1):
        dR, dR_au = pru.add(
            x=R[sc + 1],
            x_au=R_au[sc + 1],
            y=-R[sc],
            y_au=R_au[sc],
        )
        _Rmean, _Rmean_au = pru.add(
            x=R[sc + 1],
            x_au=R_au[sc + 1],
            y=R[sc],
            y_au=R_au[sc],
        )
        Rmean, Rmean_au = pru.multiply(
            x=0.5,
            x_au=0.0,
            y=_Rmean,
            y_au=_Rmean_au,
        )
        dS = 1e3 * scatter_bin[pk]["widths"][sc]
        _dRdS, _dRdS_au = pru.divide(x=dR, x_au=dR_au, y=dS, y_au=0.0)
        dRdS[sc], dRdS_au[sc] = pru.divide(
            x=_dRdS,
            x_au=_dRdS_au,
            y=Rmean,
            y_au=Rmean_au,
        )

    sebplt.ax_add_histogram(
        ax=ax,
        bin_edges=1e3 * scatter_bin[pk]["edges"][0:-1],
        bincounts=dRdS,
        bincounts_lower=(dRdS - dRdS_au),
        bincounts_upper=(dRdS + dRdS_au),
        linecolor=res.PARTICLE_COLORS[pk],
        face_color=res.PARTICLE_COLORS[pk],
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
fig.savefig(opj(res.paths["out_dir"], "diff-trigger-rate_vs_scatter.jpg"))
sebplt.close(fig)


AXSPAN = copy.deepcopy(irf.summary.figure.AX_SPAN)
AXSPAN = [AXSPAN[0], AXSPAN[1], AXSPAN[2], AXSPAN[3]]

for pk in res.COSMIC_RAYS:
    print("plot 2D", pk)

    dRdE = rates[pk]["differential"]["dRdE"]
    dRdE_au = rates[pk]["differential"]["dRdE_au"]

    # integrate along energy to get rate
    # ----------------------------------
    R = np.zeros(shape=dRdE.shape)
    for sc in range(scatter_bin[pk]["num"]):
        R[sc, :] = dRdE[sc, :] * energy_bin["widths"]

    # differentiate w.r.t. scatter
    # ----------------------------
    dRdS = np.zeros(shape=(R.shape[0] - 1, R.shape[1]))
    for eb in range(energy_bin["num"]):
        for sc in range(scatter_bin[pk]["num"] - 1):
            dR = R[sc + 1, eb] - R[sc, eb]
            Rmean = 0.5 * (R[sc + 1, eb] + R[sc, eb])
            dS = 1e3 * scatter_bin[pk]["widths"][sc]

            with np.errstate(divide="ignore", invalid="ignore"):
                dRdS[sc, eb] = (dR / dS) / Rmean

    dRdS[np.isnan(dRdS)] = 0.0

    fig = sebplt.figure(style=irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=[AXSPAN[0], AXSPAN[1], 0.55, 0.7])

    ax_cb = sebplt.add_axes(
        fig=fig,
        span=[0.8, AXSPAN[1], 0.02, 0.7],
        # style=sebplt.AXES_BLANK,
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
        norm=sebplt.plt_colors.LogNorm(vmin=1e-4, vmax=1e0),
        cmap="terrain_r",
    )

    sebplt.plt.colorbar(
        pcm_ratio,
        cax=ax_cb,
        label=(
            "R: trigger-rate / s$^{-1}$\n"
            "S: scatter solid angle / msr\n"
            "dR/dS R$^{-1}$ / (msr)$^{-1}$"
        ),
    )
    sebplt.ax_add_grid(ax=ax)

    fig.savefig(
        opj(
            res.paths["out_dir"],
            "{:s}_diff-trigger-rate_vs_scatter_vs_energy.jpg".format(
                pk,
            ),
        )
    )
    sebplt.close(fig)

res.stop()
