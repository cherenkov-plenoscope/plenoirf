#!/usr/bin/python
import sys
import copy
import plenoirf as irf
import confusion_matrix
import sparse_numeric_table as snt
import os
from os.path import join as opj
import pandas
import numpy as np
import pickle
import json_utils
import binning_utils
import sebastians_matplotlib_addons as sebplt

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

reconstructed = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"],
        "0065_learning_airshower_maximum_and_energy",
        "MultiLayerPerceptron",
    ),
)

altitude_bin = binning_utils.Binning(bin_edges=np.linspace(5e3, 25e3, 16))

cta = irf.other_instruments.cherenkov_telescope_array_south
fermi_lat = irf.other_instruments.fermi_lat

min_number_samples = 5
mk = "z_emission_p50_m"
M_TO_KM = 1e-3

for pk in res.PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={"cherenkovpool": ("uid", "z_emission_p50_m")},
            indices=reconstructed[pk][mk]["uid"],
            sort=True,
        )

    true_altitude = event_table["cherenkovpool"]["z_emission_p50_m"]
    reco_altitude = irf.analysis.energy.align_on_idx(
        input_idx=reconstructed[pk][mk]["uid"],
        input_values=reconstructed[pk][mk][mk],
        target_idxs=event_table["cherenkovpool"]["uid"],
    )

    cm = confusion_matrix.init(
        ax0_key="true_altitude",
        ax0_values=true_altitude,
        ax0_bin_edges=altitude_bin["edges"],
        ax1_key="reco_altitude",
        ax1_values=reco_altitude,
        ax1_bin_edges=altitude_bin["edges"],
        min_exposure_ax0=min_number_samples,
        default_low_exposure=0.0,
    )

    # explicit rename for conditional probability
    # -------------------------------------------
    cm["reco_given_true"] = cm.pop("counts_normalized_on_ax0")
    cm["reco_given_true_abs_unc"] = cm.pop("counts_normalized_on_ax0_au")

    json_utils.write(opj(res.paths["out_dir"], f"{pk}.json"), cm)

    # performace
    if pk == "gamma":
        (
            deltaA_over_A,
            deltaA_over_A_relunc,
        ) = irf.analysis.energy.estimate_energy_resolution_vs_reco_energy(
            true_energy=true_altitude,
            reco_energy=reco_altitude,
            reco_energy_bin_edges=altitude_bin["edges"],
            containment_fraction=0.68,
        )

        fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
        ax1 = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

        sebplt.ax_add_histogram(
            ax=ax1,
            bin_edges=altitude_bin["edges"] * M_TO_KM,
            bincounts=deltaA_over_A,
            bincounts_upper=deltaA_over_A * (1 + deltaA_over_A_relunc),
            bincounts_lower=deltaA_over_A * (1 - deltaA_over_A_relunc),
            face_color="k",
            face_alpha=0.1,
        )
        # ax1.semilogx()
        ax1.set_xlim(altitude_bin["limits"] * M_TO_KM)
        ax1.axvline(
            altitude_bin["start"], color="black", linestyle=":", alpha=0.1
        )
        ax1.axvline(
            altitude_bin["stop"], color="black", linestyle=":", alpha=0.1
        )
        ax1.set_ylim([0, 0.3])
        ax1.set_xlabel("reco. altitude / km")
        ax1.set_ylabel(r"$\Delta{}$Alt./Alt. 68% / 1")
        # ax1.legend(loc="best", fontsize=10)

        fig.savefig(opj(res.paths["out_dir"], f"{pk}_resolution.jpg"))
        sebplt.close(fig)

    fig = sebplt.figure(sebplt.FIGURE_1_1)
    ax_c = sebplt.add_axes(fig=fig, span=[0.15, 0.27, 0.65, 0.65])
    ax_h = sebplt.add_axes(fig=fig, span=[0.15, 0.11, 0.65, 0.1])
    ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
    _pcm_confusion = ax_c.pcolormesh(
        cm["ax0_bin_edges"] * M_TO_KM,
        cm["ax1_bin_edges"] * M_TO_KM,
        np.transpose(cm["reco_given_true"]),
        cmap=res.PARTICLE_COLORMAPS[pk],
        norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
    )
    ax_c.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    sebplt.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
    # irf.summary.figure.mark_ax_thrown_spectrum(ax=ax_c)
    ax_c.set_aspect("equal")
    # ax_c.set_title(r"$P$ $($ reco. energy $\vert$ true energy $)$")
    ax_c.set_ylabel("reco. altitude / km")
    # ax_c.loglog()
    ax_c.set_xticklabels([])
    # ax_h.semilogx()
    ax_h.set_xlim(altitude_bin["limits"] * M_TO_KM)
    ax_h.set_xlabel("true altitude / km")
    ax_h.set_ylabel("num. events / 1")
    # irf.summary.figure.mark_ax_thrown_spectrum(ax_h)
    ax_h.axhline(min_number_samples, linestyle=":", color="k")
    sebplt.ax_add_histogram(
        ax=ax_h,
        bin_edges=cm["ax0_bin_edges"] * M_TO_KM,
        bincounts=cm["exposure_ax0"],
        linestyle="-",
        linecolor=res.PARTICLE_COLORS[pk],
    )
    fig.savefig(opj(res.paths["out_dir"], f"{pk}.jpg"))
    sebplt.close(fig)

    # unc
    numE = altitude_bin["num"]
    ax_step = 0.8 * 1 / numE
    fig = sebplt.figure(sebplt.FIGURE_1_1)
    axstyle_stack = {"spines": ["bottom"], "axes": [], "grid": False}
    axstyle_bottom = {"spines": ["bottom"], "axes": ["x"], "grid": False}
    for ebin in range(numE):
        axe = sebplt.add_axes(
            fig=fig,
            span=[0.1, 0.1 + ax_step * ebin, 0.8, ax_step],
            style=axstyle_bottom if ebin == 0 else axstyle_stack,
        )
        mm = cm["reco_given_true"][:, ebin]
        mm_abs_unc = cm["reco_given_true_abs_unc"][:, ebin]
        sebplt.ax_add_histogram(
            ax=axe,
            bin_edges=cm["ax0_bin_edges"] * M_TO_KM,
            bincounts=mm,
            bincounts_upper=mm + mm_abs_unc,
            bincounts_lower=mm - mm_abs_unc,
            face_color=res.PARTICLE_COLORS[pk],
            face_alpha=0.25,
            linestyle="-",
            linecolor=res.PARTICLE_COLORS[pk],
        )
        axe.set_ylim([0, 1])
        axe.set_xlim(altitude_bin["limits"] * M_TO_KM)
        axe.semilogx()
        if ebin == 0:
            axe.set_xlabel("true altitude / m")
    fig.savefig(opj(res.paths["out_dir"], f"{pk}_confusion_matrix_unc.jpg"))
    sebplt.close(fig)

res.stop()
