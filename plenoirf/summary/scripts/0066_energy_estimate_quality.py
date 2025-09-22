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
import sklearn
import pickle
import json_utils
from sklearn import neural_network
from sklearn import ensemble
from sklearn import model_selection
from sklearn import utils
import sebastians_matplotlib_addons as sebplt

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

reconstructed_energy = json_utils.tree.read(
    opj(
        res.paths["analysis_dir"],
        "0065_learning_airshower_maximum_and_energy",
        "MultiLayerPerceptron",
    ),
)

energy_bin = res.energy_binning(key="trigger_acceptance_onregion")

cta = irf.other_instruments.cherenkov_telescope_array_south
fermi_lat = irf.other_instruments.fermi_lat

min_number_samples = 5
mk = "energy_GeV"


for pk in res.PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={"primary": ["uid", "energy_GeV"]},
            indices=reconstructed_energy[pk][mk]["uid"],
            sort=True,
        )

    true_energy = event_table["primary"]["energy_GeV"]
    reco_energy = irf.analysis.energy.align_on_idx(
        input_idx=reconstructed_energy[pk][mk]["uid"],
        input_values=reconstructed_energy[pk][mk][mk],
        target_idxs=event_table["primary"]["uid"],
    )

    cm = confusion_matrix.init(
        ax0_key="true_energy",
        ax0_values=true_energy,
        ax0_bin_edges=energy_bin["edges"],
        ax1_key="reco_energy",
        ax1_values=reco_energy,
        ax1_bin_edges=energy_bin["edges"],
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
            deltaE_over_E,
            deltaE_over_E_relunc,
        ) = irf.analysis.energy.estimate_energy_resolution_vs_reco_energy(
            true_energy=true_energy,
            reco_energy=reco_energy,
            reco_energy_bin_edges=energy_bin["edges"],
            containment_fraction=0.68,
        )

        fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
        ax1 = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

        sebplt.ax_add_histogram(
            ax=ax1,
            bin_edges=energy_bin["edges"],
            bincounts=deltaE_over_E,
            bincounts_upper=deltaE_over_E * (1 + deltaE_over_E_relunc),
            bincounts_lower=deltaE_over_E * (1 - deltaE_over_E_relunc),
            face_color="k",
            face_alpha=0.1,
        )
        cta_res = cta.energy_resolution()
        assert cta_res["reconstructed_energy"]["unit"] == "GeV"
        ax1.plot(
            cta_res["reconstructed_energy"]["values"],
            cta_res["energy_resolution_68"]["values"],
            color=cta.COLOR,
            label=cta.LABEL,
        )
        fermi_lat_res = fermi_lat.energy_resolution()
        assert fermi_lat_res["reconstructed_energy"]["unit"] == "GeV"
        ax1.plot(
            fermi_lat_res["reconstructed_energy"]["values"],
            fermi_lat_res["energy_resolution_68"]["values"],
            color=fermi_lat.COLOR,
            label=fermi_lat.LABEL,
        )
        ax1.semilogx()
        ax1.set_xlim([1e-1, 1e4])
        ax1.axvline(
            energy_bin["start"], color="black", linestyle=":", alpha=0.1
        )
        ax1.axvline(
            energy_bin["stop"], color="black", linestyle=":", alpha=0.1
        )
        ax1.set_ylim([0, 1])
        ax1.set_xlabel("reco. energy / GeV")
        ax1.set_ylabel(r"$\Delta{}$E/E 68% / 1")
        # ax1.legend(loc="best", fontsize=10)

        fig.savefig(opj(res.paths["out_dir"], f"{pk}_resolution.jpg"))
        sebplt.close(fig)

    fig = sebplt.figure(sebplt.FIGURE_1_1)
    ax_c = sebplt.add_axes(fig=fig, span=[0.15, 0.27, 0.65, 0.65])
    ax_h = sebplt.add_axes(fig=fig, span=[0.15, 0.11, 0.65, 0.1])
    ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
    _pcm_confusion = ax_c.pcolormesh(
        cm["ax0_bin_edges"],
        cm["ax1_bin_edges"],
        np.transpose(cm["reco_given_true"]),
        cmap=res.PARTICLE_COLORMAPS[pk],
        norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
    )
    ax_c.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    sebplt.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
    # irf.summary.figure.mark_ax_thrown_spectrum(ax=ax_c)
    ax_c.set_aspect("equal")
    # ax_c.set_title(r"$P$ $($ reco. energy $\vert$ true energy $)$")
    ax_c.set_ylabel("reco. energy / GeV")
    ax_c.loglog()
    ax_c.set_xticklabels([])
    ax_h.semilogx()
    ax_h.set_xlim([np.min(cm["ax0_bin_edges"]), np.max(cm["ax1_bin_edges"])])
    ax_h.set_xlabel("true energy / GeV")
    ax_h.set_ylabel("num. events / 1")
    # irf.summary.figure.mark_ax_thrown_spectrum(ax_h)
    ax_h.axhline(min_number_samples, linestyle=":", color="k")
    sebplt.ax_add_histogram(
        ax=ax_h,
        bin_edges=cm["ax0_bin_edges"],
        bincounts=cm["exposure_ax0"],
        linestyle="-",
        linecolor=res.PARTICLE_COLORS[pk],
    )
    fig.savefig(opj(res.paths["out_dir"], f"{pk}.jpg"))
    sebplt.close(fig)

    # unc
    numE = energy_bin["num"]
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
            bin_edges=cm["ax0_bin_edges"],
            bincounts=mm,
            bincounts_upper=mm + mm_abs_unc,
            bincounts_lower=mm - mm_abs_unc,
            face_color=res.PARTICLE_COLORS[pk],
            face_alpha=0.25,
            linestyle="-",
            linecolor=res.PARTICLE_COLORS[pk],
        )
        axe.set_ylim([0, 1])
        axe.set_xlim(energy_bin["limits"])
        axe.semilogx()
        if ebin == 0:
            axe.set_title(r"$P(E_\mathrm{true} \vert E_\mathrm{reco})$")
            axe.set_xlabel("true energy / GeV")
    fig.savefig(opj(res.paths["out_dir"], f"{pk}_confusion_matrix_unc.jpg"))
    sebplt.close(fig)

res.stop()
