#!/usr/bin/python
import sys
import copy
import plenoirf as irf
import confusion_matrix
import sparse_numeric_table as spt
import os
import pandas
import numpy as np
import sklearn
import pickle
import json_utils
from sklearn import neural_network
from sklearn import ensemble
from sklearn import model_selection
from sklearn import utils
import sebastians_matplotlib_addons as seb

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]

passing_trigger = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)
passing_trajectory_quality = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0059_passing_trajectory_quality")
)
reconstructed_energy = json_utils.tree.read(
    os.path.join(
        pa["summary_dir"], "0065_learning_airshower_maximum_and_energy"
    ),
)
energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

cta = irf.other_instruments.cherenkov_telescope_array_south
fermi_lat = irf.other_instruments.fermi_lat

min_number_samples = 1
mk = "energy"


def align_on_idx(input_idx, input_values, target_idxs):
    Q = {}
    for ii in range(len(input_idx)):
        Q[input_idx[ii]] = input_values[ii]
    aligned_values = np.nan * np.ones(target_idxs.shape[0])
    for ii in range(target_idxs.shape[0]):
        aligned_values[ii] = Q[target_idxs[ii]]
    return aligned_values


for sk in SITES:
    os.makedirs(os.path.join(pa["out_dir"], sk), exist_ok=True)
    for pk in PARTICLES:
        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                sk,
                pk,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        idx_valid = spt.intersection(
            [
                passing_trigger[sk][pk]["idx"],
                passing_quality[sk][pk]["idx"],
                passing_trajectory_quality[sk][pk]["idx"],
                reconstructed_energy[sk][pk][mk]["idx"],
            ]
        )

        valid_event_table = spt.cut_and_sort_table_on_indices(
            table=event_table,
            common_indices=idx_valid,
        )

        true_energy = valid_event_table["primary"]["energy_GeV"]
        reco_energy = align_on_idx(
            input_idx=reconstructed_energy[sk][pk][mk]["idx"],
            input_values=reconstructed_energy[sk][pk][mk]["energy"],
            target_idxs=valid_event_table["primary"]["idx"],
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

        json_utils.write(os.path.join(pa["out_dir"], sk, pk + ".json"), cm)

        # performace
        if pk == "gamma":
            (
                delta_energy,
                delta_energy_relunc,
            ) = irf.analysis.energy.estimate_energy_resolution_vs_reco_energy(
                true_energy=true_energy,
                reco_energy=reco_energy,
                reco_energy_bin_edges=energy_bin["edges"],
                containment_fraction=0.68,
            )

            fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
            ax1 = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

            seb.ax_add_histogram(
                ax=ax1,
                bin_edges=energy_bin["edges"],
                bincounts=delta_energy,
                bincounts_upper=delta_energy * (1 + delta_energy_relunc),
                bincounts_lower=delta_energy * (1 - delta_energy_relunc),
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
            ax1.set_ylim([0, 1])
            ax1.set_xlabel("reco. energy / GeV")
            ax1.set_ylabel(r"$\Delta{}$E/E 68% / 1")
            # ax1.legend(loc="best", fontsize=10)

            fig.savefig(
                os.path.join(pa["out_dir"], sk + "_" + pk + "_resolution.jpg")
            )
            seb.close(fig)

        fig = seb.figure(seb.FIGURE_1_1)
        ax_c = seb.add_axes(fig=fig, span=[0.15, 0.27, 0.65, 0.65])
        ax_h = seb.add_axes(fig=fig, span=[0.15, 0.11, 0.65, 0.1])
        ax_cb = seb.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
        _pcm_confusion = ax_c.pcolormesh(
            cm["ax0_bin_edges"],
            cm["ax1_bin_edges"],
            np.transpose(cm["reco_given_true"]),
            cmap="Greys",
            norm=seb.plt_colors.PowerNorm(gamma=0.5),
        )
        ax_c.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
        seb.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
        irf.summary.figure.mark_ax_thrown_spectrum(ax=ax_c)
        ax_c.set_aspect("equal")
        ax_c.set_title(r"$P(E_\mathrm{reco} \vert E_\mathrm{true})$")
        ax_c.set_ylabel("reco. energy / GeV")
        ax_c.loglog()
        ax_c.set_xticklabels([])
        ax_h.semilogx()
        ax_h.set_xlim(
            [np.min(cm["ax0_bin_edges"]), np.max(cm["ax1_bin_edges"])]
        )
        ax_h.set_xlabel("true energy / GeV")
        ax_h.set_ylabel("num. events / 1")
        irf.summary.figure.mark_ax_thrown_spectrum(ax_h)
        ax_h.axhline(min_number_samples, linestyle=":", color="k")
        seb.ax_add_histogram(
            ax=ax_h,
            bin_edges=cm["ax0_bin_edges"],
            bincounts=cm["exposure_ax0"],
            linestyle="-",
            linecolor="k",
        )
        fig.savefig(os.path.join(pa["out_dir"], sk + "_" + pk + ".jpg"))
        seb.close(fig)

        # unc
        numE = energy_bin["num_bins"]
        ax_step = 0.8 * 1 / numE
        fig = seb.figure(seb.FIGURE_1_1)
        axstyle_stack = {"spines": ["bottom"], "axes": [], "grid": False}
        axstyle_bottom = {"spines": ["bottom"], "axes": ["x"], "grid": False}
        for ebin in range(numE):
            axe = seb.add_axes(
                fig=fig,
                span=[0.1, 0.1 + ax_step * ebin, 0.8, ax_step],
                style=axstyle_bottom if ebin == 0 else axstyle_stack,
            )
            mm = cm["reco_given_true"][:, ebin]
            mm_abs_unc = cm["reco_given_true_abs_unc"][:, ebin]
            seb.ax_add_histogram(
                ax=axe,
                bin_edges=cm["ax0_bin_edges"],
                bincounts=mm,
                bincounts_upper=mm + mm_abs_unc,
                bincounts_lower=mm - mm_abs_unc,
                face_color="k",
                face_alpha=0.25,
                linestyle="-",
                linecolor="k",
            )
            axe.set_ylim([0, 1])
            axe.set_xlim(energy_bin["limits"])
            axe.semilogx()
            if ebin == 0:
                axe.set_title(r"$P(E_\mathrm{true} \vert E_\mathrm{reco})$")
                axe.set_xlabel("true energy / GeV")
        fig.savefig(
            os.path.join(
                pa["out_dir"], sk + "_" + pk + "_confusion_matrix_unc.jpg"
            )
        )
        seb.close(fig)
