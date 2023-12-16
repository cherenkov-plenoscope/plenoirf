#!/usr/bin/python
import sys
import copy
import numpy as np
import propagate_uncertainties as pru
import flux_sensitivity
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import json_utils


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
COSMIC_RAYS = irf.utils.filter_particles_with_electric_charge(PARTICLES)
ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

# load
# ----
energy_binning = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]

energy_migration = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0066_energy_estimate_quality")
)

acceptance = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

airshower_fluxes = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0017_flux_of_airshowers_rebin")
)

# prepare
# -------
diff_flux = {}
diff_flux_au = {}
for sk in SITES:
    diff_flux[sk] = {}
    diff_flux_au[sk] = {}
    for pk in COSMIC_RAYS:
        diff_flux[sk][pk] = airshower_fluxes[sk][pk]["differential_flux"]
        diff_flux_au[sk][pk] = airshower_fluxes[sk][pk]["absolute_uncertainty"]

# work
# ----
gk = "diffuse"  # geometry-key (gk) for source.

# cosmic-ray-rate
# in reconstructed energy
Rreco = {}
Rreco_au = {}  # absolute uncertainty

# in true energy
Rtrue = {}
Rtrue_au = {}

for sk in SITES:
    Rreco[sk] = {}
    Rreco_au[sk] = {}
    Rtrue[sk] = {}
    Rtrue_au[sk] = {}
    for ok in ONREGION_TYPES:
        Rreco[sk][ok] = {}
        Rreco_au[sk][ok] = {}
        Rtrue[sk][ok] = {}
        Rtrue_au[sk][ok] = {}
        for pk in COSMIC_RAYS:
            print(sk, pk, ok)

            (
                Rtrue[sk][ok][pk],
                Rtrue_au[sk][ok][pk],
            ) = flux_sensitivity.differential.estimate_rate_in_true_energy(
                energy_bin_edges_GeV=energy_bin["edges"],
                acceptance_m2_sr=acceptance[sk][ok][pk][gk]["mean"],
                acceptance_m2_sr_au=acceptance[sk][ok][pk][gk][
                    "absolute_uncertainty"
                ],
                differential_flux_per_m2_per_sr_per_s_per_GeV=diff_flux[sk][
                    pk
                ],
                differential_flux_per_m2_per_sr_per_s_per_GeV_au=diff_flux_au[
                    sk
                ][pk],
            )

            flux_sensitivity.differential.assert_energy_reco_given_true_ax0true_ax1reco_is_normalized(
                energy_reco_given_true_ax0true_ax1reco=energy_migration[sk][
                    pk
                ]["reco_given_true"],
                margin=1e-2,
            )

            (
                Rreco[sk][ok][pk],
                Rreco_au[sk][ok][pk],
            ) = flux_sensitivity.differential.estimate_rate_in_reco_energy(
                energy_bin_edges_GeV=energy_bin["edges"],
                acceptance_m2_sr=acceptance[sk][ok][pk][gk]["mean"],
                acceptance_m2_sr_au=acceptance[sk][ok][pk][gk][
                    "absolute_uncertainty"
                ],
                differential_flux_per_m2_per_sr_per_s_per_GeV=diff_flux[sk][
                    pk
                ],
                differential_flux_per_m2_per_sr_per_s_per_GeV_au=diff_flux_au[
                    sk
                ][pk],
                energy_reco_given_true_ax0true_ax1reco=energy_migration[sk][
                    pk
                ]["reco_given_true"],
                energy_reco_given_true_ax0true_ax1reco_au=energy_migration[sk][
                    pk
                ]["reco_given_true_abs_unc"],
            )

            flux_sensitivity.differential.assert_integral_rates_are_similar_in_reco_and_true_energy(
                rate_in_reco_energy_per_s=Rreco[sk][ok][pk],
                rate_in_true_energy_per_s=Rtrue[sk][ok][pk],
                margin=0.3,
            )

# export
# ------
for sk in SITES:
    for ok in ONREGION_TYPES:
        for pk in COSMIC_RAYS:
            os.makedirs(os.path.join(pa["out_dir"], sk, ok, pk), exist_ok=True)

for sk in SITES:
    for ok in ONREGION_TYPES:
        for pk in COSMIC_RAYS:
            json_utils.write(
                os.path.join(pa["out_dir"], sk, ok, pk, "reco" + ".json"),
                {
                    "comment": "Rate after all cuts VS reco energy",
                    "unit": "s$^{-1}$",
                    "mean": Rreco[sk][ok][pk],
                    "absolute_uncertainty": Rreco_au[sk][ok][pk],
                    "energy_binning_key": energy_bin["key"],
                    "symbol": "Rreco",
                },
            )

            json_utils.write(
                os.path.join(pa["out_dir"], sk, ok, pk, "true" + ".json"),
                {
                    "comment": "Rate after all cuts VS true energy",
                    "unit": "s$^{-1}$",
                    "mean": Rtrue[sk][ok][pk],
                    "absolute_uncertainty": Rtrue[sk][ok][pk],
                    "energy_binning_key": energy_bin["key"],
                    "symbol": "Rtrue",
                },
            )

# plot
# ----
for sk in SITES:
    for ok in ONREGION_TYPES:
        fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
        for pk in COSMIC_RAYS:
            seb.ax_add_histogram(
                ax=ax,
                bin_edges=energy_bin["edges"],
                bincounts=Rreco[sk][ok][pk],
                bincounts_upper=Rreco[sk][ok][pk] - Rreco_au[sk][ok][pk],
                bincounts_lower=Rreco[sk][ok][pk] + Rreco_au[sk][ok][pk],
                linestyle="-",
                linecolor=sum_config["plot"]["particle_colors"][pk],
                face_color=sum_config["plot"]["particle_colors"][pk],
                face_alpha=0.25,
            )

            alpha = 0.25
            seb.ax_add_histogram(
                ax=ax,
                bin_edges=energy_bin["edges"],
                bincounts=Rtrue[sk][ok][pk],
                bincounts_upper=Rtrue[sk][ok][pk] - Rtrue_au[sk][ok][pk],
                bincounts_lower=Rtrue[sk][ok][pk] + Rtrue_au[sk][ok][pk],
                linecolor=sum_config["plot"]["particle_colors"][pk],
                linealpha=alpha,
                linestyle=":",
                face_color=sum_config["plot"]["particle_colors"][pk],
                face_alpha=alpha * 0.25,
            )

        ax.set_ylabel("rate / s$^{-1}$")
        ax.set_xlabel("reco. energy / GeV")
        ax.set_ylim([1e-6, 1e4])
        ax.loglog()
        fig.savefig(
            os.path.join(
                pa["out_dir"],
                sk + "_" + ok + "_differential_rates_vs_reco_energy.jpg",
            )
        )
        seb.close(fig)
