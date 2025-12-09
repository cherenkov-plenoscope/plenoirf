#!/usr/bin/python
import sys
import copy
import numpy as np
import propagate_uncertainties as pru
import flux_sensitivity
import plenoirf as irf
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

# load
# ----
energy_bin = res.energy_binning(key="trigger_acceptance_onregion")
zenith_bin = res.zenith_binning("3_bins_per_45deg")

energy_migration = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0066_energy_estimate_quality")
)
acceptance = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0300_onregion_trigger_acceptance")
)
airshower_fluxes = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0017_flux_of_airshowers_rebin")
)
ONREGION_TYPES = res.analysis["on_off_measuremnent"]["onregion_types"]


# prepare
# -------
diff_flux = {}
diff_flux_au = {}
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    diff_flux[zk] = {}
    diff_flux_au[zk] = {}
    for pk in res.COSMIC_RAYS:
        diff_flux[zk][pk] = airshower_fluxes[pk]["differential_flux"]
        diff_flux_au[zk][pk] = airshower_fluxes[pk]["absolute_uncertainty"]

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

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    Rreco[zk] = {}
    Rreco_au[zk] = {}
    Rtrue[zk] = {}
    Rtrue_au[zk] = {}
    for ok in ONREGION_TYPES:
        Rreco[zk][ok] = {}
        Rreco_au[zk][ok] = {}
        Rtrue[zk][ok] = {}
        Rtrue_au[zk][ok] = {}
        for pk in res.COSMIC_RAYS:
            print(zk, pk, ok)

            (
                Rtrue[zk][ok][pk],
                Rtrue_au[zk][ok][pk],
            ) = flux_sensitivity.differential.estimate_rate_in_true_energy(
                energy_bin_edges_GeV=energy_bin["edges"],
                acceptance_m2_sr=acceptance[zk][ok][pk][gk]["mean"],
                acceptance_m2_sr_au=acceptance[zk][ok][pk][gk][
                    "absolute_uncertainty"
                ],
                differential_flux_per_m2_per_sr_per_s_per_GeV=diff_flux[zk][
                    pk
                ],
                differential_flux_per_m2_per_sr_per_s_per_GeV_au=diff_flux_au[
                    zk
                ][pk],
            )

            flux_sensitivity.differential.assert_energy_reco_given_true_ax0true_ax1reco_is_normalized(
                energy_reco_given_true_ax0true_ax1reco=energy_migration[pk][
                    "reco_given_true"
                ],
                margin=1e-2,
            )

            (
                Rreco[zk][ok][pk],
                Rreco_au[zk][ok][pk],
            ) = flux_sensitivity.differential.estimate_rate_in_reco_energy(
                energy_bin_edges_GeV=energy_bin["edges"],
                acceptance_m2_sr=acceptance[zk][ok][pk][gk]["mean"],
                acceptance_m2_sr_au=acceptance[zk][ok][pk][gk][
                    "absolute_uncertainty"
                ],
                differential_flux_per_m2_per_sr_per_s_per_GeV=diff_flux[zk][
                    pk
                ],
                differential_flux_per_m2_per_sr_per_s_per_GeV_au=diff_flux_au[
                    zk
                ][pk],
                energy_reco_given_true_ax0true_ax1reco=energy_migration[pk][
                    "reco_given_true"
                ],
                energy_reco_given_true_ax0true_ax1reco_au=energy_migration[pk][
                    "reco_given_true_abs_unc"
                ],
            )

            flux_sensitivity.differential.assert_integral_rates_are_similar_in_reco_and_true_energy(
                rate_in_reco_energy_per_s=Rreco[zk][ok][pk],
                rate_in_true_energy_per_s=Rtrue[zk][ok][pk],
                margin=0.3,
            )

# export
# ------
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    for ok in ONREGION_TYPES:
        for pk in res.COSMIC_RAYS:
            os.makedirs(opj(res.paths["out_dir"], zk, ok, pk), exist_ok=True)

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    for ok in ONREGION_TYPES:
        for pk in res.COSMIC_RAYS:
            json_utils.write(
                opj(res.paths["out_dir"], zk, ok, pk, "reco" + ".json"),
                {
                    "comment": "Rate after all cuts VS reco energy",
                    "zenith_key": zk,
                    "particle_key": pk,
                    "onregion_key": ok,
                    "unit": "s$^{-1}$",
                    "mean": Rreco[zk][ok][pk],
                    "absolute_uncertainty": Rreco_au[zk][ok][pk],
                    "symbol": "Rreco",
                },
            )

            json_utils.write(
                opj(res.paths["out_dir"], zk, ok, pk, "true" + ".json"),
                {
                    "comment": "Rate after all cuts VS true energy",
                    "zenith_key": zk,
                    "particle_key": pk,
                    "onregion_key": ok,
                    "unit": "s$^{-1}$",
                    "mean": Rtrue[zk][ok][pk],
                    "absolute_uncertainty": Rtrue[zk][ok][pk],
                    "symbol": "Rtrue",
                },
            )

sfig, sax = irf.summary.figure.style(key="4:3")

# plot
# ----
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    for ok in ONREGION_TYPES:
        fig = sebplt.figure(sfig)
        ax = sebplt.add_axes(fig=fig, span=sax)

        sebplt.add_axes_zenith_range_indicator(
            fig=fig,
            span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
            zenith_bin_edges_rad=zenith_bin["edges"],
            zenith_bin=zd,
            fontsize=6,
        )

        for pk in res.COSMIC_RAYS:

            dReco_dE = Rreco[zk][ok][pk] / energy_bin["centers"]
            dReco_dE_au = Rreco_au[zk][ok][pk] / energy_bin["centers"]
            dRec0_dE_upper = dReco_dE + dReco_dE_au
            dReco_dE_lower = dReco_dE - dReco_dE_au

            sebplt.ax_add_histogram(
                ax=ax,
                bin_edges=energy_bin["edges"],
                bincounts=dReco_dE,
                bincounts_upper=dRec0_dE_upper,
                bincounts_lower=dReco_dE_lower,
                linestyle="-",
                linecolor=res.PARTICLE_COLORS[pk],
                face_color=res.PARTICLE_COLORS[pk],
                face_alpha=0.25,
            )

            dRtrue_dE = Rtrue[zk][ok][pk] / energy_bin["centers"]
            dRtrue_dE_au = Rtrue[zk][ok][pk] / energy_bin["centers"]
            dRtrue_dE_upper = dRtrue_dE + dRtrue_dE_au
            dRtrue_dE_lower = dRtrue_dE - dRtrue_dE_au

            alpha = 0.25
            sebplt.ax_add_histogram(
                ax=ax,
                bin_edges=energy_bin["edges"],
                bincounts=dRtrue_dE,
                bincounts_upper=dRtrue_dE_upper,
                bincounts_lower=dRtrue_dE_lower,
                linecolor=res.PARTICLE_COLORS[pk],
                linealpha=alpha,
                linestyle=":",
                face_color=res.PARTICLE_COLORS[pk],
                face_alpha=alpha * 0.25,
            )

        ax.set_ylabel(r"differential rate / s$^{-1}$ (GeV)$^{-1}$")
        ax.set_xlabel(r"reco. energy / GeV")
        ax.set_ylim([1e-5, 1e3])
        ax.loglog()
        fig.savefig(
            opj(
                res.paths["out_dir"],
                f"{zk:s}_{ok:s}_differential_rates_vs_reco_energy.jpg",
            )
        )
        sebplt.close(fig)

res.stop()
