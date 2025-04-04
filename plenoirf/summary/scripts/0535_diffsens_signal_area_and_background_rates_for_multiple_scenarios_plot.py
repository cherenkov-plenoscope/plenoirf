#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import flux_sensitivity
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(
    run_dir=paths["plenoirf_dir"]
)
sum_config = irf.summary.read_summary_config(summary_dir=paths["analysis_dir"])
sebplt.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(paths["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
COSMIC_RAYS = irf.utils.filter_particles_with_electric_charge(PARTICLES)
ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

# load
# ----
energy_binning = json_utils.read(
    opj(paths["analysis_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]

acceptance = json_utils.tree.read(
    opj(paths["analysis_dir"], "0300_onregion_trigger_acceptance")
)

scenarios = json_utils.tree.read(
    opj(
        paths["analysis_dir"],
        "0534_diffsens_signal_area_and_background_rates_for_multiple_scenarios",
    )
)

# plot
# ----
for sk in SITES:
    for ok in ONREGION_TYPES:
        os.makedirs(opj(paths["out_dir"], sk, ok), exist_ok=True)


for sk in SITES:
    for ok in ONREGION_TYPES:
        for dk in flux_sensitivity.differential.SCENARIOS:
            elabel = flux_sensitivity.differential.SCENARIOS[dk][
                "energy_axes_label"
            ]

            fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
            ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
            for ck in COSMIC_RAYS:
                ck_Rt = scenarios[sk][ok][dk][ck]["rate"]["mean"]
                ck_Rt_au = scenarios[sk][ok][dk][ck]["rate"][
                    "absolute_uncertainty"
                ]
                sebplt.ax_add_histogram(
                    ax=ax,
                    bin_edges=energy_bin["edges"],
                    bincounts=ck_Rt,
                    linestyle="-",
                    linecolor=sum_config["plot"]["particle_colors"][ck],
                    linealpha=1.0,
                    bincounts_upper=ck_Rt + ck_Rt_au,
                    bincounts_lower=ck_Rt - ck_Rt_au,
                    face_color=sum_config["plot"]["particle_colors"][ck],
                    face_alpha=0.2,
                    label=None,
                    draw_bin_walls=False,
                )
            ax.set_ylabel("rate / s$^{-1}$")
            ax.set_xlabel("reco. energy / GeV")
            ax.set_ylim([1e-5, 1e5])
            ax.loglog()
            fig.savefig(
                opj(
                    paths["out_dir"],
                    sk,
                    ok,
                    dk + "_background_rate_vs_reco_energy.jpg",
                )
            )
            sebplt.close(fig)

            fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
            ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
            A_gamma_scenario = scenarios[sk][ok][dk]["gamma"]["area"]["mean"]
            A_gamma_scenario_au = scenarios[sk][ok][dk]["gamma"]["area"][
                "absolute_uncertainty"
            ]

            A_gamma = acceptance[sk][ok]["gamma"]["point"]["mean"]
            A_gamma_au = acceptance[sk][ok]["gamma"]["point"][
                "absolute_uncertainty"
            ]

            sebplt.ax_add_histogram(
                ax=ax,
                bin_edges=energy_bin["edges"],
                bincounts=A_gamma_scenario,
                linestyle="-",
                linecolor=sum_config["plot"]["particle_colors"]["gamma"],
                linealpha=1.0,
                bincounts_upper=A_gamma_scenario + A_gamma_scenario_au,
                bincounts_lower=A_gamma_scenario - A_gamma_scenario_au,
                face_color=sum_config["plot"]["particle_colors"]["gamma"],
                face_alpha=0.2,
                label=None,
                draw_bin_walls=False,
            )
            ax.plot(energy_bin["centers"], A_gamma, "+k")
            ax.set_ylabel("area / m$^{2}$")
            ax.set_xlabel(elabel + "energy / GeV")
            ax.set_ylim([1e0, 1e6])
            ax.loglog()
            fig.savefig(
                opj(
                    paths["out_dir"],
                    sk,
                    ok,
                    dk + "_area_gamma.jpg",
                )
            )
            sebplt.close(fig)

            # G_matrix
            # ---------------------------
            G_matrix = scenarios[sk][ok][dk]["gamma"]["scenario"]["G_matrix"]
            fig = sebplt.figure(sebplt.FIGURE_1_1)
            ax_c = sebplt.add_axes(fig=fig, span=[0.16, 0.16, 0.7, 0.7])
            ax_cb = sebplt.add_axes(fig=fig, span=[0.88, 0.16, 0.02, 0.7])
            _pcm_confusion = ax_c.pcolormesh(
                energy_bin["edges"],
                energy_bin["edges"],
                np.transpose(G_matrix),
                cmap="Greys",
                norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
                vmin=0,
                vmax=1,
            )
            ax_c.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
            sebplt.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
            ax_c.set_aspect("equal")
            ax_c.set_ylabel("reco. energy / GeV")
            ax_c.loglog()
            ax_c.set_xlabel("energy / GeV")
            fig.savefig(
                opj(
                    paths["out_dir"],
                    sk,
                    ok,
                    dk + "_G_matrix.jpg",
                )
            )
            sebplt.close(fig)

            # B_matrix
            # --------
            B_matrix = scenarios[sk][ok][dk]["gamma"]["scenario"]["B_matrix"]
            fig = sebplt.figure(sebplt.FIGURE_1_1)
            ax_c = sebplt.add_axes(fig=fig, span=[0.16, 0.16, 0.7, 0.7])
            ax_cb = sebplt.add_axes(fig=fig, span=[0.88, 0.16, 0.02, 0.7])
            _pcm_confusion = ax_c.pcolormesh(
                energy_bin["edges"],
                energy_bin["edges"],
                np.transpose(B_matrix),
                cmap="Greys",
                norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
                vmin=0,
                vmax=1,
            )
            ax_c.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
            sebplt.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
            ax_c.set_aspect("equal")
            ax_c.set_ylabel("reco. energy / GeV")
            ax_c.loglog()
            ax_c.set_xlabel("energy / GeV")
            fig.savefig(
                opj(
                    paths["out_dir"],
                    sk,
                    ok,
                    dk + "_B_matrix.jpg",
                )
            )
            sebplt.close(fig)
