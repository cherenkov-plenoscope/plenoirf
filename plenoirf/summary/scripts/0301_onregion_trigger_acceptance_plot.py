#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
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
TRIGGER = sum_config["trigger"]

# trigger
# -------
A = json_utils.tree.read(
    opj(pa["summary_dir"], "0100_trigger_acceptance_for_cosmic_particles")
)

# trigger fix onregion
# --------------------
G = json_utils.tree.read(
    opj(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

energy_binning = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
A_energy_bin = energy_binning["trigger_acceptance"]
G_energy_bin = energy_binning["trigger_acceptance_onregion"]

ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

particle_colors = sum_config["plot"]["particle_colors"]


for sk in SITES:
    trigger_thresholds = TRIGGER[sk]["ratescan_thresholds_pe"]
    trigger_threshold = TRIGGER[sk]["threshold_pe"]
    idx_trigger_threshold = np.where(
        np.array(trigger_thresholds) == trigger_threshold,
    )[0][0]
    assert trigger_threshold in trigger_thresholds

    for ok in ONREGION_TYPES:
        for gk in irf.summary.figure.SOURCES:
            fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
            ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

            text_y = 0
            for pk in PARTICLES:
                Q = G[sk][ok][pk][gk]["mean"]
                Q_au = G[sk][ok][pk][gk]["absolute_uncertainty"]

                seb.ax_add_histogram(
                    ax=ax,
                    bin_edges=G_energy_bin["edges"],
                    bincounts=Q,
                    linestyle="-",
                    linecolor=particle_colors[pk],
                    bincounts_upper=Q + Q_au,
                    bincounts_lower=Q - Q_au,
                    face_color=particle_colors[pk],
                    face_alpha=0.25,
                )

                ax.text(
                    0.9,
                    0.1 + text_y,
                    pk,
                    color=particle_colors[pk],
                    transform=ax.transAxes,
                )
                text_y += 0.06

            ax.set_xlabel("energy / GeV")
            ax.set_ylabel(
                "{:s} / {:s}".format(
                    irf.summary.figure.SOURCES[gk]["label"],
                    irf.summary.figure.SOURCES[gk]["unit"],
                )
            )
            ax.set_ylim(
                irf.summary.figure.SOURCES[gk]["limits"]["passed_all_cuts"]
            )
            ax.loglog()
            ax.set_xlim(G_energy_bin["limits"])

            fig.savefig(
                os.path.join(
                    pa["out_dir"],
                    "{:s}_{:s}_{:s}.jpg".format(sk, ok, gk),
                )
            )
            seb.close(fig)


for sk in irf_config["config"]["sites"]:
    for pk in irf_config["config"]["particles"]:
        for gk in irf.summary.figure.SOURCES:
            acc_trg = A[sk][pk][gk]["mean"][idx_trigger_threshold]

            acc_trg_au = A[sk][pk][gk]["absolute_uncertainty"][
                idx_trigger_threshold
            ]

            for ok in ONREGION_TYPES:
                acc_trg_onregion = G[sk][ok][pk][gk]["mean"]
                acc_trg_onregion_au = G[sk][ok][pk][gk]["absolute_uncertainty"]

                fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
                ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

                seb.ax_add_histogram(
                    ax=ax,
                    bin_edges=A_energy_bin["edges"],
                    bincounts=acc_trg,
                    linestyle="-",
                    linecolor="gray",
                    bincounts_upper=acc_trg + acc_trg_au,
                    bincounts_lower=acc_trg - acc_trg_au,
                    face_color=particle_colors[pk],
                    face_alpha=0.05,
                )
                seb.ax_add_histogram(
                    ax=ax,
                    bin_edges=G_energy_bin["edges"],
                    bincounts=acc_trg_onregion,
                    linestyle="-",
                    linecolor=particle_colors[pk],
                    bincounts_upper=acc_trg_onregion + acc_trg_onregion_au,
                    bincounts_lower=acc_trg_onregion - acc_trg_onregion_au,
                    face_color=particle_colors[pk],
                    face_alpha=0.25,
                )

                ax.text(
                    s="onregion-radius at 100p.e.: {:.3f}".format(
                        ONREGION_TYPES[ok]["opening_angle_deg"]
                    )
                    + r"$^{\circ}$",
                    x=0.1,
                    y=0.1,
                    transform=ax.transAxes,
                )
                ax.set_xlabel("energy / GeV")
                ax.set_ylabel(
                    "{:s} / {:s}".format(
                        irf.summary.figure.SOURCES[gk]["label"],
                        irf.summary.figure.SOURCES[gk]["unit"],
                    )
                )
                ax.set_ylim(
                    irf.summary.figure.SOURCES[gk]["limits"]["passed_trigger"]
                )
                ax.loglog()
                ax.set_xlim(A_energy_bin["limits"])
                fig.savefig(
                    opj(
                        pa["out_dir"],
                        "{:s}_{:s}_{:s}_{:s}.jpg".format(sk, ok, pk, gk),
                    )
                )
                seb.close(fig)
