#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

trigger = res.trigger

# trigger
# -------
A = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"],
        "0100_trigger_acceptance_for_cosmic_particles",
    )
)

# trigger fix onregion
# --------------------
G = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0300_onregion_trigger_acceptance")
)

A_energy_bin = res.energy_binning(key="10_bins_per_decade")
G_energy_bin = res.energy_binning(key="5_bins_per_decade")
zenith_bin = res.zenith_binning("3_bins_per_45deg")


ONREGION_TYPES = res.analysis["on_off_measuremnent"]["onregion_types"]

idx_trigger_threshold = np.where(
    np.array(trigger["ratescan_thresholds_pe"]) == trigger["threshold_pe"],
)[0][0]
assert trigger["threshold_pe"] in trigger["ratescan_thresholds_pe"]

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    for ok in ONREGION_TYPES:
        for gk in irf.summary.figure.SOURCES:
            fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
            ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

            sebplt.add_axes_zenith_range_indicator(
                fig=fig,
                span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
                zenith_bin_edges_rad=zenith_bin["edges"],
                zenith_bin=zd,
                fontsize=6,
            )

            text_y = 0
            for pk in res.PARTICLES:
                Q = G[zk][ok][pk][gk]["mean"]
                Q_au = G[zk][ok][pk][gk]["absolute_uncertainty"]

                sebplt.ax_add_histogram(
                    ax=ax,
                    bin_edges=G_energy_bin["edges"],
                    bincounts=Q,
                    linestyle="-",
                    linecolor=res.PARTICLE_COLORS[pk],
                    bincounts_upper=Q + Q_au,
                    bincounts_lower=Q - Q_au,
                    face_color=res.PARTICLE_COLORS[pk],
                    face_alpha=0.25,
                )

                ax.text(
                    0.9,
                    0.1 + text_y,
                    pk,
                    color=res.PARTICLE_COLORS[pk],
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
            fig.savefig(opj(res.paths["out_dir"], f"{zk:s}_{ok:s}_{gk:s}.jpg"))
            sebplt.close(fig)


compare_trigger_level_dir = opj(res.paths["out_dir"], "compare_trigger")
os.makedirs(compare_trigger_level_dir)

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    for pk in res.PARTICLES:
        for gk in irf.summary.figure.SOURCES:
            acc_trg = A[zk][pk][gk]["mean"][idx_trigger_threshold]

            acc_trg_au = A[zk][pk][gk]["absolute_uncertainty"][
                idx_trigger_threshold
            ]

            for ok in ONREGION_TYPES:
                acc_trg_onregion = G[zk][ok][pk][gk]["mean"]
                acc_trg_onregion_au = G[zk][ok][pk][gk]["absolute_uncertainty"]

                fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
                ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

                sebplt.add_axes_zenith_range_indicator(
                    fig=fig,
                    span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
                    zenith_bin_edges_rad=zenith_bin["edges"],
                    zenith_bin=zd,
                    fontsize=6,
                )

                sebplt.ax_add_histogram(
                    ax=ax,
                    bin_edges=A_energy_bin["edges"],
                    bincounts=acc_trg,
                    linestyle="-",
                    linecolor="gray",
                    bincounts_upper=acc_trg + acc_trg_au,
                    bincounts_lower=acc_trg - acc_trg_au,
                    face_color=res.PARTICLE_COLORS[pk],
                    face_alpha=0.05,
                )
                sebplt.ax_add_histogram(
                    ax=ax,
                    bin_edges=G_energy_bin["edges"],
                    bincounts=acc_trg_onregion,
                    linestyle="-",
                    linecolor=res.PARTICLE_COLORS[pk],
                    bincounts_upper=acc_trg_onregion + acc_trg_onregion_au,
                    bincounts_lower=acc_trg_onregion - acc_trg_onregion_au,
                    face_color=res.PARTICLE_COLORS[pk],
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
                        compare_trigger_level_dir,
                        f"{zk:s}_{ok:s}_{pk:s}_{gk:s}.jpg",
                    )
                )
                sebplt.close(fig)

res.stop()
