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

cr = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"],
        "0100_trigger_acceptance_for_cosmic_particles",
    )
)

energy_bin = res.energy_binning(key="trigger_acceptance")
zenith_bin = res.zenith_binning("once")

trigger_thresholds = np.array(
    res.analysis["trigger"][res.site_key]["ratescan_thresholds_pe"]
)
analysis_trigger_threshold = res.analysis["trigger"][res.site_key][
    "threshold_pe"
]

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    zd_dir = opj(res.paths["out_dir"], zk)
    os.makedirs(zd_dir, exist_ok=True)

    for source_key in irf.summary.figure.SOURCES:
        for tt in range(len(trigger_thresholds)):
            fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)

            sebplt.add_axes_zenith_range_indicator(
                fig=fig,
                span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
                zenith_bin_edges_rad=zenith_bin["edges"],
                zenith_bin=zd,
                fontsize=6,
            )

            ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

            text_y = 0
            for pk in res.PARTICLES:
                Q = np.array(cr[zk][pk][source_key]["mean"][tt])
                Q_au = np.array(
                    cr[zk][pk][source_key]["absolute_uncertainty"][tt]
                )

                sebplt.ax_add_histogram(
                    ax=ax,
                    bin_edges=energy_bin["edges"],
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
                    irf.summary.figure.SOURCES[source_key]["label"],
                    irf.summary.figure.SOURCES[source_key]["unit"],
                )
            )
            ax.set_ylim(
                irf.summary.figure.SOURCES[source_key]["limits"][
                    "passed_trigger"
                ]
            )
            ax.loglog()
            ax.set_xlim(energy_bin["limits"])

            if trigger_thresholds[tt] == analysis_trigger_threshold:
                fig.savefig(
                    opj(res.paths["out_dir"], f"{source_key:s}_zd{zd:d}.jpg")
                )
            ax.set_title(
                "trigger-threshold: {:d} p.e.".format(trigger_thresholds[tt]),
                fontsize=5,
            )
            fig.savefig(opj(zd_dir, f"{source_key:s}_{tt:06d}.jpg"))
            sebplt.close(fig)

res.stop()
