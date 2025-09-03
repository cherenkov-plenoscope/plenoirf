#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import sebastians_matplotlib_addons as sebplt
import os
from os.path import join as opj
import copy
import json_utils
import numpy as np
import pandas
import propagate_uncertainties as pu

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

energy_bin = res.energy_binning(key="point_spread_function")
zenith_bin = res.zenith_binning("once")

passing_array_trigger = json_utils.tree.read(
    opj(
        res.paths["analysis_dir"],
        "0820_passing_trigger_of_outer_array_of_small_telescopes",
    )
)
_passing_plenoscope_trigger = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)
zenith_assignment = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0019_zenith_bin_assignment")
)


ARRAY_CONFIGS = copy.deepcopy(
    res.analysis["outer_telescope_array_configurations"]
)

AX_SPAN = list(irf.summary.figure.AX_SPAN)
AX_SPAN[3] = AX_SPAN[3] * 0.85


passing_plenoscope_trigger = {}
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    passing_plenoscope_trigger[zk] = {}
    for pk in res.PARTICLES:
        passing_plenoscope_trigger[zk][pk] = {}
        passing_plenoscope_trigger[zk][pk]["uid"] = snt.logic.intersection(
            zenith_assignment[zk][pk],
            _passing_plenoscope_trigger[pk]["uid"],
        )

pv = {}
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    pv[zk] = {}
    for pk in res.PARTICLES:
        pv[zk][pk] = {}

        with res.open_event_table(particle_key=pk) as arc:
            event_table = arc.query(
                levels_and_columns={
                    "primary": ["uid", "energy_GeV"],
                }
            )

        for ak in ARRAY_CONFIGS:
            print("estimate trigger ratio", zk, pk, ak)

            passing_plenoscope_and_not_array = np.array(
                list(
                    set.difference(
                        set(passing_plenoscope_trigger[zk][pk]["uid"]),
                        set(passing_array_trigger[zk][pk][ak]["uid"]),
                    )
                )
            )

            pleno_table = snt.logic.cut_table_on_indices(
                table=event_table,
                common_indices=passing_plenoscope_trigger[zk][pk]["uid"],
            )

            veto_table = snt.logic.cut_table_on_indices(
                table=event_table,
                common_indices=passing_plenoscope_and_not_array,
            )

            pv[zk][pk][ak] = {}
            pv[zk][pk][ak]["num_plenoscope"] = np.histogram(
                pleno_table["primary"]["energy_GeV"],
                bins=energy_bin["edges"],
            )[0]
            pv[zk][pk][ak]["num_plenoscope_au"] = np.sqrt(
                pv[zk][pk][ak]["num_plenoscope"]
            )

            pv[zk][pk][ak]["num_outer_array"] = np.histogram(
                veto_table["primary"]["energy_GeV"],
                bins=energy_bin["edges"],
            )[0]
            pv[zk][pk][ak]["num_outer_array_au"] = np.sqrt(
                pv[zk][pk][ak]["num_outer_array"]
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                (
                    pv[zk][pk][ak]["ratio"],
                    pv[zk][pk][ak]["ratio_au"],
                ) = pu.divide(
                    x=pv[zk][pk][ak]["num_outer_array"].astype(float),
                    x_au=pv[zk][pk][ak]["num_outer_array_au"],
                    y=pv[zk][pk][ak]["num_plenoscope"].astype(float),
                    y_au=pv[zk][pk][ak]["num_plenoscope_au"],
                )


for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    for ak in ARRAY_CONFIGS:
        print("plot trigger ratio all particles", zk, ak)

        fig = sebplt.figure(style=irf.summary.figure.FIGURE_STYLE)
        ax = sebplt.add_axes(fig=fig, span=AX_SPAN)
        sebplt.add_axes_zenith_range_indicator(
            fig=fig,
            span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
            zenith_bin_edges_rad=zenith_bin["edges"],
            zenith_bin=zd,
            fontsize=6,
        )

        for pk in res.PARTICLES:
            sebplt.ax_add_histogram(
                ax=ax,
                bin_edges=energy_bin["edges"],
                bincounts=pv[zk][pk][ak]["ratio"],
                linestyle="-",
                linecolor=irf.summary.figure.PARTICLE_COLORS[pk],
                linealpha=1.0,
                bincounts_upper=pv[zk][pk][ak]["ratio"]
                + pv[zk][pk][ak]["ratio_au"],
                bincounts_lower=pv[zk][pk][ak]["ratio"]
                - pv[zk][pk][ak]["ratio_au"],
                face_color=irf.summary.figure.PARTICLE_COLORS[pk],
                face_alpha=0.1,
                label=None,
                draw_bin_walls=False,
            )
        ax.semilogx()
        ax.set_xlim(energy_bin["limits"])
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlabel("energy / GeV")
        ax.set_ylabel(
            "trigger(plenoscope)\nAND NOT\nany(trigger(outer telescopes)) / 1"
        )
        fig.savefig(opj(res.paths["out_dir"], f"{zk:s}_{ak:s}.jpg"))
        sebplt.close(fig)

res.stop()
