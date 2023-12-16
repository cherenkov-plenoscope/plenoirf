#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import sebastians_matplotlib_addons as seb
import os
import copy
import json_utils
import numpy as np
import pandas
import propagate_uncertainties as pu

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
PLT = sum_config["plot"]

os.makedirs(pa["out_dir"], exist_ok=True)

energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["point_spread_function"]

passing_array_trigger = json_utils.tree.read(
    os.path.join(
        pa["summary_dir"],
        "0820_passing_trigger_of_outer_array_of_small_telescopes",
    )
)
passing_plenoscope_trigger = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)

ARRAY_CONFIGS = copy.deepcopy(
    sum_config["outer_telescope_array_configurations"]
)

AX_SPAN = list(irf.summary.figure.AX_SPAN)
AX_SPAN[3] = AX_SPAN[3] * 0.85

pv = {}
for sk in SITES:
    pv[sk] = {}
    for pk in PARTICLES:
        pv[sk][pk] = {}

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

        for ak in ARRAY_CONFIGS:
            print("estimate trigger ratio", sk, pk, ak)

            passing_plenoscope_and_not_array = np.array(
                list(
                    set.difference(
                        set(passing_plenoscope_trigger[sk][pk]["idx"]),
                        set(passing_array_trigger[sk][pk][ak]["idx"]),
                    )
                )
            )

            pleno_table = spt.cut_table_on_indices(
                table=event_table,
                common_indices=passing_plenoscope_trigger[sk][pk]["idx"],
                level_keys=[
                    "primary",
                ],
            )

            veto_table = spt.cut_table_on_indices(
                table=event_table,
                common_indices=passing_plenoscope_and_not_array,
                level_keys=[
                    "primary",
                ],
            )

            pv[sk][pk][ak] = {}
            pv[sk][pk][ak]["num_plenoscope"] = np.histogram(
                pleno_table["primary"]["energy_GeV"],
                bins=energy_bin["edges"],
            )[0]
            pv[sk][pk][ak]["num_plenoscope_au"] = np.sqrt(
                pv[sk][pk][ak]["num_plenoscope"]
            )

            pv[sk][pk][ak]["num_outer_array"] = np.histogram(
                veto_table["primary"]["energy_GeV"],
                bins=energy_bin["edges"],
            )[0]
            pv[sk][pk][ak]["num_outer_array_au"] = np.sqrt(
                pv[sk][pk][ak]["num_outer_array"]
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                (
                    pv[sk][pk][ak]["ratio"],
                    pv[sk][pk][ak]["ratio_au"],
                ) = pu.divide(
                    x=pv[sk][pk][ak]["num_outer_array"].astype(np.float),
                    x_au=pv[sk][pk][ak]["num_outer_array_au"],
                    y=pv[sk][pk][ak]["num_plenoscope"].astype(np.float),
                    y_au=pv[sk][pk][ak]["num_plenoscope_au"],
                )


for sk in SITES:
    for ak in ARRAY_CONFIGS:
        print("plot trigger ratio all particles", sk, ak)

        fig = seb.figure(style=irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=AX_SPAN)
        for pk in PARTICLES:
            seb.ax_add_histogram(
                ax=ax,
                bin_edges=energy_bin["edges"],
                bincounts=pv[sk][pk][ak]["ratio"],
                linestyle="-",
                linecolor=irf.summary.figure.PARTICLE_COLORS[pk],
                linealpha=1.0,
                bincounts_upper=pv[sk][pk][ak]["ratio"]
                + pv[sk][pk][ak]["ratio_au"],
                bincounts_lower=pv[sk][pk][ak]["ratio"]
                - pv[sk][pk][ak]["ratio_au"],
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
        fig.savefig(
            os.path.join(
                pa["out_dir"],
                "{:s}_{:s}.jpg".format(sk, ak),
            )
        )
        seb.close(fig)
