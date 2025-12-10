#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import rename_after_writing as rnw
import os
from os.path import join as opj
import json_utils
import spherical_coordinates
import solid_angle_utils
import binning_utils
import spherical_histogram
import sebastians_matplotlib_addons as sebplt


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

energy_bin = res.energy_binning(key="10_bins_per_decade")

max_energy_in_magnetic_delfection_tables_GeV = (
    binning_utils.power10.lower_bin_edge(
        **res.config["magnetic_deflection"]["energy_stop_GeV_power10"]
    )
)


def percentile(x, p):
    if len(x) == 0:
        return float("nan")
    return float(np.percentile(x, p))


def p16_p50_p84(x):
    return percentile(x, p=16), percentile(x, p=50), percentile(x, p=84)


def init_hist(energy_bin):
    return {
        "p16": np.zeros(energy_bin["num"]),
        "p50": np.zeros(energy_bin["num"]),
        "p84": np.zeros(energy_bin["num"]),
    }


thrown_cache_path = os.path.join(res.paths["cache_dir"], "thrown.json")
containment_cache_path = os.path.join(
    res.paths["cache_dir"], "containment.json"
)

if not os.path.exists(thrown_cache_path) or not os.path.exists(
    containment_cache_path
):
    thrown = {}
    containment = {}
    for pk in res.PARTICLES:
        thrown[pk] = init_hist(energy_bin=energy_bin)
        containment[pk] = init_hist(energy_bin=energy_bin)
        for ebin in range(energy_bin["num"]):
            energy_start_GeV = energy_bin["edges"][ebin]
            energy_stop_GeV = energy_bin["edges"][ebin + 1]

            primary = res.event_table(particle_key=pk).query(
                energy_start_GeV=energy_start_GeV,
                energy_stop_GeV=energy_stop_GeV,
                levels_and_columns={
                    "primary": [
                        "uid",
                        "containment_quantile_in_solid_angle_thrown",
                        "solid_angle_thrown_sr",
                    ]
                },
            )["primary"]

            (
                containment[pk]["p16"][ebin],
                containment[pk]["p50"][ebin],
                containment[pk]["p84"][ebin],
            ) = p16_p50_p84(
                primary["containment_quantile_in_solid_angle_thrown"]
            )

            (
                thrown[pk]["p16"][ebin],
                thrown[pk]["p50"][ebin],
                thrown[pk]["p84"][ebin],
            ) = p16_p50_p84(primary["solid_angle_thrown_sr"])

    with rnw.open(thrown_cache_path, "wt") as f:
        f.write(json_utils.dumps(thrown))
    with rnw.open(containment_cache_path, "wt") as f:
        f.write(json_utils.dumps(containment))

thrown = json_utils.read(thrown_cache_path)
containment = json_utils.read(containment_cache_path)

PLOTS = {
    "containment_quantile_in_solid_angle_thrown": {
        "hist": containment,
        "y_label": "solid angle containment / 1",
        "y_lim": [0, 1],
        "loglog": False,
    },
    "solid_angle_thrown_sr": {
        "hist": thrown,
        "y_label": "solid angle thrown / sr",
        "y_lim": [5e-3, 5e-1],
        "loglog": True,
    },
}

for key in PLOTS:
    hist = PLOTS[key]["hist"]
    y_label = PLOTS[key]["y_label"]
    y_lim = PLOTS[key]["y_lim"]

    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in res.PARTICLES:
        sebplt.ax_add_histogram(
            ax=ax,
            bin_edges=energy_bin["edges"],
            bincounts=hist[pk]["p50"],
            linestyle="-",
            linecolor=res.PARTICLE_COLORS[pk],
            bincounts_upper=hist[pk]["p16"],
            bincounts_lower=hist[pk]["p84"],
            face_color=res.PARTICLE_COLORS[pk],
            face_alpha=0.25,
        )
    ax.axvline(
        max_energy_in_magnetic_delfection_tables_GeV,
        linestyle=":",
        color="black",
        alpha=0.3,
    )
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lim)
    if PLOTS[key]["loglog"]:
        ax.loglog()
    else:
        ax.semilogx()
    ax.set_xlim(energy_bin["limits"])
    fig.savefig(opj(res.paths["out_dir"], f"{key:s}.jpg"))
    sebplt.close(fig)

res.stop()
