#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
import json_utils
import spherical_coordinates
import solid_angle_utils
import binning_utils
import spherical_histogram
import sebastians_matplotlib_addons as sebplt


paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)
sebplt.matplotlib.rcParams.update(res.analysis["plot"]["matplotlib"])

energy_bin = res.energy_binning(key="trigger_acceptance")
zenith_bin = res.zenith_binning(key="once")


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


thrown = {}
containment = {}
for pk in res.PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        primary = arc.query(
            levels_and_columns={
                "primary": [
                    "uid",
                    "energy_GeV",
                    "azimuth_rad",
                    "zenith_rad",
                    "containment_quantile_in_solid_angle_thrown",
                    "solid_angle_thrown_sr",
                ]
            }
        )["primary"]

    thrown[pk] = init_hist(energy_bin=energy_bin)
    containment[pk] = init_hist(energy_bin=energy_bin)
    for eee in range(energy_bin["num"]):
        energy_start_GeV = energy_bin["edges"][eee]
        energy_stop_GeV = energy_bin["edges"][eee + 1]
        energy_mask = np.logical_and(
            primary["energy_GeV"] >= energy_start_GeV,
            primary["energy_GeV"] < energy_stop_GeV,
        )

        (
            containment[pk]["p16"][eee],
            containment[pk]["p50"][eee],
            containment[pk]["p84"][eee],
        ) = p16_p50_p84(
            primary["containment_quantile_in_solid_angle_thrown"][energy_mask]
        )

        (
            thrown[pk]["p16"][eee],
            thrown[pk]["p50"][eee],
            thrown[pk]["p84"][eee],
        ) = p16_p50_p84(primary["solid_angle_thrown_sr"][energy_mask])


PLOTS = {
    "containment_quantile_in_solid_angle_thrown": {
        "hist": containment,
        "y_label": "solid angle containment / 1",
        "y_lim": [0, 1],
    },
    "solid_angle_thrown_sr": {
        "hist": thrown,
        "y_label": "solid angle thrown / sr",
        "y_lim": [0, 0.5],
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
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lim)
    ax.semilogx()
    ax.set_xlim(energy_bin["limits"])
    fig.savefig(os.path.join(paths["out_dir"], f"{key:s}.jpg"))
    sebplt.close(fig)
