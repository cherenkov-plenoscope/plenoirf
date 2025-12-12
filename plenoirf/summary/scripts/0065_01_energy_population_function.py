#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import numpy as np
import binning_utils
import json_utils
import sebastians_matplotlib_addons as sebplt

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

transformed_features_dir = opj(
    res.paths["analysis_dir"], "0062_transform_features"
)
passing_trigger = res.read_passed_trigger(
    opj(res.paths["analysis_dir"], "0055_passing_trigger"),
    trigger_mode_key="far_accepting_focus_and_near_rejecting_focus",
)
passing_quality = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)
passing_trajectory_quality = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0059_passing_trajectory_quality")
)
energy_bin = res.energy_binning(key="trigger_acceptance_onregion")

log10_scale = np.log10(energy_bin["stop"]) - np.log10(energy_bin["start"])
log10_shift = np.log10(energy_bin["start"])

uid_reconstructable = snt.logic.intersection(
    passing_trigger["gamma"]["uid"],
    passing_quality["gamma"]["uid"],
    passing_trajectory_quality["gamma"]["uid"],
)


with res.open_event_table(particle_key="gamma") as arc:
    table = arc.query(
        levels_and_columns={"primary": ["uid", "energy_GeV"]},
        indices=uid_reconstructable,
        sort=True,
    )
np.testing.assert_array_equal(uid_reconstructable, table["primary"]["uid"])
energy_GeV = table["primary"]["energy_GeV"]

x = energy_GeV
lx = np.log10(x)
sx = (lx - log10_shift) / log10_scale

print(f"sx [{min(sx):f}, {max(sx):f}]")

raw_sx_x, raw_sx_x_cumsum = (
    irf.features.scaling.make_bin_edges_and_cumsum_with_sliding_bin_width(x=sx)
)
num_points = raw_sx_x_cumsum.shape[0]
weight = 0.55

sx_x_cumsum = np.zeros(shape=num_points)
sx_x = np.zeros(shape=num_points)
for i in range(num_points):
    f_unity = float(i) / float(num_points)
    f_actual = raw_sx_x_cumsum[i]
    unity_weight = 1.0 - weight
    sx_x_cumsum[i] = f_unity * unity_weight + f_actual * weight

    x_unity = f_unity
    x_actual = raw_sx_x[i]
    sx_x[i] = x_unity * unity_weight + x_actual * weight

sx_x_cumsum /= np.max(sx_x_cumsum)
sx_x /= np.max(sx_x)

path = opj(res.paths["out_dir"], "energy_population_function.json")
json_utils.write(
    path,
    {
        "function": f"(log10(x) {log10_shift:+f}) / {log10_scale:f}",
        "log10_shift": log10_shift,
        "log10_scale": log10_scale,
        "xp": sx_x,
        "fp": sx_x_cumsum,
    },
)

pop = json_utils.read(path)


fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.set_aspect("equal")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
ax.plot(
    pop["xp"],
    pop["fp"],
    color="magenta",
    linestyle="-",
    alpha=0.5,
    label="population",
)
ax.plot(
    pop["fp"],
    pop["xp"],
    color="green",
    linestyle="-",
    alpha=0.5,
    label="inverse",
)
ax.set_ylim([0, 1])
ax.set_xlim([0, 1])
ax.set_xlabel("sx / 1")
ax.legend()
fig.savefig(opj(res.paths["out_dir"], f"sx.jpg"))
sebplt.close(fig)


T_sx = np.interp(xp=pop["xp"], fp=pop["fp"], x=sx)

one_bin = binning_utils.Binning(np.linspace(0, 1, 25))

sx_hist = np.histogram(sx, bins=one_bin["edges"])[0]
T_sx_hist = np.histogram(T_sx, bins=one_bin["edges"])[0]

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
sebplt.ax_add_histogram(
    ax=ax,
    bin_edges=one_bin["edges"],
    bincounts=sx_hist,
    linestyle="-",
    linecolor="black",
    label="sx",
)
sebplt.ax_add_histogram(
    ax=ax,
    bin_edges=one_bin["edges"],
    bincounts=T_sx_hist,
    linestyle="-",
    linecolor="red",
    label="T_sx",
)
ax.set_xlabel("one / 1")
ax.legend()
fig.savefig(opj(res.paths["out_dir"], f"transform.jpg"))
sebplt.close(fig)


back_sx = np.interp(xp=pop["fp"], fp=pop["xp"], x=T_sx)
fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.set_aspect("equal")
ax.scatter(
    sx,
    back_sx,
    color="magenta",
)
ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
ax.set_ylim([0, 1])
ax.set_xlim([0, 1])
ax.set_xlabel("original / 1")
ax.set_ylabel("back / 1")
fig.savefig(opj(res.paths["out_dir"], f"stability.jpg"))
sebplt.close(fig)

res.stop()
