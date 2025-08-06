#!/usr/bin/python
import sys
import os
from os.path import join as opj
import numpy as np
import sparse_numeric_table as snt
import plenoirf as irf
import sebastians_matplotlib_addons as sebplt
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

passing_trigger = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)
energy_bin = res.energy_binning(key="point_spread_function")

grid_geometry = res.config["ground_grid"]["geometry"]

veto_radius_grid_cells = 2
grid_threshold_num_photons = res.config["ground_grid"]["threshold_num_photons"]

PORTAL_TO_CTA_MST_RATIO = 35

VETO_MIRROR_RATIO = 100

VETO_TELESCOPE_TRIGGER_THRESHOLD_NUM_PHOTONS = (
    VETO_MIRROR_RATIO * grid_threshold_num_photons
)

gD = 2 * veto_radius_grid_cells + 1
plenoscope_diameter = (
    grid_geometry["bin_width"]
    / irf_config["config"]["grid"]["bin_width_overhead"]
)
VETO_STR = "outer array {:d} x {:d} telescopes\n".format(gD, gD)
VETO_STR += "spacing {:.1f}m, mirror diameter {:.1f}m.".format(
    grid_geometry["bin_width"],
    plenoscope_diameter / np.sqrt(VETO_MIRROR_RATIO),
)

MAX_NUM_PARTICLES = 250
AX_SPAN = list(irf.summary.figure.AX_SPAN)
AX_SPAN[3] = AX_SPAN[3] * 0.85

pv = {}
for pk in res.PARTICLES:
    ggi_path = opj(
        res.response_path(particle_key=pk), "ground_grid_intensity.zip"
    )

    uid_ggi = []
    ggi = irf.ground_grid.intensity.Reader(ggi_path)
    for uid in ggi:
        uid_ggi.append(uid)

    uid_trigger_ggi = list(
        set.intersection(set(uid_ggi), set(passing_trigger[pk]["uid"]))
    )

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(levels_and_columns={"primary": "__all__"})

    detected_events = snt.logic.cut_table_on_indices(
        table=event_table,
        common_indices=uid_trigger_ggi,
    )

    pv[pk] = {}
    pv[pk]["num_thrown"] = np.zeros(energy_bin["num"])
    pv[pk]["num_passed"] = np.zeros(energy_bin["num"])

    particle_counter = 0
    for shower_idx in idx_passed_trigger_and_in_debug_output:
        if particle_counter >= MAX_NUM_PARTICLES:
            break

        shower_table = snt.cut_table_on_indices(
            table=detected_events,
            common_indices=[shower_idx],
            level_keys=[
                "primary",
                "grid",
                "core",
                "cherenkovsize",
                "cherenkovpool",
                "cherenkovsizepart",
                "cherenkovpoolpart",
                "trigger",
            ],
        )
        energy_GeV = shower_table["primary"]["energy_GeV"][0]

        energy_bin_idx = np.digitize(energy_GeV, bins=energy_bin["edges"])
        if energy_bin_idx >= energy_bin["num"]:
            continue

        assert (
            energy_bin["edges"][energy_bin_idx - 1]
            <= energy_GeV
            < energy_bin["edges"][energy_bin_idx]
        )

        grid_intensity = irf.grid.bytes_to_histogram(
            detected_grid_histograms[shower_idx]
        )
        grid_bin_idx_x = shower_table["core"]["bin_idx_x"][0]
        grid_bin_idx_y = shower_table["core"]["bin_idx_y"][0]

        gR = veto_radius_grid_cells
        grid_bin_idx_veto = []
        for iix in np.arange(grid_bin_idx_x - gR, grid_bin_idx_x + gR + 1, 1):
            for iiy in np.arange(
                grid_bin_idx_y - gR, grid_bin_idx_y + gR + 1, 1
            ):
                if iix >= 0 and iix < grid_geometry["num_bins_diameter"]:
                    if iiy >= 0 and iiy < grid_geometry["num_bins_diameter"]:
                        grid_bin_idx_veto.append((iix, iiy))

        num_veto_trials = len(grid_bin_idx_veto)
        num_veto_triggers = 0

        for bin_idx in grid_bin_idx_veto:
            if (
                grid_intensity[bin_idx]
                >= VETO_TELESCOPE_TRIGGER_THRESHOLD_NUM_PHOTONS
            ):
                num_veto_triggers += 1

        pv[pk]["num_thrown"][energy_bin_idx] += 1
        if num_veto_triggers > 0:
            msg = "         "
        else:
            pv[pk]["num_passed"][energy_bin_idx] += 1
            msg = "LOW GAMMA"

        print(
            pk,
            "{:6.1f}GeV".format(energy_GeV),
            msg,
            num_veto_triggers,
            num_veto_trials,
        )

        particle_counter += 1

    pv[pk]["ratio"] = irf.utils._divide_silent(
        numerator=pv[pk]["num_passed"],
        denominator=pv[pk]["num_thrown"],
        default=float("nan"),
    )
    pv[pk]["ratio_au"] = irf.utils._divide_silent(
        numerator=np.sqrt(pv[pk]["num_passed"]),
        denominator=pv[pk]["num_thrown"],
        default=float("nan"),
    )

    pv[pk]["ratio_upper"] = pv[pk]["ratio"] + pv[pk]["ratio_au"]
    pv[pk]["ratio_lower"] = pv[pk]["ratio"] - pv[pk]["ratio_au"]

    with np.errstate(invalid="ignore"):
        pv[pk]["ratio_upper"][pv[pk]["ratio_upper"] > 1] = 1
        pv[pk]["ratio_lower"][pv[pk]["ratio_lower"] < 0] = 0

    fig = sebplt.figure(style=irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=AX_SPAN)
    sebplt.ax_add_histogram(
        ax=ax,
        bin_edges=energy_bin["edges"],
        bincounts=pv[pk]["ratio"],
        linestyle="-",
        linecolor=irf.summary.figure.PARTICLE_COLORS[pk],
        linealpha=1.0,
        bincounts_upper=pv[pk]["ratio_upper"],
        bincounts_lower=pv[pk]["ratio_lower"],
        face_color=irf.summary.figure.PARTICLE_COLORS[pk],
        face_alpha=0.1,
        label=None,
        draw_bin_walls=False,
    )
    ax.set_title(VETO_STR)
    ax.semilogx()
    ax.set_xlim(energy_bin["limits"])
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel(
        "trigger(plenoscope)\nAND NOT\nany(trigger(outer telescopes)) / 1"
    )
    fig.savefig(opj(res.paths["out_dir"], f"{pk:s}.jpg"))
    sebplt.close(fig)

res.stop()
