#!/usr/bin/python
import sys
import os
from os.path import join as opj
import numpy as np
import sparse_numeric_table as spt
import plenoirf as irf
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

passing_trigger = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)

energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["point_spread_function"]

grid_geometry = irf_config["grid_geometry"]

veto_radius_grid_cells = 2
grid_threshold_num_photons = irf_config["config"]["grid"][
    "threshold_num_photons"
]

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

for sk in SITES:
    pv = {}
    for pk in PARTICLES:
        prefix_str = "{:s}_{:s}".format(sk, pk)

        # read
        # ----
        detected_grid_histograms = irf.grid.read_histograms(
            path=opj(
                pa["run_dir"],
                "event_table",
                sk,
                pk,
                "grid.tar",
            ),
            indices=passing_trigger[sk][pk]["idx"],
        )
        idx_passed_trigger_and_in_debug_output = np.array(
            list(detected_grid_histograms.keys())
        )

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

        detected_events = spt.cut_table_on_indices(
            table=event_table,
            common_indices=idx_passed_trigger_and_in_debug_output,
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

        pv[pk] = {}
        pv[pk]["num_thrown"] = np.zeros(energy_bin["num_bins"])
        pv[pk]["num_passed"] = np.zeros(energy_bin["num_bins"])

        particle_counter = 0
        for shower_idx in idx_passed_trigger_and_in_debug_output:
            if particle_counter >= MAX_NUM_PARTICLES:
                break

            shower_table = spt.cut_table_on_indices(
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
            if energy_bin_idx >= energy_bin["num_bins"]:
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
            for iix in np.arange(
                grid_bin_idx_x - gR, grid_bin_idx_x + gR + 1, 1
            ):
                for iiy in np.arange(
                    grid_bin_idx_y - gR, grid_bin_idx_y + gR + 1, 1
                ):
                    if iix >= 0 and iix < grid_geometry["num_bins_diameter"]:
                        if (
                            iiy >= 0
                            and iiy < grid_geometry["num_bins_diameter"]
                        ):
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
                sk,
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

        fig = seb.figure(style=irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=AX_SPAN)
        seb.ax_add_histogram(
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
        fig.savefig(
            os.path.join(
                pa["out_dir"],
                "{:s}_{:s}.jpg".format(sk, pk),
            )
        )
        seb.close(fig)
