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
import binning_utils
import atmospheric_cherenkov_response

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

zenith_bin = res.zenith_binning("once")

plenoscope_trigger_vs_cherenkov_density = json_utils.tree.read(
    opj(
        res.paths["analysis_dir"],
        "0074_trigger_probability_vs_cherenkov_density_on_ground",
    )
)
zenith_assignment = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0019_zenith_bin_assignment")
)


prng = np.random.Generator(np.random.PCG64(res.analysis["random_seed"]))

ground_grid_geometry = irf.ground_grid.GroundGrid(
    bin_width_m=res.config["ground_grid"]["geometry"]["bin_width_m"],
    num_bins_each_axis=res.config["ground_grid"]["geometry"][
        "num_bins_each_axis"
    ],
    center_x_m=0,
    center_y_m=0,
)

plenoscope_mirror_radius_m = res.instrument["light_field_sensor_geometry"][
    "expected_imaging_system_aperture_radius"
]
plenoscope_mirror_area_m2 = np.pi * plenoscope_mirror_radius_m**2

ARRAY_CONFIGS = copy.deepcopy(
    res.analysis["outer_telescope_array_configurations"]
)

for ak in ARRAY_CONFIGS:
    ARRAY_CONFIGS[ak]["mask"] = (
        irf.outer_telescope_array.init_mask_from_telescope_positions(
            positions=ARRAY_CONFIGS[ak]["positions"],
        )
    )

CB = irf.outer_telescope_array.init_binning()["center_bin"]
NB = irf.outer_telescope_array.init_binning()["num_bins_on_edge"]

ROI_RADIUS = np.ceil(3) + 1
for ak in ARRAY_CONFIGS:
    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(
        fig=fig,
        span=irf.summary.figure.AX_SPAN,
        style={
            "spines": ["left", "bottom"],
            "axes": ["x", "y"],
            "grid": False,
        },
    )
    sebplt.ax_add_grid_with_explicit_ticks(
        ax=ax,
        xticks=np.arange(-CB - 0.5, CB + 1.5, 1),
        yticks=np.arange(-CB - 0.5, CB + 1.5, 1),
        color="k",
        linestyle="-",
        linewidth=0.11,
        alpha=0.33,
    )
    sebplt.ax_add_circle(
        ax=ax,
        x=0,
        y=0,
        r=plenoscope_mirror_radius_m / ground_grid_geometry["bin_width_m"],
        linewidth=1.0,
        linestyle="-",
        color="k",
        alpha=1,
        num_steps=128,
    )
    for iix in np.arange(NB):
        for iiy in np.arange(NB):
            if ARRAY_CONFIGS[ak]["mask"][iix, iiy]:
                sebplt.ax_add_circle(
                    ax=ax,
                    x=iix - CB,
                    y=iiy - CB,
                    r=(
                        0.5
                        * ARRAY_CONFIGS[ak]["mirror_diameter_m"]
                        / ground_grid_geometry["bin_width_m"]
                    ),
                    linewidth=1.0,
                    linestyle="-",
                    color="k",
                    alpha=1,
                    num_steps=128,
                )
    ax.set_xlim([-ROI_RADIUS, ROI_RADIUS])
    ax.set_ylim([-ROI_RADIUS, ROI_RADIUS])
    ax.set_aspect("equal")
    ax.set_xlabel("x / {:.1f}m".format(ground_grid_geometry["bin_width_m"]))
    ax.set_ylabel("y / {:.1f}m".format(ground_grid_geometry["bin_width_m"]))
    fig.savefig(
        opj(
            res.paths["out_dir"],
            "array_configuration_" + ak + ".jpg",
        )
    )
    sebplt.close(fig)


# estimate trigger probability of individual telescope in array
# -------------------------------------------------------------
KEY = "passing_trigger_if_only_accepting_not_rejecting"
telescope_trigger = {}
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    telescope_trigger[zk] = {}
    for pk in res.PARTICLES:
        telescope_trigger[zk][pk] = {}

        pleno_prb = plenoscope_trigger_vs_cherenkov_density[zk][pk][KEY][
            "mean"
        ]
        pleno_den_bin_edges = plenoscope_trigger_vs_cherenkov_density[zk][pk][
            KEY
        ]["Cherenkov_density_bin_edges_per_m2"]
        pleno_den = binning_utils.centers(bin_edges=pleno_den_bin_edges)

        for ak in ARRAY_CONFIGS:
            assert (
                ARRAY_CONFIGS[ak]["mirror_diameter_m"]
                < ground_grid_geometry["bin_width_m"]
            ), "telescope mirror must not exceed grid-cell."

            telescope_mirror_area_m2 = (
                np.pi * (0.5 * ARRAY_CONFIGS[ak]["mirror_diameter_m"]) ** 2
            )

            _tprb = plenoscope_trigger_vs_cherenkov_density[zk][pk][KEY][
                "mean"
            ]
            _tprb = irf.utils.fill_nans_from_end(arr=_tprb, val=1.0)
            _tprb = irf.utils.fill_nans_from_start(arr=_tprb, val=0.0)
            _tden = (
                plenoscope_mirror_area_m2 / telescope_mirror_area_m2
            ) * pleno_den

            telescope_trigger[zk][pk][ak] = {
                "probability": _tprb,
                "cherenkov_density_per_m2": _tden,
            }


# plot trigger probability of individual telescope in array
# ---------------------------------------------------------
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    for ak in ARRAY_CONFIGS:
        fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
        ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
        sebplt.add_axes_zenith_range_indicator(
            fig=fig,
            span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
            zenith_bin_edges_rad=zenith_bin["edges"],
            zenith_bin=zd,
            fontsize=6,
        )

        for pk in res.PARTICLES:
            ax.plot(
                telescope_trigger[zk][pk][ak]["cherenkov_density_per_m2"],
                telescope_trigger[zk][pk][ak]["probability"],
                color=res.PARTICLE_COLORS[pk],
                linestyle="-",
            )
        ax.semilogx()
        ax.semilogy()
        ax.set_xlim([np.min(pleno_den_bin_edges), np.max(pleno_den_bin_edges)])
        ax.set_ylim([1e-6, 1.5e-0])
        ax.set_xlabel("density of Cherenkov-photons / m$^{-2}$")
        ax.set_ylabel("telescope\ntrigger-probability / 1")
        fig.savefig(
            opj(
                res.paths["out_dir"],
                f"{zk:s}_{ak:s}_telescope_trigger_probability.jpg",
            )
        )
        sebplt.close(fig)


def histogram_ground_grid_intensity(
    intensity, bin_idx_x, bin_idx_y, num_bins_radius
):
    x_bins = np.arange(
        bin_idx_x - num_bins_radius, bin_idx_x + num_bins_radius + 2
    )
    y_bins = np.arange(
        bin_idx_y - num_bins_radius, bin_idx_y + num_bins_radius + 2
    )
    h = np.histogram2d(
        x=intensity["x_bin"],
        y=intensity["y_bin"],
        bins=(x_bins, y_bins),
        weights=intensity["size"],
    )[0]
    return h


# simulate telescope triggers
# ---------------------------

for zk in zenith_assignment:
    for pk in zenith_assignment[zk]:
        zenith_assignment[zk][pk] = set(zenith_assignment[zk][pk])

out = {}
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    out[zk] = {}
    for pk in res.PARTICLES:
        out[zk][pk] = {}
        for ak in ARRAY_CONFIGS:
            out[zk][pk][ak] = []

for pk in res.PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        _event_table = arc.query(
            levels_and_columns={
                "groundgrid_choice": ["uid", "bin_idx_x", "bin_idx_y"],
            }
        )
        groundgrid_choice_by_uid = {}
        for item in _event_table["groundgrid_choice"]:
            groundgrid_choice_by_uid[item["uid"]] = (
                item["bin_idx_x"],
                item["bin_idx_y"],
            )

    with irf.ground_grid.intensity.Reader(
        path=opj(
            res.response_path(particle_key=pk),
            "ground_grid_intensity_roi.zip",
        )
    ) as grid_reader:

        for shower_uid in grid_reader:

            bin_idx_x, bin_idx_y = groundgrid_choice_by_uid[shower_uid]
            grid_cherenkov_intensity = histogram_ground_grid_intensity(
                intensity=grid_reader[shower_uid],
                bin_idx_x=bin_idx_x,
                bin_idx_y=bin_idx_y,
                num_bins_radius=(NB - 1) // 2,
            )
            assert grid_cherenkov_intensity.shape == (NB, NB)
            grid_cherenkov_density_per_m2 = (
                grid_cherenkov_intensity / ground_grid_geometry["bin_area_m2"]
            )

            for zk in zenith_assignment:
                if shower_uid in zenith_assignment[zk]:
                    break

            for ak in ARRAY_CONFIGS:
                num_teles = np.sum(ARRAY_CONFIGS[ak]["mask"])
                array_den = grid_cherenkov_density_per_m2[
                    ARRAY_CONFIGS[ak]["mask"]
                ]
                telescope_trigger_probability = np.interp(
                    array_den,
                    xp=telescope_trigger[zk][pk][ak][
                        "cherenkov_density_per_m2"
                    ],
                    fp=telescope_trigger[zk][pk][ak]["probability"],
                )
                uniform = prng.uniform(size=num_teles)
                trg = np.any(telescope_trigger_probability > uniform)
                if trg:
                    out[zk][pk][ak].append(shower_uid)
                    print(
                        zk,
                        pk,
                        ak,
                        irf.bookkeeping.uid.UID_FOTMAT_STR.format(shower_uid),
                    )
                    break

# export triggers
# ---------------
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    for pk in res.PARTICLES:
        for ak in ARRAY_CONFIGS:
            os.makedirs(opj(res.paths["out_dir"], zk, pk, ak), exist_ok=True)

            json_utils.write(
                opj(res.paths["out_dir"], zk, pk, ak, "idx.json"),
                out[zk][pk][ak],
            )

res.stop()
