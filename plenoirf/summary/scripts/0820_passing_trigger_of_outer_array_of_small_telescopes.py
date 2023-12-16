#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import sebastians_matplotlib_addons as seb
import os
import copy
import json_utils
import numpy as np
import binning_utils
import atmospheric_cherenkov_response

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
PLT = sum_config["plot"]

os.makedirs(pa["out_dir"], exist_ok=True)

plenoscope_trigger_vs_cherenkov_density = json_utils.tree.read(
    os.path.join(
        pa["summary_dir"],
        "0074_trigger_probability_vs_cherenkov_density_on_ground",
    )
)

prng = np.random.Generator(
    np.random.generator.PCG64(sum_config["random_seed"])
)

grid_bin_area_m2 = irf_config["grid_geometry"]["bin_area"]

plenoscope_mirror_diameter_m = (
    irf_config["grid_geometry"]["bin_width"]
    / irf_config["config"]["grid"]["bin_width_overhead"]
)
plenoscope_mirror_area_m2 = np.pi * (0.5 * plenoscope_mirror_diameter_m) ** 2

ARRAY_CONFIGS = copy.deepcopy(
    sum_config["outer_telescope_array_configurations"]
)

for ak in ARRAY_CONFIGS:
    ARRAY_CONFIGS[ak][
        "mask"
    ] = irf.outer_telescope_array.init_mask_from_telescope_positions(
        positions=ARRAY_CONFIGS[ak]["positions"],
    )

CB = irf.outer_telescope_array.CENTER_BIN
NB = irf.outer_telescope_array.NUM_BINS_ON_EDGE

ROI_RADIUS = np.ceil(3) + 1
for ak in ARRAY_CONFIGS:
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(
        fig=fig,
        span=irf.summary.figure.AX_SPAN,
        style={
            "spines": ["left", "bottom"],
            "axes": ["x", "y"],
            "grid": False,
        },
    )
    seb.ax_add_grid_with_explicit_ticks(
        ax=ax,
        xticks=np.arange(-CB - 0.5, CB + 1.5, 1),
        yticks=np.arange(-CB - 0.5, CB + 1.5, 1),
        color="k",
        linestyle="-",
        linewidth=0.11,
        alpha=0.33,
    )
    seb.ax_add_circle(
        ax=ax,
        x=0,
        y=0,
        r=0.5
        * plenoscope_mirror_diameter_m
        / irf_config["grid_geometry"]["bin_width"],
        linewidth=1.0,
        linestyle="-",
        color="k",
        alpha=1,
        num_steps=128,
    )
    for iix in np.arange(NB):
        for iiy in np.arange(NB):
            if ARRAY_CONFIGS[ak]["mask"][iix, iiy]:
                seb.ax_add_circle(
                    ax=ax,
                    x=iix - CB,
                    y=iiy - CB,
                    r=(
                        0.5
                        * ARRAY_CONFIGS[ak]["mirror_diameter_m"]
                        / irf_config["grid_geometry"]["bin_width"]
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
    ax.set_xlabel(
        "x / {:.1f}m".format(irf_config["grid_geometry"]["bin_width"])
    )
    ax.set_ylabel(
        "y / {:.1f}m".format(irf_config["grid_geometry"]["bin_width"])
    )
    fig.savefig(
        os.path.join(
            pa["out_dir"],
            "array_configuration_" + ak + ".jpg",
        )
    )
    seb.close(fig)


# estimate trigger probability of individual telescope in array
# -------------------------------------------------------------
KEY = "passing_trigger_if_only_accepting_not_rejecting"
telescope_trigger = {}
for sk in SITES:
    telescope_trigger[sk] = {}
    for pk in PARTICLES:
        telescope_trigger[sk][pk] = {}

        pleno_prb = plenoscope_trigger_vs_cherenkov_density[sk][pk][KEY][
            "mean"
        ]
        pleno_den_bin_edges = plenoscope_trigger_vs_cherenkov_density[sk][pk][
            KEY
        ]["Cherenkov_density_bin_edges_per_m2"]
        pleno_den = binning_utils.centers(bin_edges=pleno_den_bin_edges)

        for ak in ARRAY_CONFIGS:
            assert (
                ARRAY_CONFIGS[ak]["mirror_diameter_m"]
                < irf_config["grid_geometry"]["bin_width"]
            ), "telescope mirror must not exceed grid-cell."

            telescope_mirror_area_m2 = (
                np.pi * (0.5 * ARRAY_CONFIGS[ak]["mirror_diameter_m"]) ** 2
            )

            _tprb = plenoscope_trigger_vs_cherenkov_density[sk][pk][KEY][
                "mean"
            ]
            _tprb = irf.utils.fill_nans_from_end(arr=_tprb, val=1.0)
            _tprb = irf.utils.fill_nans_from_start(arr=_tprb, val=0.0)
            _tden = (
                plenoscope_mirror_area_m2 / telescope_mirror_area_m2
            ) * pleno_den

            telescope_trigger[sk][pk][ak] = {
                "probability": _tprb,
                "cherenkov_density_per_m2": _tden,
            }


# plot trigger probability of individual telescope in array
# ---------------------------------------------------------
for sk in SITES:
    for ak in ARRAY_CONFIGS:
        fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
        for pk in PARTICLES:
            ax.plot(
                telescope_trigger[sk][pk][ak]["cherenkov_density_per_m2"],
                telescope_trigger[sk][pk][ak]["probability"],
                color=PLT["particle_colors"][pk],
                linestyle="-",
            )
        ax.semilogx()
        ax.semilogy()
        ax.set_xlim([np.min(pleno_den_bin_edges), np.max(pleno_den_bin_edges)])
        ax.set_ylim([1e-6, 1.5e-0])
        ax.set_xlabel("density of Cherenkov-photons / m$^{-2}$")
        ax.set_ylabel("telescope\ntrigger-probability / 1")
        fig.savefig(
            os.path.join(
                pa["out_dir"],
                sk + "_" + ak + "_telescope_trigger_probability" + ".jpg",
            )
        )
        seb.close(fig)


# simulate telescope triggers
# ---------------------------
out = {}
for sk in SITES:
    out[sk] = {}
    for pk in PARTICLES:
        out[sk][pk] = {}
        for ak in ARRAY_CONFIGS:
            out[sk][pk][ak] = []

        grid_reader = (
            atmospheric_cherenkov_response.grid.serialization.GridReader(
                path=os.path.join(
                    pa["run_dir"],
                    "event_table",
                    sk,
                    pk,
                    "grid_roi_pasttrigger.tar",
                )
            )
        )

        for shower in grid_reader:
            shower_idx, grid_cherenkov_intensity = shower
            assert grid_cherenkov_intensity.shape == (NB, NB)
            grid_cherenkov_density_per_m2 = (
                grid_cherenkov_intensity / grid_bin_area_m2
            )

            for ak in ARRAY_CONFIGS:
                num_teles = np.sum(ARRAY_CONFIGS[ak]["mask"])
                array_den = grid_cherenkov_density_per_m2[
                    ARRAY_CONFIGS[ak]["mask"]
                ]
                telescope_trigger_probability = np.interp(
                    array_den,
                    xp=telescope_trigger[sk][pk][ak][
                        "cherenkov_density_per_m2"
                    ],
                    fp=telescope_trigger[sk][pk][ak]["probability"],
                )
                uniform = prng.uniform(size=num_teles)
                trg = np.any(telescope_trigger_probability > uniform)
                if trg:
                    out[sk][pk][ak].append(shower_idx)
                    print(
                        sk,
                        pk,
                        ak,
                        irf.unique.UID_FOTMAT_STR.format(shower_idx),
                    )
                    break

# export triggers
# ---------------
for sk in SITES:
    for pk in PARTICLES:
        for ak in ARRAY_CONFIGS:
            sk_pk_ak_dir = os.path.join(pa["out_dir"], sk, pk, ak)
            os.makedirs(sk_pk_ak_dir, exist_ok=True)

            json_utils.write(
                path=os.path.join(sk_pk_ak_dir, "idx.json"),
                out_dict=out[sk][pk][ak],
            )
