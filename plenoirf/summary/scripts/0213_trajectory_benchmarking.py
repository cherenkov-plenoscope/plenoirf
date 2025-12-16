#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import pandas
import plenopy as pl
import iminuit
import scipy
import sebastians_matplotlib_addons as sebplt
import json_utils

"""
Objective
=========

Quantify the angular resolution of the plenoscope.

Input
-----
- List of reconstructed gamma-ray-directions
- List of true gamma-ray-directions, energy, and more...

Quantities
----------
- theta
- theta parallel component
- theta perpendicular component

histogram theta2
----------------
- in energy
- in core-radius

"""

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

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

# energy
# ------
energy_bin = res.energy_binning(key="5_bins_per_decade")

# core-radius bins
# ----------------
core_radius_square_bin_edges_m2 = np.linspace(
    start=0.0,
    stop=res.analysis["point_spread_function"]["core_radius"]["max_radius_m"]
    ** 2,
    num=res.analysis["point_spread_function"]["core_radius"]["num_bins"] + 1,
)
num_core_radius_bins = core_radius_square_bin_edges_m2.shape[0] - 1

# theta square bins
# -----------------
theta_square_bin_edges_deg2 = np.linspace(
    start=0.0,
    stop=res.analysis["point_spread_function"]["theta_square"]["max_angle_deg"]
    ** 2,
    num=res.analysis["point_spread_function"]["theta_square"]["num_bins"] + 1,
)

psf_containment_factor = res.analysis["point_spread_function"][
    "containment_factor"
]
pivot_energy_GeV = res.analysis["point_spread_function"]["pivot_energy_GeV"]

# psf image bins
# --------------
num_c_bins = 32
fov_radius_deg = 3.05
fov_radius_fine_deg = (1.0 / 5.0) * fov_radius_deg
c_bin_edges_deg = np.linspace(-fov_radius_deg, fov_radius_deg, num_c_bins)
c_bin_edges_fine_deg = np.linspace(
    -fov_radius_fine_deg, fov_radius_fine_deg, num_c_bins
)

theta_square_max_deg = (
    res.analysis["point_spread_function"]["theta_square"]["max_angle_deg"] ** 2
)

num_containment_fractions = 20
containment_fractions = np.linspace(0.0, 1.0, num_containment_fractions + 1)[
    1:-1
]


def energy_range_string(start_GeV, stop_GeV):
    start_GeV = float(start_GeV)
    stop_GeV = float(stop_GeV)

    if start_GeV < 1:
        # MeV regime
        scale = 1e3
        unit = "MeV"
    elif start_GeV < 1000:
        # GeV regime
        scale = 1
        unit = "GeV"
    else:
        # TeV regime
        scale = 1e-3
        unit = "TeV"

    start = start_GeV * scale
    stop = stop_GeV * scale

    if start < 10:
        num_decimals = 3
    elif start < 100:
        num_decimals = 2
    else:
        num_decimals = 1

    float_template = f"7.{num_decimals:d}f"

    range_string_template = (
        "[{:" + float_template + "}, {:" + float_template + "})"
    )
    range_string = range_string_template.format(start, stop)

    out = range_string + r"$\,$" + unit
    return out


def empty_dim2(dim0, dim1):
    return [[None for ii in range(dim1)] for jj in range(dim0)]


def estimate_containments_theta_deg(
    containment_fractions,
    theta_deg,
):
    conta_deg = np.nan * np.ones(containment_fractions.shape[0])
    conta_deg_relunc = np.nan * np.ones(containment_fractions.shape[0])
    for con in range(containment_fractions.shape[0]):
        ca = irf.analysis.gamma_direction.estimate_containment_radius(
            theta_deg=theta_deg,
            psf_containment_factor=containment_fractions[con],
        )
        conta_deg[con] = ca[0]
        conta_deg_relunc[con] = ca[1]
    return conta_deg, conta_deg_relunc


def guess_theta_square_bin_edges_deg(
    theta_square_max_deg,
    theta_deg,
    num_min=10,
    num_max=2048,
):
    num_t2_bins = int(np.sqrt(theta_deg.shape[0]))
    num_t2_bins = np.max([num_min, num_t2_bins])

    it = 0
    while True:
        it += 1
        if it > 64:
            break

        theta_square_bin_edges_deg2 = np.linspace(
            start=0.0,
            stop=theta_square_max_deg,
            num=num_t2_bins + 1,
        )

        bc = irf.analysis.gamma_direction.histogram_theta_square(
            theta_deg=theta_deg,
            theta_square_bin_edges_deg2=theta_square_bin_edges_deg2,
        )[0]

        # print("it1", it, "bc", bc[0:num_min])

        if np.argmax(bc) != 0:
            break

        if bc[0] > 2.0 * bc[1] and bc[1] > 0:
            num_t2_bins = int(1.1 * num_t2_bins)
            num_t2_bins = np.min([num_max, num_t2_bins])
        else:
            break

    it2 = 0
    while True:
        it2 += 1
        if it2 > 64:
            break

        # print("it2", it, "bc", bc[0:num_min])

        theta_square_bin_edges_deg2 = np.linspace(
            start=0.0,
            stop=theta_square_max_deg,
            num=num_t2_bins + 1,
        )

        bc = irf.analysis.gamma_direction.histogram_theta_square(
            theta_deg=theta_deg,
            theta_square_bin_edges_deg2=theta_square_bin_edges_deg2,
        )[0]

        if np.sum(bc[0:num_min] == 0) > 0.33 * num_min:
            num_t2_bins = int(0.8 * num_t2_bins)
            num_t2_bins = np.max([num_min, num_t2_bins])
        else:
            break

    return theta_square_bin_edges_deg2


psf_ax_style = {"spines": [], "axes": ["x", "y"], "grid": True}

for pk in res.PARTICLES:
    pk_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    uid_common = snt.logic.intersection(
        passing_trigger[pk]["uid"],
        passing_quality[pk]["uid"],
        passing_trajectory_quality[pk]["trajectory_quality"]["uid"],
    )

    event_table = res.event_table(particle_key=pk).query(
        levels_and_columns={
            "primary": "__all__",
            "instrument_pointing": "__all__",
            "groundgrid_choice": "__all__",
            "reconstructed_trajectory": "__all__",
            "features": "__all__",
        },
        indices=uid_common,
        sort=True,
    )

    reconstructed_event_table = (
        irf.reconstruction.trajectory_quality.make_rectangular_table(
            event_table=event_table,
            instrument_pointing_model=res.config["pointing"]["model"],
        )
    )

    rectab = reconstructed_event_table

    # theta-square vs energy vs core-radius
    # -------------------------------------

    hist_ene_rad = {
        "energy_bin_edges_GeV": energy_bin["edges"],
        "core_radius_square_bin_edges_m2": core_radius_square_bin_edges_m2,
        "histogram": empty_dim2(energy_bin["num"], num_core_radius_bins),
    }

    hist_ene = {
        "energy_bin_edges_GeV": energy_bin["edges"],
        "histogram": [None for ii in range(energy_bin["num"])],
    }

    cont_ene_rad = {
        "energy_bin_edges_GeV": energy_bin["edges"],
        "core_radius_square_bin_edges_m2": core_radius_square_bin_edges_m2,
        "containment_fractions": containment_fractions,
        "containment": empty_dim2(energy_bin["num"], num_core_radius_bins),
    }
    cont_ene = {
        "energy_bin_edges_GeV": energy_bin["edges"],
        "containment_fractions": containment_fractions,
        "containment": [None for ii in range(energy_bin["num"])],
    }

    for the in ["theta", "theta_para", "theta_perp"]:
        h_ene_rad = dict(hist_ene_rad)
        h_ene = dict(hist_ene)

        c_ene_rad = dict(cont_ene_rad)
        c_ene = dict(cont_ene)

        for ene in range(energy_bin["num"]):
            energy_start = energy_bin["edges"][ene]
            energy_stop = energy_bin["edges"][ene + 1]
            ene_mask = np.logical_and(
                rectab["primary/energy_GeV"] >= energy_start,
                rectab["primary/energy_GeV"] < energy_stop,
            )

            the_key = "trajectory/" + the + "_rad"
            ene_theta_deg = np.rad2deg(rectab[the_key][ene_mask])
            ene_theta_deg = np.abs(ene_theta_deg)

            ene_theta_square_bin_edges_deg2 = guess_theta_square_bin_edges_deg(
                theta_square_max_deg=theta_square_max_deg,
                theta_deg=ene_theta_deg,
                num_min=10,
                num_max=2**12,
            )

            ene_hi = irf.analysis.gamma_direction.histogram_theta_square(
                theta_deg=ene_theta_deg,
                theta_square_bin_edges_deg2=ene_theta_square_bin_edges_deg2,
            )
            h_ene["histogram"][ene] = {
                "theta_square_bin_edges_deg2": ene_theta_square_bin_edges_deg2,
                "intensity": ene_hi[0],
                "intensity_relative_uncertainty": ene_hi[1],
            }

            ene_co = estimate_containments_theta_deg(
                containment_fractions=containment_fractions,
                theta_deg=ene_theta_deg,
            )
            c_ene["containment"][ene] = {
                "theta_deg": ene_co[0],
                "theta_deg_relative_uncertainty": ene_co[1],
            }

            for rad in range(num_core_radius_bins):
                radius_sq_start = core_radius_square_bin_edges_m2[rad]
                radius_sq_stop = core_radius_square_bin_edges_m2[rad + 1]

                rad_mask = np.logical_and(
                    rectab["true_trajectory/r_m"] ** 2 >= radius_sq_start,
                    rectab["true_trajectory/r_m"] ** 2 < radius_sq_stop,
                )

                ene_rad_mask = np.logical_and(ene_mask, rad_mask)
                ene_rad_theta_deg = np.rad2deg(rectab[the_key][ene_rad_mask])
                ene_rad_theta_deg = np.abs(ene_rad_theta_deg)

                ene_rad_theta_square_bin_edges_deg2 = (
                    guess_theta_square_bin_edges_deg(
                        theta_square_max_deg=theta_square_max_deg,
                        theta_deg=ene_rad_theta_deg,
                        num_min=10,
                        num_max=2**12,
                    )
                )

                ene_rad_hi = irf.analysis.gamma_direction.histogram_theta_square(
                    theta_deg=ene_rad_theta_deg,
                    theta_square_bin_edges_deg2=ene_rad_theta_square_bin_edges_deg2,
                )
                h_ene_rad["histogram"][ene][rad] = {
                    "theta_square_bin_edges_deg2": ene_rad_theta_square_bin_edges_deg2,
                    "intensity": ene_rad_hi[0],
                    "intensity_relative_uncertainty": ene_rad_hi[1],
                }

                ene_rad_co = estimate_containments_theta_deg(
                    containment_fractions=containment_fractions,
                    theta_deg=ene_rad_theta_deg,
                )
                c_ene_rad["containment"][ene][rad] = {
                    "theta_deg": ene_rad_co[0],
                    "theta_deg_relative_uncertainty": ene_rad_co[1],
                }

        json_utils.write(
            opj(
                pk_dir,
                "{theta_key:s}_square_histogram_vs_energy_vs_core_radius.json".format(
                    theta_key=the
                ),
            ),
            h_ene_rad,
        )

        json_utils.write(
            opj(
                pk_dir,
                "{theta_key:s}_square_histogram_vs_energy.json".format(
                    theta_key=the
                ),
            ),
            h_ene,
        )

        json_utils.write(
            opj(
                pk_dir,
                "{theta_key:s}_containment_vs_energy_vs_core_radius.json".format(
                    theta_key=the
                ),
            ),
            c_ene_rad,
        )

        json_utils.write(
            opj(
                pk_dir,
                "{theta_key:s}_containment_vs_energy.json".format(
                    theta_key=the
                ),
            ),
            c_ene,
        )

    # image of point-spread-function
    # -------------------------------

    delta_cx_deg = np.rad2deg(
        rectab["reconstructed_trajectory/cx_rad"]
        - rectab["true_trajectory/cx_rad"]
    )
    delta_cy_deg = np.rad2deg(
        rectab["reconstructed_trajectory/cy_rad"]
        - rectab["true_trajectory/cy_rad"]
    )

    num_panels = energy_bin["num"] + 1
    num_cols = 3
    num_rows = num_panels // num_cols
    num_pixel_on_edge = 427

    fig = sebplt.figure(
        {
            "rows": (1 + num_rows) * num_pixel_on_edge,
            "cols": num_cols * num_pixel_on_edge,
            "fontsize": 1.2,
        }
    )
    _colw = 1.0 / num_cols
    _colh = 1.0 / num_rows
    fov_shrink = 0.7
    fov_radius_shrink_deg = fov_radius_deg * fov_shrink
    c_bin_edges_shrink_deg = np.linspace(
        -fov_radius_shrink_deg,
        fov_radius_shrink_deg,
        num_c_bins,
    )
    y_global_shift = 0.01

    for ene in range(num_panels):
        _xi = np.mod(ene, num_cols)
        _yi = ene // num_cols

        _xx = _xi * _colw
        _yy = 1.0 - ((_yi + 1) * _colh) + y_global_shift

        ax1 = sebplt.add_axes(
            fig=fig,
            span=[_xx, _yy, _colw * 0.95, _colh * 0.95],
            style={"spines": [], "axes": [], "grid": False},
        )

        if ene == energy_bin["num"]:
            fig.text(
                s="1$^{\circ}$",
                x=_xx + 0.5 * _colw,
                y=_yy + 0.5 * _colh,
            )
            ax1.plot(
                [
                    0,
                    1,
                ],
                [
                    0,
                    0,
                ],
                color="black",
                linestyle="-",
            )
        else:
            ene_start = energy_bin["edges"][ene]
            ene_stop = energy_bin["edges"][ene + 1]

            fig.text(
                s=energy_range_string(start_GeV=ene_start, stop_GeV=ene_stop),
                x=_xx,
                y=_yy - y_global_shift / 2,
            )

            ene_mask = np.logical_and(
                rectab["primary/energy_GeV"] >= ene_start,
                rectab["primary/energy_GeV"] < ene_stop,
            )

            ene_delta_cx_deg = delta_cx_deg[ene_mask]
            ene_delta_cy_deg = delta_cy_deg[ene_mask]

            ene_psf_image = np.histogram2d(
                ene_delta_cx_deg,
                ene_delta_cy_deg,
                bins=(c_bin_edges_shrink_deg, c_bin_edges_shrink_deg),
            )[0]
            ax1.pcolor(
                c_bin_edges_shrink_deg,
                c_bin_edges_shrink_deg,
                ene_psf_image,
                cmap="magma_r",
                vmax=None,
            )

        sebplt.ax_add_grid_with_explicit_ticks(
            ax=ax1,
            xticks=np.linspace(-2, 2, 5),
            yticks=np.linspace(-2, 2, 5),
            color="k",
            linestyle="-",
            linewidth=0.33,
            alpha=0.11,
        )
        ax1.set_aspect("equal")
        _frs = fov_radius_shrink_deg
        ax1.set_xlim([-1.01 * _frs, 1.01 * _frs])
        ax1.set_ylim([-1.01 * _frs, 1.01 * _frs])

    fig.savefig(
        opj(
            res.paths["out_dir"],
            "{:s}_psf_image_all.jpg".format(pk),
        )
    )
    sebplt.close(fig)

res.stop()
