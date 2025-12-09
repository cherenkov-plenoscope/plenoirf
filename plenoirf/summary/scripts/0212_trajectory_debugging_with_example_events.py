#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import plenopy as pl
import gamma_ray_reconstruction as gamrec
import sebastians_matplotlib_addons as sebplt
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

passing_trigger = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)
zenith_assignment = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0019_zenith_bin_assignment")
)
zenith_bin = res.zenith_binning("3_bins_per_45deg")

fuzzy_config = gamrec.trajectory.v2020nov12fuzzy0.config.compile_user_config(
    user_config=res.config["reconstruction"]["trajectory"]["fuzzy_method"]
)
model_fit_config = (
    gamrec.trajectory.v2020dec04iron0b.config.compile_user_config(
        user_config=res.config["reconstruction"]["trajectory"]["core_axis_fit"]
    )
)
onreion_config = res.analysis["on_off_measuremnent"]["onregion_types"]["large"]

lfg = pl.LightFieldGeometry(
    opj(
        res.paths["plenoirf_dir"],
        "plenoptics",
        "instruments",
        res.instrument_key,
        "light_field_geometry",
    )
)
fov_radius_deg = np.rad2deg(
    0.5 * lfg.sensor_plane2imaging_system.max_FoV_diameter
)


def add_axes_fuzzy_debug(ax, ring_binning, fuzzy_result, fuzzy_debug):
    azi = fuzzy_result["main_axis_azimuth"]
    ax.plot(
        np.rad2deg(ring_binning["bin_edges"]),
        fuzzy_debug["azimuth_ring_smooth"],
        "k",
    )
    ax.plot(np.rad2deg(azi), 1.0, "or")

    unc = 0.5 * fuzzy_result["main_axis_azimuth_uncertainty"]
    ax.plot(np.rad2deg([azi - unc, azi + unc]), [0.5, 0.5], "-r")

    ax.set_xlim([0, 360])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("main-axis-azimuth / deg")
    ax.set_ylabel("probability density / 1")


axes_style = {"spines": [], "axes": ["x", "y"], "grid": True}


def read_shower_maximum_object_distance(
    site_key, particle_key, key="image_smallest_ellipse_object_distance"
):
    event_table = snt.read(
        path=opj(
            res.paths["plenoirf_dir"],
            "event_table",
            site_key,
            particle_key,
            "event_table.tar",
        ),
        structure=irf.table.STRUCTURE,
    )

    return (
        irf.production.estimate_primary_trajectory.get_column_as_dict_by_index(
            table=event_table, level_key="features", column_key=key
        )
    )


PLOT_RING = False
PLOT_OVERVIEW = True
PLOT_ONREGION = True

NUM_EVENTS_PER_PARTICLE = 10

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    for pk in res.PARTICLES:
        os.makedirs(opj(res.paths["out_dir"], zk, pk), exist_ok=True)

        uid_common = snt.logic.intersection(
            passing_trigger[pk]["uid"],
            zenith_assignment[zk][pk],
            passing_quality[pk]["uid"],
        )

        with res.open_event_table(particle_key=pk) as arc:
            event_table = arc.query(
                levels_and_columns={
                    "primary": (
                        "uid",
                        "azimuth_rad",
                        "zenith_rad",
                        "energy_GeV",
                    ),
                    "instrument_pointing": (
                        "uid",
                        "azimuth_rad",
                        "zenith_rad",
                    ),
                    "groundgrid_choice": ("uid", "core_x_m", "core_y_m"),
                    "reconstructed_trajectory": (
                        "uid",
                        "cx_rad",
                        "cy_rad",
                        "x_m",
                        "y_m",
                    ),
                    "features": (
                        "uid",
                        "image_smallest_ellipse_object_distance",
                        "image_half_depth_shift_cx",
                        "image_half_depth_shift_cy",
                    ),
                }
            )
            uid_common = snt.logic.intersection(
                *(
                    [uid_common]
                    + [event_table[tab]["uid"] for tab in event_table]
                )
            )
            event_table = snt.logic.cut_and_sort_table_on_indices(
                table=event_table,
                common_indices=uid_common,
            )

        uid_common = set(uid_common)

        loph_path = opj(
            res.response_path(particle_key=pk),
            "reconstructed_cherenkov.loph.tar",
        )

        event_counter = 0
        with pl.photon_stream.loph.LopfTarReader(loph_path) as run:

            while event_counter <= NUM_EVENTS_PER_PARTICLE:
                airshower_uid, loph_record = next(run)

                if airshower_uid not in uid_common:
                    continue
                else:
                    event_counter += 1

                event_entry = snt.logic.cut_and_sort_table_on_indices(
                    table=event_table,
                    common_indices=[airshower_uid],
                )
                fit, debug = gamrec.trajectory.v2020dec04iron0b.estimate(
                    loph_record=loph_record,
                    light_field_geometry=lfg,
                    shower_maximum_object_distance=event_entry["features"][
                        "image_smallest_ellipse_object_distance"
                    ][0],
                    fuzzy_config=fuzzy_config,
                    model_fit_config=model_fit_config,
                )

                if not gamrec.trajectory.v2020dec04iron0b.is_valid_estimate(
                    fit
                ):
                    print(
                        "airshower_uid",
                        airshower_uid,
                        " Can not reconstruct trajectory",
                    )

                # true response
                # -------------
                event_truth_entry = irf.reconstruction.trajectory_quality.make_rectangular_table(
                    event_table=event_entry,
                    instrument_pointing_model=res.config["pointing"]["model"],
                )
                true_response = gamrec.trajectory.v2020dec04iron0b.model_response_for_true_trajectory(
                    true_cx=event_truth_entry["true_trajectory/cx_rad"][0],
                    true_cy=event_truth_entry["true_trajectory/cy_rad"][0],
                    true_x=event_truth_entry["true_trajectory/x_m"][0],
                    true_y=event_truth_entry["true_trajectory/y_m"][0],
                    loph_record=loph_record,
                    light_field_geometry=lfg,
                    model_fit_config=model_fit_config,
                )

                if PLOT_RING:
                    fig = sebplt.figure(sebplt.FIGURE_16_9)
                    ax = sebplt.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
                    add_axes_fuzzy_debug(
                        ax=ax,
                        ring_binning=fuzzy_config["azimuth_ring"],
                        fuzzy_result=debug["fuzzy_result"],
                        fuzzy_debug=debug["fuzzy_debug"],
                    )
                    path = opj(
                        res.paths["out_dir"],
                        zk,
                        pk,
                        "{:09d}_ring.jpg".format(
                            airshower_uid,
                        ),
                    )
                    fig.savefig(path)
                    sebplt.close(fig)

                if PLOT_OVERVIEW:
                    split_light_field = (
                        pl.split_light_field.make_split_light_field(
                            loph_record=loph_record, light_field_geometry=lfg
                        )
                    )

                    fit_cx_deg = np.rad2deg(fit["primary_particle_cx"])
                    fit_cy_deg = np.rad2deg(fit["primary_particle_cy"])
                    fit_x = fit["primary_particle_x"]
                    fit_y = fit["primary_particle_y"]

                    fig = sebplt.figure(sebplt.FIGURE_16_9)
                    ax = sebplt.add_axes(
                        fig=fig, span=[0.075, 0.1, 0.4, 0.8], style=axes_style
                    )
                    ax_core = sebplt.add_axes(
                        fig=fig, span=[0.575, 0.1, 0.4, 0.8], style=axes_style
                    )
                    for pax in range(split_light_field["number_paxel"]):
                        ax.plot(
                            np.rad2deg(
                                split_light_field["image_sequences"][pax][:, 0]
                            ),
                            np.rad2deg(
                                split_light_field["image_sequences"][pax][:, 1]
                            ),
                            "xb",
                            alpha=0.03,
                        )
                    ax.pcolor(
                        np.rad2deg(fuzzy_config["image"]["c_bin_edges"]),
                        np.rad2deg(fuzzy_config["image"]["c_bin_edges"]),
                        debug["fuzzy_debug"]["fuzzy_image_smooth"],
                        cmap="Reds",
                    )
                    sebplt.ax_add_grid(ax)
                    sebplt.ax_add_circle(ax=ax, x=0.0, y=0.0, r=fov_radius_deg)
                    ax.plot(
                        [
                            np.rad2deg(fit["main_axis_support_cx"]),
                            np.rad2deg(fit["main_axis_support_cx"])
                            + 100 * np.cos(fit["main_axis_azimuth"]),
                        ],
                        [
                            np.rad2deg(fit["main_axis_support_cy"]),
                            np.rad2deg(fit["main_axis_support_cy"])
                            + 100 * np.sin(fit["main_axis_azimuth"]),
                        ],
                        ":c",
                    )

                    ax.plot(
                        np.rad2deg(debug["fuzzy_result"]["reco_cx"]),
                        np.rad2deg(debug["fuzzy_result"]["reco_cy"]),
                        "og",
                    )
                    ax.plot(fit_cx_deg, fit_cy_deg, "oc")
                    ax.plot(
                        np.rad2deg(
                            event_truth_entry[
                                "reconstructed_trajectory/cx_rad"
                            ][0]
                        ),
                        np.rad2deg(
                            event_truth_entry[
                                "reconstructed_trajectory/cy_rad"
                            ][0]
                        ),
                        "xk",
                    )

                    if PLOT_ONREGION:
                        onregion = (
                            irf.reconstruction.onregion.estimate_onregion(
                                reco_cx=fit["primary_particle_cx"],
                                reco_cy=fit["primary_particle_cy"],
                                reco_main_axis_azimuth=fit[
                                    "main_axis_azimuth"
                                ],
                                reco_num_photons=len(
                                    loph_record["photons"][
                                        "arrival_time_slices"
                                    ]
                                ),
                                reco_core_radius=np.hypot(
                                    fit["primary_particle_x"],
                                    fit["primary_particle_y"],
                                ),
                                config=onreion_config,
                            )
                        )

                        ellxy = irf.reconstruction.onregion.make_polygon(
                            onregion=onregion
                        )

                        hit = irf.reconstruction.onregion.is_direction_inside(
                            cx=event_truth_entry[
                                "reconstructed_trajectory/cx_rad"
                            ][0],
                            cy=event_truth_entry[
                                "reconstructed_trajectory/cy_rad"
                            ][0],
                            onregion=onregion,
                        )

                        if hit:
                            look = "c"
                        else:
                            look = ":c"

                        ax.plot(
                            np.rad2deg(ellxy[:, 0]),
                            np.rad2deg(ellxy[:, 1]),
                            look,
                        )

                    info_str = ""
                    info_str += "Energy: {: .1f}GeV, ".format(
                        event_truth_entry["primary/energy_GeV"][0]
                    )
                    info_str += "reco. Cherenkov: {: 4d}p.e.\n ".format(
                        loph_record["photons"]["channels"].shape[0]
                    )
                    info_str += (
                        "response of shower model: {:.4f} ({:.4f})".format(
                            fit["shower_model_response"],
                            true_response,
                        )
                    )

                    ax.set_title(info_str)

                    ax.set_xlim(
                        [-1.05 * fov_radius_deg, 1.05 * fov_radius_deg]
                    )
                    ax.set_ylim(
                        [-1.05 * fov_radius_deg, 1.05 * fov_radius_deg]
                    )
                    ax.set_aspect("equal")
                    ax.set_xlabel(r"$cx$ / 1$^{\circ}$")
                    ax.set_ylabel(r"$cy$ / 1$^{\circ}$")

                    ax_core.plot(fit_x, fit_y, "oc")
                    ax_core.plot([0, fit_x], [0, fit_y], "c", alpha=0.5)

                    ax_core.plot(
                        event_truth_entry["true_trajectory/x_m"][0],
                        event_truth_entry["true_trajectory/y_m"][0],
                        "xk",
                    )
                    ax_core.plot(
                        [0, event_truth_entry["true_trajectory/x_m"][0]],
                        [0, event_truth_entry["true_trajectory/y_m"][0]],
                        "k",
                        alpha=0.5,
                    )

                    ax_core.set_xlim([-640, 640])
                    ax_core.set_ylim([-640, 640])
                    ax_core.set_aspect("equal")
                    ax_core.set_xlabel(r"$x$ / m")
                    ax_core.set_ylabel(r"$y$ / m")
                    path = opj(
                        res.paths["out_dir"],
                        zk,
                        pk,
                        "{:09d}.jpg".format(
                            airshower_uid,
                        ),
                    )

                    fig.savefig(path)
                    sebplt.close(fig)

res.stop()
