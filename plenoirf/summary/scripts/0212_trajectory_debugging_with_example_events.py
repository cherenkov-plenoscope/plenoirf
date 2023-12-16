#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import plenopy as pl
import gamma_ray_reconstruction as gamrec
import sebastians_matplotlib_addons as seb
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

passing_trigger = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)

onreion_config = sum_config["on_off_measuremnent"]["onregion_types"]["large"]

# READ light-field-geometry
# =========================
lfg = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
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


fuzzy_config = gamrec.trajectory.v2020nov12fuzzy0.config.compile_user_config(
    user_config=irf_config["config"]["reconstruction"]["trajectory"][
        "fuzzy_method"
    ]
)

long_fit_cfg = gamrec.trajectory.v2020dec04iron0b.config.compile_user_config(
    user_config=irf_config["config"]["reconstruction"]["trajectory"][
        "core_axis_fit"
    ]
)

truth_by_index = {}
for sk in irf_config["config"]["sites"]:
    truth_by_index[sk] = {}
    for pk in irf_config["config"]["particles"]:
        truth_by_index[sk][pk] = {}

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar"
            ),
            structure=irf.table.STRUCTURE,
        )
        common_idx = spt.intersection(
            [passing_trigger[sk][pk]["idx"], passing_quality[sk][pk]["idx"]]
        )
        all_truth = spt.cut_and_sort_table_on_indices(
            event_table,
            common_indices=common_idx,
            level_keys=[
                "primary",
                "cherenkovsize",
                "grid",
                "cherenkovpool",
                "cherenkovsizepart",
                "cherenkovpoolpart",
                "core",
                "trigger",
                "pasttrigger",
                "cherenkovclassification",
            ],
        )

        (
            true_cx,
            true_cy,
        ) = irf.analysis.gamma_direction.momentum_to_cx_cy_wrt_aperture(
            momentum_x_GeV_per_c=all_truth["primary"]["momentum_x_GeV_per_c"],
            momentum_y_GeV_per_c=all_truth["primary"]["momentum_y_GeV_per_c"],
            momentum_z_GeV_per_c=all_truth["primary"]["momentum_z_GeV_per_c"],
            plenoscope_pointing=irf_config["config"]["plenoscope_pointing"],
        )

        for ii in range(all_truth["primary"].shape[0]):
            airshower_id = all_truth["primary"][spt.IDX][ii]
            truth_by_index[sk][pk][airshower_id] = {
                "cx": true_cx[ii],
                "cy": true_cy[ii],
                "x": -all_truth["core"]["core_x_m"][ii],
                "y": -all_truth["core"]["core_y_m"][ii],
                "energy_GeV": all_truth["primary"]["energy_GeV"][ii],
            }

axes_style = {"spines": [], "axes": ["x", "y"], "grid": True}


def read_shower_maximum_object_distance(
    site_key, particle_key, key="image_smallest_ellipse_object_distance"
):
    event_table = spt.read(
        path=os.path.join(
            pa["run_dir"],
            "event_table",
            site_key,
            particle_key,
            "event_table.tar",
        ),
        structure=irf.table.STRUCTURE,
    )

    return spt.get_column_as_dict_by_index(
        table=event_table, level_key="features", column_key=key
    )


PLOT_RING = False
PLOT_OVERVIEW = True
PLOT_ONREGION = True


NUM_EVENTS_PER_PARTICLE = 10


for sk in irf_config["config"]["sites"]:
    for pk in irf_config["config"]["particles"]:
        reco_obj = read_shower_maximum_object_distance(
            site_key=sk, particle_key=pk
        )

        run = pl.photon_stream.loph.LopfTarReader(
            os.path.join(
                pa["run_dir"], "event_table", sk, pk, "cherenkov.phs.loph.tar"
            )
        )

        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        event_counter = 0
        while event_counter <= NUM_EVENTS_PER_PARTICLE:
            event = next(run)
            airshower_id, loph_record = event

            if airshower_id not in truth_by_index[sk][pk]:
                continue
            else:
                event_counter += 1

            truth = dict(truth_by_index[sk][pk][airshower_id])

            fit, debug = gamrec.trajectory.v2020dec04iron0b.estimate(
                loph_record=loph_record,
                light_field_geometry=lfg,
                shower_maximum_object_distance=reco_obj[airshower_id],
                fuzzy_config=fuzzy_config,
                model_fit_config=long_fit_cfg,
            )

            if not gamrec.trajectory.v2020dec04iron0b.is_valid_estimate(fit):
                print(
                    "airshower_id",
                    airshower_id,
                    " Can not reconstruct trajectory",
                )

            # true response
            # -------------

            true_response = gamrec.trajectory.v2020dec04iron0b.model_response_for_true_trajectory(
                true_cx=truth["cx"],
                true_cy=truth["cy"],
                true_x=truth["x"],
                true_y=truth["y"],
                loph_record=loph_record,
                light_field_geometry=lfg,
                model_fit_config=long_fit_cfg,
            )

            if PLOT_RING:
                fig = seb.figure(seb.FIGURE_16_9)
                ax = seb.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
                add_axes_fuzzy_debug(
                    ax=ax,
                    ring_binning=fuzzy_config["azimuth_ring"],
                    fuzzy_result=debug["fuzzy_result"],
                    fuzzy_debug=debug["fuzzy_debug"],
                )
                path = os.path.join(
                    pa["out_dir"],
                    sk,
                    pk,
                    "{:09d}_ring.jpg".format(
                        airshower_id,
                    ),
                )
                fig.savefig(path)
                seb.close(fig)

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

                fig = seb.figure(seb.FIGURE_16_9)
                ax = seb.add_axes(
                    fig=fig, span=[0.075, 0.1, 0.4, 0.8], style=axes_style
                )
                ax_core = seb.add_axes(
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
                seb.ax_add_grid(ax)
                seb.ax_add_circle(ax=ax, x=0.0, y=0.0, r=fov_radius_deg)
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
                ax.plot(np.rad2deg(truth["cx"]), np.rad2deg(truth["cy"]), "xk")

                if PLOT_ONREGION:
                    onregion = irf.reconstruction.onregion.estimate_onregion(
                        reco_cx=fit["primary_particle_cx"],
                        reco_cy=fit["primary_particle_cy"],
                        reco_main_axis_azimuth=fit["main_axis_azimuth"],
                        reco_num_photons=len(
                            loph_record["photons"]["arrival_time_slices"]
                        ),
                        reco_core_radius=np.hypot(
                            fit["primary_particle_x"],
                            fit["primary_particle_y"],
                        ),
                        config=onreion_config,
                    )

                    ellxy = irf.reconstruction.onregion.make_polygon(
                        onregion=onregion
                    )

                    hit = irf.reconstruction.onregion.is_direction_inside(
                        cx=truth["cx"], cy=truth["cy"], onregion=onregion
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
                info_str += "Energy: {: .1f}GeV, ".format(truth["energy_GeV"])
                info_str += "reco. Cherenkov: {: 4d}p.e.\n ".format(
                    loph_record["photons"]["channels"].shape[0]
                )
                info_str += "response of shower-model: {:.4f} ({:.4f})".format(
                    fit["shower_model_response"],
                    true_response,
                )

                ax.set_title(info_str)

                ax.set_xlim([-1.05 * fov_radius_deg, 1.05 * fov_radius_deg])
                ax.set_ylim([-1.05 * fov_radius_deg, 1.05 * fov_radius_deg])
                ax.set_aspect("equal")
                ax.set_xlabel("cx / deg")
                ax.set_ylabel("cy / deg")

                ax_core.plot(fit_x, fit_y, "oc")
                ax_core.plot([0, fit_x], [0, fit_y], "c", alpha=0.5)

                ax_core.plot(truth["x"], truth["y"], "xk")
                ax_core.plot([0, truth["x"]], [0, truth["y"]], "k", alpha=0.5)

                ax_core.set_xlim([-640, 640])
                ax_core.set_ylim([-640, 640])
                ax_core.set_aspect("equal")
                ax_core.set_xlabel("x / m")
                ax_core.set_ylabel("y / m")
                path = os.path.join(
                    pa["out_dir"],
                    sk,
                    pk,
                    "{:09d}.jpg".format(
                        airshower_id,
                    ),
                )

                fig.savefig(path)
                seb.close(fig)
