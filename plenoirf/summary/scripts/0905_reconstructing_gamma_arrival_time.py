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


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

passing_trigger = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)
passing_trajectory_quality = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0059_passing_trajectory_quality")
)
zenith_assignment = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0019_zenith_bin_assignment")
)
energy_bin = res.energy_binning(key="point_spread_function")
zenith_bin = res.zenith_binning("once")

SPEED_OF_LIGHT_IN_VACUUM = 299792458
SPEED_OF_GAMMA_RAY_IN_ATMOSPHERE = SPEED_OF_LIGHT_IN_VACUUM  # good enough
SPEED_OF_CHERENKOVLIGHT_IN_ATMOSPHERE = SPEED_OF_LIGHT_IN_VACUUM / 1.0003

TIME_BIN_EDGES = np.linspace(-50e-9, 50e-9, 26)


def normed(v):
    return v / np.linalg.norm(v)


tds = {}
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    tds[zk] = {}
    for pk in ["gamma"]:
        tds[zk][pk] = {}

        with res.open_event_table(particle_key=pk) as arc:
            event_table = arc.query(
                levels_and_columns={
                    "primary": "__all__",
                    "groundgrid_choice": "__all__",
                    "groundgrid": "__all__",
                    "instrument": "__all__",
                    "features": "__all__",
                    "reconstructed_trajectory": "__all__",
                }
            )

        uid_common = snt.logic.intersection(
            zenith_assignment[zk][pk],
            passing_trigger[pk]["uid"],
            passing_quality[pk]["uid"],
            passing_trajectory_quality[pk]["uid"],
        )

        event_table = snt.logic.cut_table_on_indices(
            table=event_table,
            common_indices=uid_common,
        )

        et = snt.logic.make_rectangular_DataFrame(event_table).to_records()

        tds[zk][pk]["true_wrt_corsika_wrt_collection_plane_s"] = []
        tds[zk][pk]["reco_simple_wrt_corsika_wrt_collection_plane_s"] = []
        tds[zk][pk]["reco_better_wrt_corsika_wrt_collection_plane_s"] = []
        tds[zk][pk]["tDelta_simple"] = []
        tds[zk][pk]["tDelta_better"] = []
        tds[zk][pk]["energy_GeV"] = []

        for i in range(len(et)):
            """
            true_core_r_m = np.hypot(et["core/core_x_m"][i], et["core/core_y_m"][i])
            if true_core_r_m > 100:
                continue
            """

            tds[zk][pk]["energy_GeV"].append(et["primary/energy_GeV"][i])
            # true arrival-time
            # -----------------
            particle_starting_direction = np.array(
                [
                    et["primary/momentum_x_GeV_per_c"][i],
                    et["primary/momentum_y_GeV_per_c"][i],
                    et["primary/momentum_z_GeV_per_c"][i],
                ]
            )
            particle_starting_direction = normed(particle_starting_direction)

            particle_starting_position = np.array(
                [
                    et["primary/starting_x_m"][i]
                    - et["groundgrid_choice/core_x_m"][i],
                    et["primary/starting_y_m"][i]
                    - et["groundgrid_choice/core_y_m"][i],
                    et["primary/starting_height_asl_m"][i],
                ]
            )

            instrument_position = np.array(
                [
                    0,
                    0,
                    res.SITE["observation_level_asl_m"],
                ]
            )

            distance_to_collection_plane = (
                irf.utils.ray_parameter_for_closest_distance_to_point(
                    ray_support=particle_starting_position,
                    ray_direction=particle_starting_direction,
                    point=instrument_position,
                )
            )

            _gamma_core = irf.utils.ray_at(
                ray_support=particle_starting_position,
                ray_direction=particle_starting_direction,
                parameter=distance_to_collection_plane,
            )

            t_true = (
                distance_to_collection_plane / SPEED_OF_GAMMA_RAY_IN_ATMOSPHERE
            )
            tds[zk][pk]["true_wrt_corsika_wrt_collection_plane_s"].append(
                t_true
            )

            # reconstructed arrival-time
            # --------------------------

            # simple: assume time of detection is the gamma-ray's arriving time
            # -----------------------------------------------------------------
            t_simple = et["instrument/start_time_of_exposure_s"][i]
            tds[zk][pk][
                "reco_simple_wrt_corsika_wrt_collection_plane_s"
            ].append(t_simple)
            tds[zk][pk]["tDelta_simple"].append(t_simple - t_true)

            # better:
            # -------
            gmax = et["features/image_smallest_ellipse_object_distance"][i]

            tCorr = gmax * (
                1 / SPEED_OF_LIGHT_IN_VACUUM
                - 1 / SPEED_OF_CHERENKOVLIGHT_IN_ATMOSPHERE
            )
            t_better = et["instrument/start_time_of_exposure_s"][i] + tCorr
            tds[zk][pk][
                "reco_better_wrt_corsika_wrt_collection_plane_s"
            ].append(t_better)
            tds[zk][pk]["tDelta_better"].append(t_better - t_true)

        tDelta = tds[zk][pk]["tDelta_better"] - np.median(
            tds[zk][pk]["tDelta_better"]
        )
        tDelta_rel = tDelta - np.median(tDelta)

        E_MAX = 5.0
        e_mask = np.array(tds[zk][pk]["energy_GeV"]) <= E_MAX
        tDelta_10 = tDelta[e_mask]
        tDelta_rel_10 = tDelta_10 - np.median(tDelta_10)

        Y_LIM = [-0.05, 0.5]
        ALPHA_ALL_ENERGY = 0.33
        # plot all energies
        fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
        ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
        sebplt.add_axes_zenith_range_indicator(
            fig=fig,
            span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
            zenith_bin_edges_rad=zenith_bin["edges"],
            zenith_bin=zd,
            fontsize=6,
        )
        bincounts = np.histogram(a=tDelta_rel, bins=TIME_BIN_EDGES)[0]
        numall = np.sum(bincounts)
        bincounts = bincounts / numall
        sebplt.ax_add_histogram(
            ax=ax,
            bin_edges=TIME_BIN_EDGES * 1e9,
            bincounts=bincounts,
            linestyle="-",
            linecolor="k",
            linealpha=ALPHA_ALL_ENERGY,
            bincounts_upper=None,
            bincounts_lower=None,
            face_color="black",
            face_alpha=None,
            label=None,
            draw_bin_walls=True,
        )
        ax.text(
            x=0.1,
            y=0.8,
            s="std. dev.: {:0.1f}ns".format(1e9 * np.std(tDelta_rel)),
            transform=ax.transAxes,
            alpha=ALPHA_ALL_ENERGY,
        )
        bincounts10 = np.histogram(a=tDelta_rel_10, bins=TIME_BIN_EDGES)[0]
        num10 = np.sum(bincounts10)
        bincounts10 = bincounts10 / num10
        sebplt.ax_add_histogram(
            ax=ax,
            bin_edges=TIME_BIN_EDGES * 1e9,
            bincounts=bincounts10,
            linestyle="-",
            linecolor="k",
            linealpha=1,
            bincounts_upper=None,
            bincounts_lower=None,
            face_color="black",
            face_alpha=None,
            label=None,
            draw_bin_walls=True,
        )
        ax.text(
            x=0.1,
            y=0.9,
            s="std. dev.: {:0.1f}ns".format(1e9 * np.std(tDelta_rel_10)),
            transform=ax.transAxes,
        )

        ax.text(
            x=0.6,
            y=0.9,
            s="below {:.1f}GeV ({:,d} events)".format(E_MAX, num10),
            transform=ax.transAxes,
        )
        ax.text(
            x=0.6,
            y=0.8,
            s="all energies ({:,d} events)".format(numall),
            transform=ax.transAxes,
            alpha=ALPHA_ALL_ENERGY,
        )
        ax.set_ylim(Y_LIM)
        ax.set_xlabel("(reconstructed - true) time of arrival / ns")
        ax.set_ylabel("intensity")
        fig.savefig(
            opj(
                res.paths["out_dir"],
                f"{zk:s}_{pk:s}_arrival_time_spread.jpg",
            )
        )
        sebplt.close(fig)
