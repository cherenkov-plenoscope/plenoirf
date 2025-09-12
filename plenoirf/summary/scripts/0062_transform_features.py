#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import rename_after_writing as rnw
import os
from os.path import join as opj
import pandas
import numpy as np
import sebastians_matplotlib_addons as sebplt
import json_utils
import warnings


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

os.makedirs(res.paths["out_dir"], exist_ok=True)

ORIGINAL_FEATURES = irf.event_table.structure.init_features_level_structure()
COMBINED_FEATURES = (
    irf.features.combined_features.init_combined_features_structure()
)
ALL_FEATURES = irf.features.init_all_features_structure()


def open_event_frame(pk):
    passing_trigger = json_utils.tree.read(
        opj(res.paths["analysis_dir"], "0055_passing_trigger")
    )
    passing_quality = json_utils.tree.read(
        opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
    )

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={
                "features": ["uid"],
                "reconstructed_trajectory": ["uid"],
            }
        )
        common_indices = snt.logic.intersection(
            passing_trigger[pk]["uid"],
            passing_quality[pk]["uid"],
            event_table["features"]["uid"],
            event_table["reconstructed_trajectory"]["uid"],
        )
        event_table = arc.query(
            levels_and_columns={
                "features": "__all__",
                "reconstructed_trajectory": "__all__",
            },
            indices=common_indices,
            sort=True,
        )
    return snt.logic.make_rectangular_DataFrame(event_table)


particle_colors = res.analysis["plot"]["particle_colors"]

ft_trafo = {}
for pk in ["gamma"]:
    event_frame = open_event_frame(pk=pk)

    for fk in ALL_FEATURES:
        if fk in ORIGINAL_FEATURES:
            f_raw = event_frame[f"features/{fk:s}"]
        else:
            f_raw = COMBINED_FEATURES[fk]["generator"](event_frame)

        ft_trafo[fk] = irf.features.find_transformation(
            feature_raw=f_raw,
            transformation_instruction=ALL_FEATURES[fk]["transformation"],
        )


transformed_features = {}
for pk in res.PARTICLES:
    transformed_features[pk] = {}

    event_frame = open_event_frame(pk=pk)

    transformed_features[pk]["uid"] = np.array(event_frame["uid"])

    for fk in ALL_FEATURES:
        if fk in ORIGINAL_FEATURES:
            f_raw = event_frame[f"features/{fk:s}"]
        else:
            f_raw = COMBINED_FEATURES[fk]["generator"](event_frame)

        transformed_features[pk][fk] = irf.features.transform(
            feature_raw=f_raw, transformation=ft_trafo[fk]
        )

    pk_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    out_level = snt.testing.dict_to_recarray(transformed_features[pk])
    out_table = snt.SparseNumericTable(index_key="uid")
    out_table["transformed_features"] = out_level
    with rnw.Path(opj(pk_dir, "transformed_features.zip")) as path:
        with snt.open(
            file=path, mode="w", dtypes_and_index_key_from=out_table
        ) as arc:
            arc.append_table(out_table)


start_stop = {}
for fk in ALL_FEATURES:
    start_stop[fk] = []
    for pk in res.PARTICLES:
        start = np.quantile(transformed_features[pk][fk], 0.01)
        stop = np.quantile(transformed_features[pk][fk], 0.99)
        if np.isnan(start) or np.isnan(stop):
            start = -1
            stop = 1
        start_stop[fk].append([start, stop])

for fk in ALL_FEATURES:
    start_stop[fk] = np.asarray(start_stop[fk])
    start_stop[fk] = [
        np.min(start_stop[fk][:, 0]),
        np.max(start_stop[fk][:, 1]),
    ]


NOT_VERY_USEFULL = [
    "image_num_islands",
    "aperture_num_islands_watershed_rel_thr_2",
    "aperture_num_islands_watershed_rel_thr_4",
    "aperture_num_islands_watershed_rel_thr_8",
    "image_infinity_num_photons_on_edge_field_of_view",
    "image_smallest_ellipse_num_photons_on_edge_field_of_view",
    "light_front_cx",
    "light_front_cy",
    "image_infinity_cx_mean",
    "image_infinity_cy_mean",
    "image_infinity_cx_std",
    "image_infinity_cy_std",
    "paxel_intensity_median_x",
    "paxel_intensity_median_y",
    "image_half_depth_shift_cx",
    "image_half_depth_shift_cy",
]


for fk in ALL_FEATURES:
    if fk in NOT_VERY_USEFULL:
        os.makedirs(
            opj(res.paths["out_dir"], "not_very_usefull"), exist_ok=True
        )
        fig_path = opj(res.paths["out_dir"], "not_very_usefull", f"{fk}.jpg")
    else:
        fig_path = opj(res.paths["out_dir"], f"{fk}.jpg")

    start, stop = start_stop[fk]

    if not os.path.exists(fig_path):
        fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
        ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

        for pk in res.PARTICLES:
            bin_edges_fk = np.linspace(start, stop, 101)
            bin_counts_fk = np.histogram(
                transformed_features[pk][fk], bins=bin_edges_fk
            )[0]

            bin_counts_unc_fk = irf.utils._divide_silent(
                numerator=np.sqrt(bin_counts_fk),
                denominator=bin_counts_fk,
                default=np.nan,
            )
            bin_counts_norm_fk = irf.utils._divide_silent(
                numerator=bin_counts_fk,
                denominator=(
                    np.ones(shape=bin_counts_fk.shape) * np.sum(bin_counts_fk)
                ),
                default=0,
            )

            bincounts_lower = bin_counts_norm_fk * (1 - bin_counts_unc_fk)
            bincounts_lower[bincounts_lower < 0] = 0

            sebplt.ax_add_histogram(
                ax=ax,
                bin_edges=bin_edges_fk,
                bincounts=bin_counts_norm_fk,
                linestyle="-",
                linecolor=particle_colors[pk],
                linealpha=1.0,
                bincounts_upper=bin_counts_norm_fk * (1 + bin_counts_unc_fk),
                bincounts_lower=bincounts_lower,
                face_color=particle_colors[pk],
                face_alpha=0.3,
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                category=UserWarning,
            )
            ax.semilogy()
        irf.summary.figure.mark_ax_thrown_spectrum(ax=ax)
        ax.set_xlabel("transformed {:s} / 1".format(fk))
        ax.set_ylabel("relative intensity / 1")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                category=UserWarning,
            )
            ax.set_xlim([start, stop])
        ax.set_ylim([1e-5, 1.0])
        fig.savefig(fig_path)
        sebplt.close(fig)

res.stop()
