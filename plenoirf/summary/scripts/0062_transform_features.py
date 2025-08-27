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

particle_colors = res.analysis["plot"]["particle_colors"]

ft_trafo = {}
for pk in ["gamma"]:
    with res.open_event_table(particle_key=pk) as arc:
        _table = arc.query(levels_and_columns={"features": "__all__"})

    features = _table["features"]

    for fk in ALL_FEATURES:
        if fk in ORIGINAL_FEATURES:
            f_raw = features[fk]
        else:
            f_raw = COMBINED_FEATURES[fk]["generator"](features)

        ft_trafo[fk] = irf.features.find_transformation(
            feature_raw=f_raw,
            transformation_instruction=ALL_FEATURES[fk]["transformation"],
        )


transformed_features = {}
for pk in res.PARTICLES:
    transformed_features[pk] = {}

    with res.open_event_table(particle_key=pk) as arc:
        _table = arc.query(levels_and_columns={"features": "__all__"})
        features = _table["features"]

    transformed_features[pk]["uid"] = np.array(features["uid"])

    for fk in ALL_FEATURES:
        if fk in ORIGINAL_FEATURES:
            f_raw = features[fk]
        else:
            f_raw = COMBINED_FEATURES[fk]["generator"](features)

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


for fk in ALL_FEATURES:
    fig_path = opj(res.paths["out_dir"], f"{fk}.jpg")

    if not os.path.exists(fig_path):
        fig = sebplt.figure(sebplt.FIGURE_16_9)
        ax = sebplt.add_axes(fig=fig, span=(0.1, 0.1, 0.8, 0.8))

        for pk in res.PARTICLES:
            start = -5
            stop = 5

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
