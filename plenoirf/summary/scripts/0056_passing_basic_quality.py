#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import json_utils
import warnings
import numpy as np
import sebastians_matplotlib_addons as sebplt


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)


def get_uid_passing_size_cut(features, min_reconstructed_photons):
    mask = features["num_photons"] >= min_reconstructed_photons
    return features["uid"][mask]


def get_uid_passing_relative_leakage_cut(features, max_relative_leakage):
    relative_leakage = (
        features["image_smallest_ellipse_num_photons_on_edge_field_of_view"]
        / features["num_photons"]
    )
    mask = relative_leakage <= max_relative_leakage
    return features["uid"][mask]


def get_uid_passing_aperture_flatness_cut(features, min_mean_over_std):
    mean_over_std = 1.0 / features["paxel_intensity_peakness_std_over_mean"]
    mask = mean_over_std >= min_mean_over_std
    return features["uid"][mask]


def _count_num_passing_cut_scan(features, cut_function, cut_values):
    num_steps = cut_values.shape[0]
    num_passing = np.nan * np.ones(shape=num_steps)
    for i in range(num_steps):
        _uids = cut_function(features, cut_values[i])
        num_passing[i] = _uids.shape[0]
    return num_passing


def analyse_cut_scan(features, cut_function, cut_values):
    out = {}
    out["count"] = _count_num_passing_cut_scan(
        features=features, cut_function=cut_function, cut_values=cut_values
    )
    out["count_au"] = np.sqrt(out["count"])
    num_total = features.shape[0]

    out["ratio"] = out["count"] / num_total
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in divide"
        )
        out["ratio_au"] = out["count_au"] / out["count"]
    return out


def lims(x):
    return min(x), max(x)


NUM_STEPS = 101

CUTS = {
    "size": {
        "key": "min_reconstructed_photons",
        "cut_function": get_uid_passing_size_cut,
        "cut_values": np.linspace(
            0.1 * res.analysis["quality"]["min_reconstructed_photons"],
            10.0 * res.analysis["quality"]["min_reconstructed_photons"],
            NUM_STEPS,
        ),
    },
    "relative_leakage": {
        "key": "max_relative_leakage",
        "cut_function": get_uid_passing_relative_leakage_cut,
        "cut_values": np.linspace(
            0.5 * res.analysis["quality"]["max_relative_leakage"],
            5.0 * res.analysis["quality"]["max_relative_leakage"],
            NUM_STEPS,
        ),
    },
    "aperture_flatness": {
        "key": "min_aperture_intensity_flatness_mean_over_std",
        "cut_function": get_uid_passing_aperture_flatness_cut,
        "cut_values": np.linspace(
            0.0,
            4.0,
            NUM_STEPS,
        ),
    },
}

for cut_key in CUTS:
    CUTS[cut_key]["cut_value"] = res.analysis["quality"][CUTS[cut_key]["key"]]


scans = {}

for pk in res.PARTICLES:
    scans[pk] = {}
    pk_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    with res.open_event_table(particle_key=pk) as arc:
        _event_table = arc.query(
            levels_and_columns={
                "features": [
                    "uid",
                    "num_photons",
                    "image_smallest_ellipse_num_photons_on_edge_field_of_view",
                    "paxel_intensity_peakness_std_over_mean",
                ]
            }
        )
        features = _event_table["features"]

    for cut_key in CUTS:
        scans[pk][cut_key] = analyse_cut_scan(
            features=features,
            cut_function=CUTS[cut_key]["cut_function"],
            cut_values=CUTS[cut_key]["cut_values"],
        )

    # Applying the cuts
    # -----------------
    uid_passed_size = get_uid_passing_size_cut(
        features=features,
        min_reconstructed_photons=res.analysis["quality"][
            "min_reconstructed_photons"
        ],
    )
    uid_passed_leakage = get_uid_passing_relative_leakage_cut(
        features=features,
        max_relative_leakage=res.analysis["quality"]["max_relative_leakage"],
    )
    uid_passed_aperture_flatness = get_uid_passing_aperture_flatness_cut(
        features=features,
        min_mean_over_std=res.analysis["quality"][
            "min_aperture_intensity_flatness_mean_over_std"
        ],
    )

    uid_passed = snt.logic.intersection(
        uid_passed_size,
        uid_passed_leakage,
        uid_passed_aperture_flatness,
    )

    json_utils.write(opj(pk_dir, "uid.json"), uid_passed)


for cut_key in CUTS:
    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in scans:
        ax.plot(
            CUTS[cut_key]["cut_values"],
            scans[pk][cut_key]["ratio"],
            color=res.PARTICLE_COLORS[pk],
        )
    ax.axvline(
        x=CUTS[cut_key]["cut_value"], color="black", alpha=0.5, linestyle="-."
    )
    ax.text(
        s=f" Cut = {CUTS[cut_key]['cut_value']:.3f}",
        x=CUTS[cut_key]["cut_value"],
        y=0.9,
    )
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim(lims(CUTS[cut_key]["cut_values"]))
    ax.set_xlabel(CUTS[cut_key]["key"])
    ax.set_ylabel("fraction passing / 1")
    fig.savefig(opj(res.paths["out_dir"], f"cut_{cut_key:s}.jpg"))
    sebplt.close(fig)

res.stop()
