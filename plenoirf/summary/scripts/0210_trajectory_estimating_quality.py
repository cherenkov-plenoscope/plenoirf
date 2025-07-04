#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils
import sparse_numeric_table as snt


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)


passing_trigger = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)
passing_trajectory = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0059_passing_trajectory_quality")
)
weights_thrown2expected = json_utils.tree.read(
    opj(
        res.paths["analysis_dir"],
        "0040_weights_from_thrown_to_expected_energy_spectrum",
    )
)
min_trajectory_quality = res.analysis["quality"]["min_trajectory_quality"]

theta_bin_edges_deg = np.linspace(0.0, 3.0, 15)

# feature correlation
# ===================
feature_correlations = [
    {
        "key": "reconstructed_trajectory/r_m",
        "label": "reco. core-radius / m",
        "bin_edges": np.linspace(0.0, 640, 17),
        "log": False,
    },
    {
        "key": "features/image_smallest_ellipse_object_distance",
        "label": "object-distance / m",
        "bin_edges": np.geomspace(5e3, 50e3, 17),
        "log": True,
    },
    {
        "key": "features/image_smallest_ellipse_solid_angle",
        "label": "smallest ellipse solid angle / sr",
        "bin_edges": np.geomspace(1e-7, 1e-3, 17),
        "log": True,
    },
    {
        "key": "features/num_photons",
        "label": "reco. num. photons / p.e.",
        "bin_edges": np.geomspace(1e1, 1e5, 17),
        "log": True,
    },
    {
        "key": "features/image_num_islands",
        "label": "num. islands / 1",
        "bin_edges": np.arange(1, 7),
        "log": False,
    },
    {
        "key": "features/image_half_depth_shift_c",
        "label": "image_half_depth_shift / rad",
        "bin_edges": np.deg2rad(np.linspace(0.0, 0.2, 17)),
        "log": False,
    },
]


def write_correlation_figure(
    path,
    x,
    y,
    x_bin_edges,
    y_bin_edges,
    x_label,
    y_label,
    min_exposure_x,
    logx=False,
    logy=False,
    log_exposure_counter=False,
    x_cut=None,
):
    valid = np.logical_and(
        np.logical_not((np.isnan(x))), np.logical_not((np.isnan(y)))
    )

    cm = confusion_matrix.init(
        ax0_key="x",
        ax0_values=x[valid],
        ax0_bin_edges=x_bin_edges,
        ax1_key="y",
        ax1_values=y[valid],
        ax1_bin_edges=y_bin_edges,
        min_exposure_ax0=min_exposure_x,
        default_low_exposure=0.0,
    )

    fig = sebplt.figure(sebplt.FIGURE_1_1)
    ax = sebplt.add_axes(fig=fig, span=[0.25, 0.27, 0.55, 0.65])
    ax_h = sebplt.add_axes(fig=fig, span=[0.25, 0.11, 0.55, 0.1])
    ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
    _pcm_confusion = ax.pcolormesh(
        cm["ax0_bin_edges"],
        cm["ax1_bin_edges"],
        np.transpose(cm["counts_normalized_on_ax0"]),
        cmap="Greys",
        norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
    )
    sebplt.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
    ax.set_title("normalized for each column")
    ax.set_ylabel(y_label)
    ax.set_xticklabels([])
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    ax_h.set_xlim([np.min(cm["ax0_bin_edges"]), np.max(cm["ax0_bin_edges"])])
    ax_h.set_xlabel(x_label)
    ax_h.set_ylabel("num. events / 1")
    ax_h.axhline(cm["min_exposure_ax0"], linestyle=":", color="k")
    sebplt.ax_add_histogram(
        ax=ax_h,
        bin_edges=cm["ax0_bin_edges"],
        bincounts=cm["exposure_ax0"],
        linestyle="-",
        linecolor="k",
    )

    if x_cut is not None:
        for aa in [ax, ax_h]:
            aa.plot([x_cut, x_cut], aa.get_ylim(), "k:")

    if logx:
        ax.semilogx()
        ax_h.semilogx()

    if logy:
        ax.semilogy()

    if log_exposure_counter:
        ax_h.semilogy()

    fig.savefig(path)
    sebplt.close(fig)


def align_values_with_event_frame(event_frame, uids, values):
    Q = {}
    for ii in range(len(uids)):
        Q[uids[ii]] = values[ii]

    aligned_values = np.nan * np.ones(event_frame.shape[0])
    for ii in range(event_frame.shape[0]):
        aligned_values[ii] = Q[event_frame["uid"][ii]]
    return aligned_values


the = "theta"

QP = {}
QP["quality_cuts"] = np.linspace(0.0, 1.0, 137)
QP["fraction_passing"] = {}
QP["fraction_passing_w"] = {}

for pk in res.PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={
                "primary": "__all__",
                "instrument_pointing": "__all__",
                "groundgrid_choice": "__all__",
                "reconstructed_trajectory": "__all__",
                "features": "__all__",
            }
        )

    uid_common = snt.logic.intersection(
        [
            passing_trigger[pk]["uid"],
            passing_quality[pk]["uid"],
            passing_trajectory[pk]["trajectory_quality"]["uid"],
        ]
    )
    event_table = snt.logic.cut_and_sort_table_on_indices(
        table=event_table,
        common_indices=uid_common,
        level_keys=[
            "primary",
            "instrument_pointing",
            "groundgrid_choice",
            "reconstructed_trajectory",
            "features",
        ],
    )

    event_frame = irf.reconstruction.trajectory_quality.make_rectangular_table(
        event_table=event_table,
        instrument_pointing_model=res.config["pointing"]["model"],
    )

    quality = align_values_with_event_frame(
        event_frame=event_frame,
        uids=passing_trajectory[pk]["trajectory_quality"]["uid"],
        values=passing_trajectory[pk]["trajectory_quality"]["quality"],
    )

    write_correlation_figure(
        path=opj(
            res.paths["out_dir"],
            "{:s}_{:s}_vs_quality.jpg".format(pk, the),
        ),
        x=quality,
        y=np.rad2deg(event_frame["reconstructed_trajectory/" + the + "_rad"]),
        x_bin_edges=np.linspace(0, 1, 15),
        y_bin_edges=theta_bin_edges_deg,
        x_label="quality / 1",
        y_label=the + r" / $1^{\circ}$",
        min_exposure_x=100,
        logx=False,
        logy=False,
        log_exposure_counter=False,
        x_cut=min_trajectory_quality,
    )

    if pk == "gamma":
        write_correlation_figure(
            path=opj(
                res.paths["out_dir"],
                "{:s}_energy_vs_quality.jpg".format(pk, the),
            ),
            x=quality,
            y=event_frame["primary/energy_GeV"],
            x_bin_edges=np.linspace(0, 1, 15),
            y_bin_edges=np.geomspace(1, 1000, 15),
            x_label="quality / 1",
            y_label="energy / GeV",
            min_exposure_x=100,
            logx=False,
            logy=True,
            log_exposure_counter=False,
            x_cut=min_trajectory_quality,
        )

    if pk == "gamma":
        for fk in feature_correlations:
            write_correlation_figure(
                path=opj(
                    res.paths["out_dir"],
                    "{:s}_{:s}_vs_{:s}.jpg".format(
                        pk, the, str.replace(fk["key"], "/", "-")
                    ),
                ),
                x=event_frame[fk["key"]],
                y=np.rad2deg(
                    event_frame["reconstructed_trajectory/" + the + "_rad"]
                ),
                x_bin_edges=fk["bin_edges"],
                y_bin_edges=theta_bin_edges_deg,
                x_label=fk["label"],
                y_label=the + r" / $1^{\circ}$",
                min_exposure_x=100,
                logx=fk["log"],
                logy=False,
                log_exposure_counter=False,
            )

    # plot losses
    # ===========

    reweight_spectrum = np.interp(
        x=event_frame["primary/energy_GeV"],
        xp=weights_thrown2expected[pk]["weights_vs_energy"]["energy_GeV"],
        fp=weights_thrown2expected[pk]["weights_vs_energy"]["mean"],
    )

    fraction_passing = []
    fraction_passing_w = []
    for quality_cut in QP["quality_cuts"]:
        mask = quality >= quality_cut
        num_passing_cut = np.sum(mask)
        num_total = quality.shape[0]
        fraction_passing.append(num_passing_cut / num_total)

        num_passing_cut_w = np.sum(reweight_spectrum[mask])
        num_total_w = np.sum(reweight_spectrum)
        fraction_passing_w.append(num_passing_cut_w / num_total_w)

    QP["fraction_passing"][pk] = np.array(fraction_passing)
    QP["fraction_passing_w"][pk] = np.array(fraction_passing_w)


fig = sebplt.figure(sebplt.FIGURE_1_1)
ax = sebplt.add_axes(fig=fig, span=[0.16, 0.11, 0.8, 0.8])
for pk in PARTICLES:
    ax.plot(
        QP["quality_cuts"],
        QP["fraction_passing_w"][pk],
        color=sum_config["plot"]["particle_colors"][pk],
    )
ax.plot([min_trajectory_quality, min_trajectory_quality], [0.0, 1.0], "k:")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel("trajectory-quality-cut / 1")
ax.set_ylabel("passing cut / 1")
fig.savefig(opj(res.paths["out_dir"], "passing.jpg"))
sebplt.close(fig)

res.stop()
