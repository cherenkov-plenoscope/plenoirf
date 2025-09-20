#!/usr/bin/python
import sys
import numpy as np
import os
from os.path import join as opj
import plenoirf as irf
import sparse_numeric_table as snt
import sebastians_matplotlib_addons as sebplt
import json_utils


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

passing_trigger = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)

energy_bin = res.energy_binning(key="point_spread_function")

span_hist_1_1 = [0.2, 0.15, 0.75, 0.8]


def guess_num_bins(num_events):
    num_bins = int(0.2 * np.sqrt(num_events))
    return np.max([num_bins, 3])


CHCL = "cherenkovclassification"

for pk in res.PARTICLES:
    pk_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={
                "primary": ["uid", "energy_GeV"],
                "trigger": ["uid", "num_cherenkov_pe"],
                CHCL: "__all__",
                "features": ["uid", "num_photons"],
            }
        )

    uid_common = snt.logic.intersection(
        passing_trigger[pk]["uid"],
        passing_quality[pk]["uid"],
    )

    mrg_chc_fts = snt.logic.cut_and_sort_table_on_indices(
        table=event_table,
        common_indices=uid_common,
    )

    # ---------------------------------------------------------------------
    key = "confusion"
    num_bins_size_confusion_matrix = guess_num_bins(
        num_events=mrg_chc_fts["features"].shape[0]
    )
    size_bin_edges = np.geomspace(1e1, 1e5, num_bins_size_confusion_matrix + 1)
    np_bins = np.histogram2d(
        mrg_chc_fts["trigger"]["num_cherenkov_pe"],
        mrg_chc_fts["features"]["num_photons"],
        bins=[size_bin_edges, size_bin_edges],
    )[0]
    np_exposure_bins = np.histogram(
        mrg_chc_fts["trigger"]["num_cherenkov_pe"], bins=size_bin_edges
    )[0]

    np_bins_normalized = np_bins.copy()
    for true_bin in range(num_bins_size_confusion_matrix):
        if np_exposure_bins[true_bin] > 0:
            np_bins_normalized[true_bin, :] /= np_exposure_bins[true_bin]

    fig = sebplt.figure(style=sebplt.FIGURE_1_1)
    ax = sebplt.add_axes(fig=fig, span=[0.15, 0.27, 0.65, 0.65])
    ax_h = sebplt.add_axes(fig=fig, span=[0.15, 0.11, 0.65, 0.1])
    ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
    _pcm_confusion = ax.pcolormesh(
        size_bin_edges,
        size_bin_edges,
        np.transpose(np_bins_normalized),
        cmap=res.PARTICLE_COLORMAPS[pk],
        norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
    )
    sebplt.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
    ax.set_aspect("equal")
    # ax.set_title("normalized for each column")
    ax.set_ylabel("reco. Cherenkov size / p.e.")
    ax.loglog()
    ax.set_xticklabels([])
    sebplt.ax_add_grid(ax)
    sebplt.ax_add_histogram(
        ax=ax_h,
        bin_edges=size_bin_edges,
        bincounts=np_exposure_bins,
        linestyle="-",
        linecolor=res.PARTICLE_COLORS[pk],
    )
    ax_h.semilogx()
    ax_h.set_xlim([np.min(size_bin_edges), np.max(size_bin_edges)])
    ax_h.set_xlabel("true Cherenkov size / p.e.")
    ax_h.set_ylabel("num. events")
    fig.savefig(opj(res.paths["out_dir"], pk + "_" + key + ".jpg"))
    sebplt.close(fig)

    # ---------------------------------------------------------------------
    key = "sensitivity_vs_true_energy"
    tprs = []
    ppvs = []
    num_events = []
    for i in range(energy_bin["num"]):
        e_start = energy_bin["edges"][i]
        e_stop = energy_bin["edges"][i + 1]
        e_mask = np.logical_and(
            mrg_chc_fts["primary"]["energy_GeV"] >= e_start,
            mrg_chc_fts["primary"]["energy_GeV"] < e_stop,
        )
        num_matches = np.sum(e_mask)
        num_events.append(num_matches)

        if num_matches == 0:
            tprs.append(np.nan)
            ppvs.append(np.nan)
        else:
            tp = mrg_chc_fts[CHCL]["num_true_positives"][e_mask]
            fn = mrg_chc_fts[CHCL]["num_false_negatives"][e_mask]
            fp = mrg_chc_fts[CHCL]["num_false_positives"][e_mask]
            tpr = tp / (tp + fn)
            ppv = tp / (tp + fp)
            tprs.append(np.median(tpr))
            ppvs.append(np.median(ppv))

    tprs = np.array(tprs)
    ppvs = np.array(ppvs)
    num_events = np.array(num_events)
    num_events_relunc = np.nan * np.ones(num_events.shape[0])
    _v = num_events > 0
    num_events_relunc[_v] = np.sqrt(num_events[_v]) / num_events[_v]

    fstyle, axspan = irf.summary.figure.style("16:5")

    fig = sebplt.figure(fstyle)
    ax = sebplt.add_axes(fig=fig, span=axspan)
    sebplt.ax_add_histogram(
        ax=ax,
        bin_edges=energy_bin["edges"],
        bincounts=tprs,
        linestyle="-",
        linecolor=res.PARTICLE_COLORS[pk],
        bincounts_upper=tprs * (1 + num_events_relunc),
        bincounts_lower=tprs * (1 - num_events_relunc),
        face_color=res.PARTICLE_COLORS[pk],
        face_alpha=0.05,
        label="true positive rate",
    )
    sebplt.ax_add_histogram(
        ax=ax,
        bin_edges=energy_bin["edges"],
        bincounts=ppvs,
        linestyle=":",
        linecolor=res.PARTICLE_COLORS[pk],
        bincounts_upper=ppvs * (1 + num_events_relunc),
        bincounts_lower=ppvs * (1 - num_events_relunc),
        face_color=res.PARTICLE_COLORS[pk],
        face_alpha=0.05,
        label="positive predictive value",
    )
    ax.legend()
    ax.set_xlabel("energy / GeV")
    ax.set_xlim(energy_bin["limits"])
    ax.set_ylim([0, 1])
    ax.semilogx()
    fig.savefig(opj(res.paths["out_dir"], pk + "_" + key + ".jpg"))
    sebplt.close(fig)

    json_utils.write(
        opj(pk_dir, key + ".json"),
        {
            "energy_bin_edges_GeV": energy_bin["edges"],
            "num_events": num_events,
            "true_positive_rate": tprs,
            "positive_predictive_value": ppvs,
        },
    )

    # ---------------------------------------------------------------------
    key = "true_size_over_extracted_size_vs_true_energy"
    true_over_reco_ratios = []
    num_events = []
    for i in range(energy_bin["num"]):
        e_start = energy_bin["edges"][i]
        e_stop = energy_bin["edges"][i + 1]
        e_mask = np.logical_and(
            mrg_chc_fts["primary"]["energy_GeV"] >= e_start,
            mrg_chc_fts["primary"]["energy_GeV"] < e_stop,
        )
        num_matches = np.sum(e_mask)
        num_events.append(num_matches)

        if num_matches == 0:
            true_over_reco_ratios.append(np.nan)
        else:
            true_num_cherenkov_pe = mrg_chc_fts["trigger"]["num_cherenkov_pe"][
                e_mask
            ]
            num_cherenkov_pe = mrg_chc_fts["features"]["num_photons"][e_mask]
            true_over_reco_ratio = true_num_cherenkov_pe / num_cherenkov_pe
            true_over_reco_ratios.append(np.median(true_over_reco_ratio))

    true_over_reco_ratios = np.array(true_over_reco_ratios)
    num_events = np.array(num_events)
    num_events_relunc = np.nan * np.ones(num_events.shape[0])
    _v = num_events > 0
    num_events_relunc[_v] = np.sqrt(num_events[_v]) / num_events[_v]

    fig = sebplt.figure(fstyle)
    ax = sebplt.add_axes(fig=fig, span=axspan)
    sebplt.ax_add_histogram(
        ax=ax,
        bin_edges=energy_bin["edges"],
        bincounts=true_over_reco_ratios,
        linestyle="-",
        linecolor=res.PARTICLE_COLORS[pk],
        bincounts_upper=true_over_reco_ratios * (1 + num_events_relunc),
        bincounts_lower=true_over_reco_ratios * (1 - num_events_relunc),
        face_color=res.PARTICLE_COLORS[pk],
        face_alpha=0.1,
    )
    ax.axhline(y=1, color="black", linestyle=":")
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel("Cherenkov size,\ntrue/extracted / 1")
    ax.set_xlim(energy_bin["limits"])
    ax.semilogx()
    fig.savefig(opj(res.paths["out_dir"], pk + "_" + key + ".jpg"))
    sebplt.close(fig)

    json_utils.write(
        opj(pk_dir, key + ".json"),
        {
            "energy_bin_edges_GeV": energy_bin["edges"],
            "num_events": num_events,
            "true_over_reco_ratios": true_over_reco_ratios,
        },
    )

    # ---------------------------------------------------------------------
    key = "true_size_over_extracted_size_vs_true_size"
    num_ratios = []
    num_events = []
    for i in range(num_bins_size_confusion_matrix):
        pe_start = size_bin_edges[i]
        pe_stop = size_bin_edges[i + 1]
        pe_mask = np.logical_and(
            mrg_chc_fts["trigger"]["num_cherenkov_pe"] >= pe_start,
            mrg_chc_fts["trigger"]["num_cherenkov_pe"] < pe_stop,
        )
        num_matches = np.sum(pe_mask)
        num_events.append(num_matches)

        if num_matches == 0:
            num_ratios.append(np.nan)
        else:
            true_num_cherenkov_pe = mrg_chc_fts["trigger"]["num_cherenkov_pe"][
                pe_mask
            ]
            num_cherenkov_pe = mrg_chc_fts["features"]["num_photons"][pe_mask]
            num_ratio = true_num_cherenkov_pe / num_cherenkov_pe
            num_ratios.append(np.median(num_ratio))

    num_ratios = np.array(num_ratios)
    num_events = np.array(num_events)
    num_events_relunc = np.nan * np.ones(num_events.shape[0])
    _v = num_events > 0
    num_events_relunc[_v] = np.sqrt(num_events[_v]) / num_events[_v]

    fig = sebplt.figure(fstyle)
    ax = sebplt.add_axes(fig=fig, span=axspan)
    sebplt.ax_add_histogram(
        ax=ax,
        bin_edges=size_bin_edges,
        bincounts=num_ratios,
        linestyle="-",
        linecolor=res.PARTICLE_COLORS[pk],
        bincounts_upper=num_ratios * (1 + num_events_relunc),
        bincounts_lower=num_ratios * (1 - num_events_relunc),
        face_color=res.PARTICLE_COLORS[pk],
        face_alpha=0.1,
    )
    ax.axhline(y=1, color="k", linestyle=":")
    ax.set_xlabel("true Cherenkov size / p.e.")
    ax.set_ylabel("true size /\nextracted size")
    ax.set_xlim([np.min(size_bin_edges), np.max(size_bin_edges)])
    ax.semilogx()
    fig.savefig(opj(res.paths["out_dir"], pk + "_" + key + ".jpg"))
    sebplt.close(fig)

    json_utils.write(
        opj(pk_dir, key + ".json"),
        {
            "size_bin_edges_pe": size_bin_edges,
            "num_events": num_events,
            "true_over_reco_ratios": num_ratios,
        },
    )

res.stop()
