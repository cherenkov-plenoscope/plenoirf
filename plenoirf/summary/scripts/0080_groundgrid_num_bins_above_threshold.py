#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import atmospheric_cherenkov_response
import sparse_numeric_table as snt
import os
from os.path import join as opj
import json_utils
import magnetic_deflection as mdfl
import binning_utils
import spherical_coordinates
import solid_angle_utils
import binning_utils
import sebastians_matplotlib_addons as sebplt
import confusion_matrix

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

energy_bin = res.energy_binning(key="10_bins_per_decade")
zenith_bin = res.zenith_binning("3_bins_per_45deg")
nat_bin = binning_utils.Binning(
    bin_edges=np.geomspace(10, 100_000, energy_bin["num"])
)
population_statistics = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0003_population_statistics")
)


def histogram(x, uid, bin_edges):
    assert len(x) == len(uid)

    _hist, _ = np.histogram(x, bins=bin_edges)
    _hist_bin_indices = np.digitize(x, bins=bin_edges)

    _hist_bin_uids = {}
    for i in range(len(x)):
        _hist_bin_index = int(_hist_bin_indices[i])
        _uid = uid[i]

        if _hist_bin_index in _hist_bin_uids:
            _hist_bin_uids[_hist_bin_index].append(_uid)
        else:
            _hist_bin_uids[_hist_bin_index] = [_uid]

    for _hist_bin_index in _hist_bin_uids:
        _hist_bin_uids[_hist_bin_index] = np.asarray(
            _hist_bin_uids[_hist_bin_index]
        )

    return _hist, _hist_bin_uids


def size_num_bin_edges(start_decade, stop_decade, num_steps_per_decade):
    num_decades = stop_decade - start_decade
    return num_decades * num_steps_per_decade + 1


# estimate range for count histogram

MAX_COUNTS = 0
for pk in res.PARTICLES:
    _max = np.max(
        population_statistics[pk]["num_thrown_energy_vs_zenith"]["counts"]
    )
    MAX_COUNTS = np.max([MAX_COUNTS, _max])

MAX_COUNTS = 10 ** np.ceil(np.log10(MAX_COUNTS))

bbb = {}
huh = {}
for pk in res.PARTICLES:
    bbb[pk] = {"avg": [], "au": []}
    huh[pk] = {}

    with res.open_event_table(particle_key=pk) as arc:
        table = arc.query(
            levels_and_columns={
                "primary": [
                    "uid",
                    "energy_GeV",
                ],
                "instrument_pointing": ["uid", "zenith_rad"],
                "groundgrid": ["uid", "num_bins_above_threshold"],
            }
        )

    table = snt.logic.sort_table_on_common_indices(
        table=table,
        common_indices=table["primary"]["uid"],
    )

    huh[pk]["bin_edges"] = np.geomspace(
        1e0,
        1e6,
        size_num_bin_edges(
            start_decade=0, stop_decade=6, num_steps_per_decade=10
        ),
    )

    huh[pk]["bin_counts"] = {}
    huh[pk]["bin_indices"] = {}

    min_number_samples = 10
    for zd in range(zenith_bin["num"]):
        zd_start = zenith_bin["edges"][zd]
        zd_stop = zenith_bin["edges"][zd + 1]

        mask = np.logical_and(
            table["instrument_pointing"]["zenith_rad"] >= zd_start,
            table["instrument_pointing"]["zenith_rad"] < zd_stop,
        )

        huh[pk]["bin_counts"][zd], huh[pk]["bin_indices"][zd] = histogram(
            x=table["groundgrid"]["num_bins_above_threshold"][mask],
            uid=table["groundgrid"]["uid"][mask],
            bin_edges=huh[pk]["bin_edges"],
        )

        # 1D VS energy
        # ------------
        hh_exposure = np.histogram(
            table["primary"]["energy_GeV"][mask],
            bins=energy_bin["edges"],
        )[0]
        hh_num_bins = np.histogram(
            table["primary"]["energy_GeV"][mask],
            bins=energy_bin["edges"],
            weights=table["groundgrid"]["num_bins_above_threshold"][mask],
        )[0]

        avg_num_bins = irf.utils._divide_silent(
            hh_num_bins, hh_exposure, default=float("nan")
        )
        avg_num_bins_ru = irf.utils._divide_silent(
            np.sqrt(hh_exposure), hh_exposure, default=float("nan")
        )
        avg_num_bins_au = avg_num_bins * avg_num_bins_ru

        bbb[pk]["avg"].append(avg_num_bins)
        bbb[pk]["au"].append(avg_num_bins_au)

        # 2D VS energy
        # ------------
        cm = confusion_matrix.init(
            ax0_key="primary/energy_GeV",
            ax0_values=table["primary"]["energy_GeV"][mask],
            ax0_bin_edges=energy_bin["edges"],
            ax1_key="groundgrid/num_bins_above_threshold",
            ax1_values=table["groundgrid"]["num_bins_above_threshold"][mask],
            ax1_bin_edges=nat_bin["edges"],
            weights=None,
            min_exposure_ax0=min_number_samples,
            default_low_exposure=0.0,
        )

        fig = sebplt.figure(irf.summary.figure.style(key="6:7")[0])
        ax_c = sebplt.add_axes(fig=fig, span=[0.2, 0.25, 0.75, 0.65])
        ax_h = sebplt.add_axes(fig=fig, span=[0.2, 0.13, 0.75, 0.10])
        ax_cb = sebplt.add_axes(fig=fig, span=[0.25, 0.96, 0.65, 0.015])
        sebplt.add_axes_zenith_range_indicator(
            fig=fig,
            span=[0.02, 0.02, 0.12, 0.12],
            zenith_bin_edges_rad=zenith_bin["edges"],
            zenith_bin=zd,
            fontsize=6,
        )
        _pcm_confusion = ax_c.pcolormesh(
            cm["ax0_bin_edges"],
            cm["ax1_bin_edges"],
            np.transpose(cm["counts_normalized_on_ax0"]),
            cmap=res.PARTICLE_COLORMAPS[pk],
            norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
        )
        sebplt.plt.colorbar(
            _pcm_confusion, cax=ax_cb, extend="max", orientation="horizontal"
        )
        ax_c.set_ylabel("num. bins above threshod / 1")
        ax_c.loglog()
        sebplt.ax_add_grid(ax_c)
        ax_c.set_xticklabels([])
        ax_h.loglog()
        ax_h.set_xlim(
            [np.min(cm["ax0_bin_edges"]), np.max(cm["ax0_bin_edges"])]
        )
        ax_h.set_ylim([1e1, MAX_COUNTS])
        ax_h.set_xlabel("energy / GeV")
        ax_h.set_ylabel("population")
        ax_h.axhline(min_number_samples, linestyle=":", color="k")
        sebplt.ax_add_histogram(
            ax=ax_h,
            bin_edges=cm["ax0_bin_edges"],
            bincounts=cm["exposure_ax0"],
            linestyle="-",
            linecolor=res.PARTICLE_COLORS[pk],
        )
        fig.savefig(opj(res.paths["out_dir"], f"{pk:s}_zd{zd:d}_grid.jpg"))
        sebplt.close(fig)


fstyle, axspan = irf.summary.figure.style(key="1:1")
fig = sebplt.figure(fstyle)
ax = sebplt.add_axes(fig=fig, span=axspan)
linestyles = ["-", "--", ":"]
for pk in res.PARTICLES:
    for zdbin in range(zenith_bin["num"]):
        ax.plot(
            energy_bin["centers"],
            bbb[pk]["avg"][zdbin],
            linestyle=linestyles[zdbin],
            color=res.PARTICLE_COLORS[pk],
        )
ax.set_ylim([1e0, 1e5])
ax.loglog()
ax.set_xlabel("energy / GeV")
ax.set_ylabel("num. bins above threshold / 1")
fig.savefig(opj(res.paths["out_dir"], "num_bins_above_threshold.jpg"))
sebplt.close(fig)


# what was thrown
# ---------------
_ylim = []
for zd in range(zenith_bin["num"]):
    for pk in res.PARTICLES:
        val = huh[pk]["bin_counts"][zd] / np.sum(huh[pk]["bin_counts"][zd])
        _ylim.append(irf.utils.find_decade_limits(x=val))
_ylim = np.asarray(_ylim)
ylim = (np.min(_ylim[:, 0]), np.max(_ylim[:, 1]))


for zd in range(zenith_bin["num"]):
    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    ax_zd = sebplt.add_axes_zenith_range_indicator(
        fig=fig,
        zenith_bin_edges_rad=zenith_bin["edges"],
        zenith_bin=zd,
        span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
        fontsize=5,
    )
    for pk in res.PARTICLES:
        sebplt.ax_add_histogram(
            ax=ax,
            bin_edges=huh[pk]["bin_edges"],
            bincounts=huh[pk]["bin_counts"][zd]
            / np.sum(huh[pk]["bin_counts"][zd]),
            linestyle="-",
            linecolor=res.PARTICLE_COLORS[pk],
            linealpha=1.0,
            bincounts_upper=None,
            bincounts_lower=None,
            face_color=res.PARTICLE_COLORS[pk],
            face_alpha=0.3,
        )
    ax.loglog()
    ax.set_ylim(ylim)
    ax.set_xlabel("num. bins above threshold / 1")
    ax.set_ylabel("relative intensity / 1")
    fig.savefig(opj(res.paths["out_dir"], f"what_was_throwm_zd{zd:d}.jpg"))
    sebplt.close(fig)

res.stop()
