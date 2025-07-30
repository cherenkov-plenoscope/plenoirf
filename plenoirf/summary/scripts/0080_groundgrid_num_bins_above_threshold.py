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

energy_bin = res.energy_binning(key="trigger_acceptance")
zenith_bin = res.zenith_binning("once")
nat_bin = binning_utils.Binning(
    bin_edges=np.geomspace(1, 100_000, energy_bin["num"])
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
                    "azimuth_rad",
                    "zenith_rad",
                ],
                "instrument_pointing": "__all__",
                "groundgrid": "__all__",
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

        fig = sebplt.figure(style=sebplt.FIGURE_1_1)
        ax_c = sebplt.add_axes(fig=fig, span=[0.25, 0.27, 0.55, 0.65])
        ax_h = sebplt.add_axes(fig=fig, span=[0.25, 0.11, 0.55, 0.1])
        ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
        ax_zd = sebplt.add_axes_zenith_range_indicator(
            fig=fig,
            span=[0.85, 0.11, 0.1, 0.1],
            zenith_bin_edges_rad=zenith_bin["edges"],
            zenith_bin=zd,
            fontsize=5,
        )

        _pcm_confusion = ax_c.pcolormesh(
            cm["ax0_bin_edges"],
            cm["ax1_bin_edges"],
            np.transpose(cm["counts_normalized_on_ax0"]),
            cmap=res.PARTICLE_COLORMAPS[pk],
            norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
        )
        sebplt.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")

        ax_c.set_ylabel("num. bins above threshod / 1")
        ax_c.loglog()
        res.ax_add_site_marker(ax_c, x=0.2, y=0.1)
        sebplt.ax_add_grid(ax_c)
        ax_c.text(
            x=0.2,
            y=0.05,
            s="normalized for each column",
            fontsize=8,
            # horizontalalignment="center",
            verticalalignment="center",
            transform=ax_c.transAxes,
        )

        ax_h.semilogx()
        ax_h.set_xlim(
            [np.min(cm["ax0_bin_edges"]), np.max(cm["ax0_bin_edges"])]
        )
        ax_h.set_xlabel("energy / GeV")
        ax_h.set_ylabel("num. events / 1")
        ax_h.axhline(min_number_samples, linestyle=":", color="k")
        sebplt.ax_add_histogram(
            ax=ax_h,
            bin_edges=cm["ax0_bin_edges"],
            bincounts=cm["exposure_ax0"],
            linestyle="-",
            linecolor="k",
        )
        fig.savefig(opj(res.paths["out_dir"], f"{pk:s}_zd{zd:d}_grid.jpg"))
        sebplt.close(fig)


fig = sebplt.figure(style=irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=[0.175, 0.15, 0.6, 0.8])
ax_legend = sebplt.add_axes(
    fig=fig, span=[0.8, 0.15, 0.1, 0.8], style=sebplt.AXES_BLANK
)
linestyles = ["-", "--", ":"]
for pk in res.PARTICLES:
    for zdbin in range(zenith_bin["num"]):
        ax.plot(
            energy_bin["centers"],
            bbb[pk]["avg"][zdbin],
            linestyle=linestyles[zdbin],
            color=res.PARTICLE_COLORS[pk],
        )
_yy = 1
for pk in res.PARTICLES:
    _yy -= 0.1
    ax_legend.plot(
        [0, 0.3],
        [_yy, _yy],
        linestyle="-",
        color=res.PARTICLE_COLORS[pk],
    )
    ax_legend.text(
        x=0.4,
        y=_yy,
        s=pk,
        fontsize=8,
        verticalalignment="center",
    )

_yy -= 0.1
ax_legend.text(
    x=0.2,
    y=_yy,
    s="zenith ranges",
    fontsize=8,
    verticalalignment="center",
)

for zdbin in range(zenith_bin["num"]):
    _yy -= 0.1
    ax_legend.plot(
        [0, 0.3],
        [_yy, _yy],
        linestyle=linestyles[zdbin],
        color="grey",
    )
    ax_legend.text(
        x=0.4,
        y=_yy,
        s=irf.summary.make_angle_range_str(
            start_rad=zenith_bin["edges"][zdbin],
            stop_rad=zenith_bin["edges"][zdbin + 1],
        ),
        fontsize=8,
        verticalalignment="center",
    )
ax_legend.set_xlim([0, 1])

res.ax_add_site_marker(ax, x=0.8, y=0.1)
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
    fig = sebplt.figure(style=sebplt.FIGURE_1_1)
    ax = sebplt.add_axes(fig=fig, span=[0.175, 0.15, 0.70, 0.8])
    ax_zd = sebplt.add_axes_zenith_range_indicator(
        fig=fig,
        zenith_bin_edges_rad=zenith_bin["edges"],
        zenith_bin=zd,
        span=[0.85, 0.8, 0.15, 0.15],
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
    res.ax_add_site_marker(ax)
    ax.loglog()
    ax.set_ylim(ylim)
    ax.set_xlabel("num. bins above threshold / 1")
    ax.set_ylabel("relative intensity / 1")
    fig.savefig(opj(res.paths["out_dir"], f"what_was_throwm_zd{zd:d}.jpg"))
    sebplt.close(fig)

res.stop()
