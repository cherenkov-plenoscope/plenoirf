#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import rename_after_writing as rnw
import os
from os.path import join as opj
import json_utils
import numpy as np
import binning_utils
import sebastians_matplotlib_addons as sebplt
import copy
import warnings


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

energy_bin = res.energy_binning(key="5_bins_per_decade")
zenith_bin = res.zenith_binning(key="3_bins_per_45deg")

trigger_config = res.trigger

ttt_cache_path = opj(res.paths["cache_dir"], "ttt")
if not os.path.exists(ttt_cache_path):
    os.makedirs(ttt_cache_path, exist_ok=True)

    for pk in res.PARTICLES:
        tpk = {}
        tpk["ratio"] = np.zeros(
            shape=(
                zenith_bin["num"],
                energy_bin["num"],
                trigger_config["foci_bin"]["num"],
            )
        )
        tpk["num_thrown"] = np.zeros(
            shape=(zenith_bin["num"], energy_bin["num"])
        )
        tpk["num_have_at_least_one_focus_trigger"] = np.zeros(
            shape=(zenith_bin["num"], energy_bin["num"])
        )

        for zbin in range(zenith_bin["num"]):
            ZENITH_CORRECTED_TRIGGER_THRESHOLD_PE = irf.light_field_trigger.get_trigger_threshold_corrected_for_pointing_zenith(
                pointing_zenith_rad=zenith_bin["centers"][zbin],
                trigger=trigger_config,
                nominal_threshold_pe=trigger_config["threshold_pe"],
            )

            for ebin in range(energy_bin["num"]):
                print(pk, zbin, ebin)

                table = res.event_table(particle_key=pk).query(
                    energy_start_GeV=energy_bin["edges"][ebin],
                    energy_stop_GeV=energy_bin["edges"][ebin + 1],
                    zenith_start_rad=zenith_bin["edges"][zbin],
                    zenith_stop_rad=zenith_bin["edges"][zbin + 1],
                    levels_and_columns={
                        "trigger": "__all__",
                    },
                )
                num_energy_zenith = table["trigger"].shape[0]

                tpk["num_thrown"][zbin][ebin] = num_energy_zenith

                trigger_in_energy_zenith = np.zeros(
                    shape=(
                        num_energy_zenith,
                        trigger_config["foci_bin"]["num"],
                    ),
                    dtype=bool,
                )
                for fbin in range(trigger_config["foci_bin"]["num"]):
                    fkey = f"focus_{fbin:02d}_response_pe"
                    trigger_in_energy_zenith[:, fbin] = (
                        table["trigger"][fkey]
                        >= ZENITH_CORRECTED_TRIGGER_THRESHOLD_PE
                    )

                num_passed_trigger_in_energy_zenith = np.sum(
                    trigger_in_energy_zenith, axis=0
                )
                if num_energy_zenith > 0:
                    tpk["ratio"][zbin][ebin] = (
                        num_passed_trigger_in_energy_zenith / num_energy_zenith
                    )

                __sum = np.sum(tpk["ratio"][zbin][ebin])
                if __sum > 0:
                    tpk["ratio"][zbin][ebin] /= __sum

                num_have_at_least_one_focus_trigger = np.sum(
                    np.sum(trigger_in_energy_zenith, axis=1) > 0
                )
                tpk["num_have_at_least_one_focus_trigger"][zbin][
                    ebin
                ] = num_have_at_least_one_focus_trigger

        with rnw.open(opj(ttt_cache_path, f"{pk:s}.json"), "wt") as fout:
            fout.write(json_utils.dumps(tpk))

ttt = json_utils.tree.Tree(ttt_cache_path)
cmap = irf.summary.figure.make_particle_colormaps(
    particle_colors=irf.summary.figure.PARTICLE_COLORS
)

for pk in ttt:
    for zbin in range(zenith_bin["num"]):
        valid_statistics = (
            ttt[pk]["num_have_at_least_one_focus_trigger"][zbin] >= 15
        )
        ratio = copy.copy(ttt[pk]["ratio"][zbin])
        for ebin in range(energy_bin["num"]):
            if not valid_statistics[ebin]:
                ratio[ebin, :] = float("nan")

        fig = sebplt.figure(sebplt.FIGURE_1_1)
        ax_c = sebplt.add_axes(fig=fig, span=[0.15, 0.27, 0.65, 0.65])
        ax_h = sebplt.add_axes(fig=fig, span=[0.15, 0.11, 0.65, 0.1])
        ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
        ax_zd = sebplt.add_axes_zenith_range_indicator(
            fig=fig,
            zenith_bin_edges_rad=zenith_bin["edges"],
            zenith_bin=zbin,
            span=[0.85, 0.1, 0.12, 0.12],
            fontsize=6,
        )

        _pcm_confusion = ax_c.pcolormesh(
            energy_bin["edges"],
            trigger_config["foci_bin"]["edges"],
            np.transpose(ratio),
            cmap=cmap[pk],
            norm=sebplt.plt_colors.PowerNorm(gamma=1.0),
        )
        ax_c.set_xlim(energy_bin["limits"])
        ax_c.set_ylim(1e3, 100e3)
        ax_c.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
        ax_c.set_xticklabels([])
        ax_c.set_ylabel("object distance / m")
        ax_c.loglog()

        sebplt.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")

        ax_h.semilogx()
        ax_h.set_xlim(energy_bin["limits"])
        ax_h.set_ylim(
            [
                1,
                1.1
                * np.max(ttt[pk]["num_have_at_least_one_focus_trigger"][zbin]),
            ]
        )
        ax_h.set_xlabel("energy / GeV")
        ax_h.set_ylabel("num. events / 1")
        sebplt.ax_add_histogram(
            ax=ax_h,
            bin_edges=energy_bin["edges"],
            bincounts=ttt[pk]["num_have_at_least_one_focus_trigger"][zbin],
            linestyle="-",
            linecolor="k",
        )

        fig.savefig(
            opj(
                res.paths["out_dir"],
                f"{pk:s}_trigger_probability_vs_object_distance_in_zenith_bin_{zbin:02d}.jpg",
            )
        )
        sebplt.close(fig)


for zbin in range(zenith_bin["num"]):
    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    ax_zd = sebplt.add_axes_zenith_range_indicator(
        fig=fig,
        zenith_bin_edges_rad=zenith_bin["edges"],
        zenith_bin=zbin,
        span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
        fontsize=5,
    )

    ax.set_xlim(energy_bin["limits"])
    ax.set_ylim([1e-4, 1])
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel(
        "probability any refocused trigger\nimage is above threshold / 1"
    )
    ax.loglog()

    for pk in ttt:

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in divide"
            )

            ratio = (
                ttt[pk]["num_have_at_least_one_focus_trigger"][zbin]
                / ttt[pk]["num_thrown"][zbin]
            )
            ratio_au = (
                np.sqrt(ttt[pk]["num_have_at_least_one_focus_trigger"][zbin])
                / ttt[pk]["num_thrown"][zbin]
            )

        sebplt.ax_add_histogram(
            ax=ax,
            bin_edges=energy_bin["edges"],
            bincounts=ratio,
            bincounts_lower=ratio - ratio_au,
            bincounts_upper=ratio + ratio_au,
            linestyle="-",
            face_alpha=0.2,
            face_color=res.PARTICLE_COLORS[pk],
            linecolor=res.PARTICLE_COLORS[pk],
        )

    fig.savefig(
        opj(
            res.paths["out_dir"],
            f"trigger_probability_in_zenith_bin_{zbin:02d}.jpg",
        )
    )
    sebplt.close(fig)


"""
quantile 50
"""
qqq = {}
for pk in res.PARTICLES:
    qqq[pk] = np.nan * np.ones(shape=(zenith_bin["num"], energy_bin["num"]))
    for zbin in range(zenith_bin["num"]):
        for ebin in range(energy_bin["num"]):
            bin_counts = ttt[pk]["ratio"][zbin][ebin, :]
            if not np.any(np.isnan(bin_counts)) and np.sum(bin_counts) > 0:
                qqq[pk][zbin][ebin] = binning_utils.quantile(
                    bin_counts=bin_counts,
                    bin_edges=trigger_config["foci_bin"]["edges"],
                    q=0.5,
                )


yticks = trigger_config["foci_bin"]["edges"]
ytick_labels = [f"{depth:.0f}" for depth in yticks]

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.set_xlim(energy_bin["limits"])
ax.set_ylim(trigger_config["foci_bin"]["limits"])
ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
ax.set_xlabel("energy / GeV")
ax.set_ylabel("object distance / m")
ax.loglog()
ax.set_yticks(ticks=yticks, labels=ytick_labels, minor=False)
ax.set_yticks(ticks=[], labels=[], minor=True)

zenith_bin_linestyles = ["-", "--", ":"]
for pk in res.PARTICLES:
    for zbin in range(zenith_bin["num"]):
        ax.plot(
            energy_bin["centers"],
            qqq[pk][zbin],
            color=res.PARTICLE_COLORS[pk],
            linestyle=zenith_bin_linestyles[zbin],
        )
fig.savefig(
    opj(
        res.paths["out_dir"],
        f"highest_trigger_probability_vs_zenith.jpg",
    )
)
sebplt.close(fig)


res.stop()
