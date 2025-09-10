#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
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

energy_bin = res.energy_binning(key="trigger_acceptance_onregion")
zenith_bin = res.zenith_binning(key="once")
zenith_assignment = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0019_zenith_bin_assignment")
)
trigger = res.trigger

ttt = {}
for pk in res.PARTICLES:
    ttt[pk] = {}
    ttt[pk]["ratio"] = np.zeros(
        shape=(
            zenith_bin["num"],
            energy_bin["num"],
            trigger["foci_bin"]["num"],
        )
    )
    ttt[pk]["num_thrown"] = np.zeros(
        shape=(zenith_bin["num"], energy_bin["num"])
    )
    ttt[pk]["num_have_at_least_one_focus_trigger"] = np.zeros(
        shape=(zenith_bin["num"], energy_bin["num"])
    )

    for zd in range(zenith_bin["num"]):
        zk = f"zd{zd:d}"

        with res.open_event_table(particle_key=pk) as arc:
            zd_evttab = arc.query(
                levels_and_columns={
                    "primary": ["uid", "energy_GeV"],
                    "trigger": "__all__",
                    "instrument_pointing": ["uid", "zenith_rad"],
                },
                indices=zenith_assignment[zk][pk],
            )

        ZENITH_CORRECTED_TRIGGER_THRESHOLD_PE = irf.light_field_trigger.get_trigger_threshold_corrected_for_pointing_zenith(
            pointing_zenith_rad=zenith_bin["centers"][zd],
            trigger=trigger,
            nominal_threshold_pe=trigger["threshold_pe"],
        )

        uid_zd_common = snt.logic.intersection(
            zd_evttab["primary"]["uid"],
            zd_evttab["trigger"]["uid"],
            zd_evttab["instrument_pointing"]["uid"],
        )

        zd_evttab = snt.logic.cut_and_sort_table_on_indices(
            table=zd_evttab,
            common_indices=uid_zd_common,
            inplace=True,
        )

        for eee in range(energy_bin["num"]):
            energy_start_GeV = energy_bin["edges"][eee]
            energy_stop_GeV = energy_bin["edges"][eee + 1]

            energy_mask = np.logical_and(
                zd_evttab["primary"]["energy_GeV"] >= energy_start_GeV,
                zd_evttab["primary"]["energy_GeV"] < energy_stop_GeV,
            )

            num_energy_zenith = np.sum(energy_mask)

            ttt[pk]["num_thrown"][zd][eee] = num_energy_zenith

            trigger_in_energy_zenith = np.zeros(
                shape=(num_energy_zenith, trigger["foci_bin"]["num"]),
                dtype=bool,
            )
            for fff in range(trigger["foci_bin"]["num"]):
                fff_key = f"focus_{fff:02d}_response_pe"
                trigger_in_energy_zenith[:, fff] = (
                    zd_evttab["trigger"][fff_key][energy_mask]
                    >= ZENITH_CORRECTED_TRIGGER_THRESHOLD_PE
                )

            num_passed_trigger_in_energy_zenith = np.sum(
                trigger_in_energy_zenith, axis=0
            )
            if num_energy_zenith > 0:
                ttt[pk]["ratio"][zd][eee] = (
                    num_passed_trigger_in_energy_zenith / num_energy_zenith
                )

            __sum = np.sum(ttt[pk]["ratio"][zd][eee])
            if __sum > 0:
                ttt[pk]["ratio"][zd][eee] /= __sum

            num_have_at_least_one_focus_trigger = np.sum(
                np.sum(trigger_in_energy_zenith, axis=1) > 0
            )
            ttt[pk]["num_have_at_least_one_focus_trigger"][zd][
                eee
            ] = num_have_at_least_one_focus_trigger


del zd_evttab

cmap = irf.summary.figure.make_particle_colormaps(
    particle_colors=irf.summary.figure.PARTICLE_COLORS
)

for pk in ttt:
    for zd in range(zenith_bin["num"]):
        valid_statistics = (
            ttt[pk]["num_have_at_least_one_focus_trigger"][zd] >= 15
        )
        ratio = copy.copy(ttt[pk]["ratio"][zd])
        for eee in range(energy_bin["num"]):
            if not valid_statistics[eee]:
                ratio[eee, :] = float("nan")

        fig = sebplt.figure(sebplt.FIGURE_1_1)
        ax_c = sebplt.add_axes(fig=fig, span=[0.15, 0.27, 0.65, 0.65])
        ax_h = sebplt.add_axes(fig=fig, span=[0.15, 0.11, 0.65, 0.1])
        ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
        ax_zd = sebplt.add_axes_zenith_range_indicator(
            fig=fig,
            zenith_bin_edges_rad=zenith_bin["edges"],
            zenith_bin=zd,
            span=[0.85, 0.1, 0.12, 0.12],
            fontsize=6,
        )

        _pcm_confusion = ax_c.pcolormesh(
            energy_bin["edges"],
            trigger["foci_bin"]["edges"],
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
                * np.max(ttt[pk]["num_have_at_least_one_focus_trigger"][zd]),
            ]
        )
        ax_h.set_xlabel("energy / GeV")
        ax_h.set_ylabel("num. events / 1")
        sebplt.ax_add_histogram(
            ax=ax_h,
            bin_edges=energy_bin["edges"],
            bincounts=ttt[pk]["num_have_at_least_one_focus_trigger"][zd],
            linestyle="-",
            linecolor="k",
        )

        fig.savefig(
            opj(
                res.paths["out_dir"],
                f"{pk:s}_trigger_probability_vs_object_distance_in_zenith_bin_{zd:02d}.jpg",
            )
        )
        sebplt.close(fig)


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
                ttt[pk]["num_have_at_least_one_focus_trigger"][zd]
                / ttt[pk]["num_thrown"][zd]
            )
            ratio_au = (
                np.sqrt(ttt[pk]["num_have_at_least_one_focus_trigger"][zd])
                / ttt[pk]["num_thrown"][zd]
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
            f"trigger_probability_in_zenith_bin_{zd:02d}.jpg",
        )
    )
    sebplt.close(fig)


"""
quantile 50
"""
qqq = {}
for pk in res.PARTICLES:
    qqq[pk] = np.nan * np.ones(shape=(zenith_bin["num"], energy_bin["num"]))
    for zd in range(zenith_bin["num"]):
        for ee in range(energy_bin["num"]):
            bin_counts = ttt[pk]["ratio"][zd][ee, :]
            if not np.any(np.isnan(bin_counts)) and np.sum(bin_counts) > 0:
                qqq[pk][zd][ee] = binning_utils.quantile(
                    bin_counts=bin_counts,
                    bin_edges=trigger["foci_bin"]["edges"],
                    q=0.5,
                )


yticks = trigger["foci_bin"]["edges"]
ytick_labels = [f"{depth:.0f}" for depth in yticks]

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.set_xlim(energy_bin["limits"])
ax.set_ylim(trigger["foci_bin"]["limits"])
ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
ax.set_xlabel("energy / GeV")
ax.set_ylabel("object distance / m")
ax.loglog()
ax.set_yticks(ticks=yticks, labels=ytick_labels, minor=False)
ax.set_yticks(ticks=[], labels=[], minor=True)

zenith_bin_linestyles = ["-", "--", ":"]
for pk in res.PARTICLES:
    for zd in range(zenith_bin["num"]):
        ax.plot(
            energy_bin["centers"],
            qqq[pk][zd],
            color=res.PARTICLE_COLORS[pk],
            linestyle=zenith_bin_linestyles[zd],
        )
fig.savefig(
    opj(
        res.paths["out_dir"],
        f"highest_trigger_probability_vs_zenith.jpg",
    )
)
sebplt.close(fig)


res.stop()
