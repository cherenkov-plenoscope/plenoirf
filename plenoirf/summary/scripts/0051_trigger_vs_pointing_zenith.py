#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
import json_utils
import numpy as np
import sebastians_matplotlib_addons as sebplt
import copy


paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)
sebplt.matplotlib.rcParams.update(res.analysis["plot"]["matplotlib"])

energy_bin = json_utils.read(
    os.path.join(paths["analysis_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

zenith_bin = res.ZenithBinning(key="once")
trigger_object_distances_m = res.config["sum_trigger"]["object_distances_m"]
trigger_decade_step = 10 ** (1 / 22)
trigger_object_distances_bin_edges_m = np.geomspace(
    min(trigger_object_distances_m) / trigger_decade_step,
    max(trigger_object_distances_m) * trigger_decade_step,
    len(trigger_object_distances_m) * 2 + 1,
)
_trigger_bin_edges_mask = np.arange(
    0, len(trigger_object_distances_bin_edges_m), 2
)
trigger_object_distances_bin_edges_m = trigger_object_distances_bin_edges_m[
    _trigger_bin_edges_mask
]

TRIGGER_THRESHOLD_PE = res.analysis["trigger"][res.site_key]["threshold_pe"]
SOFT_TRIGGER_THRESHOLD_PE = TRIGGER_THRESHOLD_PE


ttt = {}
for pk in res.PARTICLES:
    ttt[pk] = {}

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={
                "primary": ["uid", "energy_GeV"],
                "trigger": "__all__",
                "instrument_pointing": "__all__",
            }
        )

    uid_common = snt.logic.intersection(
        [
            event_table["primary"]["uid"],
            event_table["trigger"]["uid"],
            event_table["instrument_pointing"]["uid"],
        ]
    )

    et = snt.logic.cut_and_sort_table_on_indices(
        table=event_table,
        common_indices=uid_common,
    )

    ttt[pk]["ratio"] = np.zeros(
        shape=(
            zenith_bin.num,
            energy_bin["num_bins"],
            len(trigger_object_distances_m),
        )
    )
    ttt[pk]["num_thrown"] = np.zeros(
        shape=(zenith_bin.num, energy_bin["num_bins"])
    )

    ttt[pk]["num_have_at_least_one_focus_trigger"] = np.zeros(
        shape=(zenith_bin.num, energy_bin["num_bins"])
    )

    for zzz in range(zenith_bin.num):
        zenith_start_rad = zenith_bin.edges[zzz]
        zenith_stop_rad = zenith_bin.edges[zzz + 1]

        zenith_mask = np.logical_and(
            et["instrument_pointing"]["zenith_rad"] >= zenith_start_rad,
            et["instrument_pointing"]["zenith_rad"] < zenith_stop_rad,
        )

        for eee in range(energy_bin["num_bins"]):
            energy_start_GeV = energy_bin["edges"][eee]
            energy_stop_GeV = energy_bin["edges"][eee + 1]

            energy_mask = np.logical_and(
                et["primary"]["energy_GeV"] >= energy_start_GeV,
                et["primary"]["energy_GeV"] < energy_stop_GeV,
            )

            mask_energy_zenith = np.logical_and(
                energy_mask,
                zenith_mask,
            )
            num_energy_zenith = np.sum(mask_energy_zenith)

            ttt[pk]["num_thrown"][zzz][eee] = num_energy_zenith

            trigger_in_energy_zenith = np.zeros(
                shape=(num_energy_zenith, len(trigger_object_distances_m)),
                dtype=bool,
            )
            for fff in range(len(trigger_object_distances_m)):
                fff_key = f"focus_{fff:02d}_response_pe"
                trigger_in_energy_zenith[:, fff] = (
                    et["trigger"][fff_key][mask_energy_zenith]
                    >= SOFT_TRIGGER_THRESHOLD_PE
                )

            num_passed_trigger_in_energy_zenith = np.sum(
                trigger_in_energy_zenith, axis=0
            )
            ttt[pk]["ratio"][zzz][eee] = (
                num_passed_trigger_in_energy_zenith / num_energy_zenith
            )
            ttt[pk]["ratio"][zzz][eee] /= np.sum(ttt[pk]["ratio"][zzz][eee])

            num_have_at_least_one_focus_trigger = np.sum(
                np.sum(trigger_in_energy_zenith, axis=1) > 0
            )
            ttt[pk]["num_have_at_least_one_focus_trigger"][zzz][
                eee
            ] = num_have_at_least_one_focus_trigger

cmap = irf.summary.figure.make_particle_colormaps(
    particle_colors=irf.summary.figure.PARTICLE_COLORS
)

for pk in ttt:
    for zzz in range(zenith_bin.num):
        valid_statistics = (
            ttt[pk]["num_have_at_least_one_focus_trigger"][zzz] >= 15
        )
        ratio = copy.copy(ttt[pk]["ratio"][zzz])
        for eee in range(energy_bin["num_bins"]):
            if not valid_statistics[eee]:
                ratio[eee, :] = float("nan")

        fig = sebplt.figure(sebplt.FIGURE_1_1)
        ax_c = sebplt.add_axes(fig=fig, span=[0.15, 0.27, 0.65, 0.65])
        ax_h = sebplt.add_axes(fig=fig, span=[0.15, 0.11, 0.65, 0.1])
        ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
        ax_zd = sebplt.add_axes_zenith_range_indicator(
            fig=fig,
            zenith_bin_edges_rad=zenith_bin.edges,
            zenith_bin=zzz,
            span=[0.85, 0.1, 0.12, 0.12],
            fontsize=6,
        )

        _pcm_confusion = ax_c.pcolormesh(
            energy_bin["edges"],
            trigger_object_distances_bin_edges_m,
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
                * np.max(ttt[pk]["num_have_at_least_one_focus_trigger"][zzz]),
            ]
        )
        ax_h.set_xlabel("energy / GeV")
        ax_h.set_ylabel("num. events / 1")
        sebplt.ax_add_histogram(
            ax=ax_h,
            bin_edges=energy_bin["edges"],
            bincounts=ttt[pk]["num_have_at_least_one_focus_trigger"][zzz],
            linestyle="-",
            linecolor="k",
        )

        fig.savefig(
            os.path.join(
                paths["out_dir"],
                f"{pk:s}_trigger_probability_vs_object_distance_in_zenith_bin_{zzz:02d}.jpg",
            )
        )
        sebplt.close(fig)


for zzz in range(zenith_bin.num):
    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    ax_zd = sebplt.add_axes_zenith_range_indicator(
        fig=fig,
        zenith_bin_edges_rad=zenith_bin.edges,
        zenith_bin=zzz,
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

        ratio = (
            ttt[pk]["num_have_at_least_one_focus_trigger"][zzz]
            / ttt[pk]["num_thrown"][zzz]
        )
        ratio_au = (
            np.sqrt(ttt[pk]["num_have_at_least_one_focus_trigger"][zzz])
            / ttt[pk]["num_thrown"][zzz]
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
        os.path.join(
            paths["out_dir"],
            f"trigger_probability_in_zenith_bin_{zzz:02d}.jpg",
        )
    )
    sebplt.close(fig)
