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
import shutil


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

zenith_bin = res.zenith_binning(key="3_bins_per_45deg")

trigger = res.trigger

accepting_height_above_observation_level_m = (
    trigger["modus"]["accepting_altitude_asl_m"]
    - res.SITE["observation_level_asl_m"]
)
rejecting_height_above_observation_level_m = (
    trigger["modus"]["rejecting_altitude_asl_m"]
    - res.SITE["observation_level_asl_m"]
)

TRIGGER_MODES = [
    "far_accepting_focus_and_near_rejecting_focus",
    "far_accepting_focus",
]


def make_event_table_with_search_index(
    out_path,
    out_dtypes,
    map_path,
    search_index_config,
):
    os.makedirs(out_path, exist_ok=True)
    irf.event_table.search_index.initializing._populated_bins_step_two(
        out_path=os.path.join(out_path, "bins"),
        stage_path=map_path,
        num_zenith_bins=search_index_config["zenith_bin"]["num"],
        num_energy_bins=search_index_config["energy_bin"]["num"],
        dtypes=out_dtypes,
    )
    irf.event_table.search_index.utils.write_config(
        work_dir=out_path,
        zenith_bin_edges=search_index_config["zenith_bin"]["edges"],
        energy_bin_edges=search_index_config["energy_bin"]["edges"],
    )


"""
SIZE_BIN_EDGES = np.array(
    sorted(list(set(np.round(np.geomspace(80, 8_000, 100)))))
)


fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
x_zd_rad = np.linspace(0, np.deg2rad(45), 1337)
ax.plot(
    np.rad2deg(x_zd_rad),
    irf.light_field_trigger.get_trigger_threshold_corrected_for_pointing_zenith(
        pointing_zenith_rad=x_zd_rad,
        trigger=trigger,
        nominal_threshold_pe=trigger["threshold_pe"],
    ),
    color="black",
)
ax.set_xlabel(r"zenith / (1$^{\circ}$)")
ax.set_ylabel("sum trigger threshold / p.e.")
ax.set_xlim([0, 50])
fig.savefig(
    opj(
        res.paths["out_dir"],
        "threshold_vs_zenith.jpg",
    )
)
sebplt.close(fig)



def explore_focus_ratios(
    uids,
    focus_response_pe,
    accepting_response_pe,
    zenith_corrected_threshold_pe,
    trigger_focus_bin_edges,
    pk,
    small_size_ratio_threschold,
):
    num_foci = focus_response_pe.shape[1]
    num_events = uids.shape[0]
    assert accepting_response_pe.shape[0] == num_events
    assert focus_response_pe.shape[0] == num_events
    assert zenith_corrected_threshold_pe.shape[0] == num_events

    mask_size_over_threshold = (
        accepting_response_pe >= zenith_corrected_threshold_pe
    )

    num_ratios = 30
    ratio_bin = binning_utils.Binning(
        bin_edges=np.geomspace(0.5, 1 / 0.5, num_ratios + 1)
    )

    zdfocrat = np.zeros(shape=(zenith_bin["num"], num_foci, num_ratios))

    for zd in range(zenith_bin["num"]):
        zk = f"zd{zd:d}"
        mask_zenith = snt.logic.make_mask_of_right_in_left(
            left_indices=uids,
            right_indices=zenith_assignment[zk][pk],
        )
        mask = np.logical_and(mask_zenith, mask_size_over_threshold)

        zd_focus_response_pe = focus_response_pe[mask]
        zd_accepting_response_pe = accepting_response_pe[mask]

        for foc in range(num_foci):
            zd_accepting_over_rejecting = (
                zd_accepting_response_pe / zd_focus_response_pe[:, foc]
            )

            zdfocrat[zd, foc, :] = np.histogram(
                zd_accepting_over_rejecting,
                bins=ratio_bin["edges"],
            )[0]

    for zd in range(zenith_bin["num"]):
        zk = f"zd{zd:d}"
        fig = sebplt.figure(irf.summary.figure.style(key="6:7")[0])
        ax_c = sebplt.add_axes(fig=fig, span=[0.2, 0.2, 0.75, 0.75])
        # ax_h = sebplt.add_axes(fig=fig, span=[0.2, 0.13, 0.75, 0.10])
        ax_cb = sebplt.add_axes(fig=fig, span=[0.25, 0.96, 0.65, 0.015])
        sebplt.add_axes_zenith_range_indicator(
            fig=fig,
            span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
            zenith_bin_edges_rad=zenith_bin["edges"],
            zenith_bin=zd,
            fontsize=6,
        )
        _pcm_confusion = ax_c.pcolormesh(
            ratio_bin["edges"],
            trigger_focus_bin_edges,
            zdfocrat[zd, :, :],
            cmap=res.PARTICLE_COLORMAPS[pk],
            norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
        )
        ax_c.axvline(
            small_size_ratio_threschold,
            color="black",
            linestyle=":",
            alpha=0.25,
        )
        ax_c.loglog()
        sebplt.plt.colorbar(
            _pcm_confusion, cax=ax_cb, extend="max", orientation="horizontal"
        )
        ax_c.set_xlabel("response ratio\naccepting over rejecting / 1")
        ax_c.set_ylabel("rejecting focus depth / m")
        fig.savefig(
            opj(res.paths["out_dir"], f"{zk:s}_{pk:s}_focus_ratios.jpg")
        )
        sebplt.close(fig)

    size_bin = binning_utils.Binning(bin_edges=SIZE_BIN_EDGES)

    for zd in range(zenith_bin["num"]):
        zk = f"zd{zd:d}"

        mask_zenith = snt.logic.make_mask_of_right_in_left(
            left_indices=uids,
            right_indices=zenith_assignment[zk][pk],
        )
        zd_accepting_response_pe = accepting_response_pe[mask_zenith]

        hist_accepting_response = np.histogram(
            zd_accepting_response_pe,
            bins=size_bin["edges"],
        )[0]
        rel_hist_accepting_response = (
            hist_accepting_response / zd_accepting_response_pe.shape[0]
        )

        fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
        ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
        sebplt.ax_add_histogram(
            ax=ax,
            bin_edges=size_bin["edges"],
            bincounts=rel_hist_accepting_response,
            linestyle="-",
            linecolor=res.PARTICLE_COLORS[pk],
            linealpha=1.0,
            bincounts_upper=None,
            bincounts_lower=None,
            face_color=None,
            face_alpha=None,
            label=None,
            draw_bin_walls=False,
        )
        ax.loglog()
        ax.set_xlabel("accepting response / p.e.")
        ax.set_ylabel("rel. intensity / 1\n(of what was thrown)")
        ax.set_xlim(size_bin["limits"])
        ax.set_ylim([1e-6, 1.0])
        fig.savefig(
            opj(
                res.paths["out_dir"],
                f"{zk:s}_{pk:s}_accepting_response_histogram.jpg",
            )
        )
        sebplt.close(fig)

    zenith_assignment.clear_cache()

    return zdfocrat, ratio_bin
"""


rtsc_dtypes = {}
for irs in range(len(trigger["ratescan_thresholds_pe"])):
    nominal_threshold_pe = trigger["ratescan_thresholds_pe"][irs]
    level_key = f"{nominal_threshold_pe:d}pe"
    rtsc_dtypes[level_key] = [("uid", "<u8")]


for pk in res.PARTICLES:

    # make out dirs
    # -------------
    pk_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)
    for mode_key in TRIGGER_MODES:
        pk_mode_dir = opj(pk_dir, mode_key)
        os.makedirs(pk_mode_dir, exist_ok=True)

    event_table_bin_by_bin = res.event_table(particle_key=pk).query(
        levels_and_columns={
            "trigger": "__all__",
            "instrument_pointing": ("uid", "zenith_rad"),
        },
        bin_by_bin=True,
    )
    for table_bin, zd_en_bin in event_table_bin_by_bin:
        zdbin, enbin = zd_en_bin
        print(pk, f"zd: {zdbin:d}, en: {enbin:d}")

        uid_common = snt.logic.intersection(
            table_bin["trigger"]["uid"],
            table_bin["instrument_pointing"]["uid"],
        )
        table_bin = snt.logic.cut_and_sort_table_on_indices(
            table=table_bin,
            common_indices=uid_common,
            inplace=True,
        )

        num_events = table_bin["trigger"].shape[0]

        (accepting_focus, rejecting_focus) = (
            irf.light_field_trigger.assign_accepting_and_rejecting_focus_based_on_pointing_zenith(
                pointing_zenith_rad=table_bin["instrument_pointing"][
                    "zenith_rad"
                ],
                accepting_height_above_observation_level_m=accepting_height_above_observation_level_m,
                rejecting_height_above_observation_level_m=rejecting_height_above_observation_level_m,
                trigger_foci_bin_edges_m=trigger["foci_bin"]["edges"],
            )
        )

        assert accepting_focus.shape[0] == num_events
        assert rejecting_focus.shape[0] == num_events

        focus_response_pe = (
            irf.light_field_trigger.copy_focus_response_into_matrix(
                trigger_table=table_bin["trigger"]
            )
        )
        assert focus_response_pe.shape[0] == num_events
        assert focus_response_pe.shape[1] == trigger["foci_bin"]["num"]

        (accepting_response_pe, rejecting_response_pe) = (
            irf.light_field_trigger.find_accepting_and_rejecting_response(
                accepting_focus=accepting_focus,
                rejecting_focus=rejecting_focus,
                focus_response_pe=focus_response_pe,
            )
        )
        assert accepting_response_pe.shape[0] == num_events
        assert rejecting_response_pe.shape[0] == num_events

        threshold_accepting_over_rejecting = (
            irf.light_field_trigger.get_accepting_over_rejecting(
                trigger=trigger,
                accepting_response_pe=accepting_response_pe,
            )
        )

        accepting_over_rejecting = (
            accepting_response_pe / rejecting_response_pe
        )
        is_ratio_over_threshold = (
            accepting_over_rejecting >= threshold_accepting_over_rejecting
        )

        zenith_corrected_threshold_pe = irf.light_field_trigger.get_trigger_threshold_corrected_for_pointing_zenith(
            pointing_zenith_rad=table_bin["instrument_pointing"]["zenith_rad"],
            trigger=trigger,
            nominal_threshold_pe=trigger["threshold_pe"],
        )

        """
        _small_size_pe = [150]
        _pointing_to_zenith_rad = [0]
        _small_size_ratio_threschold = (
            irf.light_field_trigger.get_accepting_over_rejecting(
                pointing_zenith_rad=_pointing_to_zenith_rad,
                trigger=trigger,
                accepting_response_pe=_small_size_pe,
            )[0]
        )

        zdfocrat, ratio_bin = explore_focus_ratios(
            uids=table_bin["trigger"]["uid"],
            focus_response_pe=focus_response_pe,
            accepting_response_pe=accepting_response_pe,
            zenith_corrected_threshold_pe=zenith_corrected_threshold_pe,
            trigger_focus_bin_edges=trigger["foci_bin"]["edges"],
            pk=pk,
            small_size_ratio_threschold=_small_size_ratio_threschold,
        )
        """

        binname = f"zd{zdbin:06d}_en{enbin:06d}"

        # export
        # ------
        for irs in range(len(trigger["ratescan_thresholds_pe"])):
            nominal_threshold_pe = trigger["ratescan_thresholds_pe"][irs]
            level_key = f"{nominal_threshold_pe:d}pe"

            tele_level_dir = opj(
                pk_dir, "far_accepting_focus", "map", level_key
            )
            plen_level_dir = opj(
                pk_dir,
                "far_accepting_focus_and_near_rejecting_focus",
                "map",
                level_key,
            )

            os.makedirs(tele_level_dir, exist_ok=True)
            os.makedirs(plen_level_dir, exist_ok=True)

            zenith_corrected_threshold_pe = irf.light_field_trigger.get_trigger_threshold_corrected_for_pointing_zenith(
                pointing_zenith_rad=table_bin["instrument_pointing"][
                    "zenith_rad"
                ],
                trigger=trigger,
                nominal_threshold_pe=nominal_threshold_pe,
            )

            is_size_over_threshold = (
                accepting_response_pe >= zenith_corrected_threshold_pe
            )
            is_pasttrigger_plen = np.logical_and(
                is_size_over_threshold, is_ratio_over_threshold
            )

            uid_pasttrigger_plen = table_bin["trigger"]["uid"][
                is_pasttrigger_plen
            ]
            uid_pasttrigger_tele = table_bin["trigger"]["uid"][
                is_size_over_threshold
            ]

            """
            _num_is_size = sum(is_size_over_threshold)
            _num_past = sum(is_pasttrigger)
            _loss = (_num_is_size - _num_past) / _num_is_size

            print(
                f"{pk:s}, num size: {_num_is_size:d}, loss: {_loss*1e2:.2f}%"
            )
            """
            filename = f"{binname:s}.recarray"

            plen_thr_bin_path = opj(plen_level_dir, filename)
            with rnw.open(plen_thr_bin_path, "wb") as fout:
                fout.write(uid_pasttrigger_plen.astype("u8").tobytes())

            tele_thr_bin_path = opj(tele_level_dir, filename)
            with rnw.open(tele_thr_bin_path, "wb") as fout:
                fout.write(uid_pasttrigger_tele.astype("u8").tobytes())

    for mode_key in TRIGGER_MODES:
        make_event_table_with_search_index(
            out_path=opj(pk_dir, mode_key, "ratescan"),
            out_dtypes=rtsc_dtypes,
            map_path=opj(pk_dir, mode_key, "map"),
            search_index_config=event_table_bin_by_bin.event_table.config,
        )
        shutil.rmtree(opj(pk_dir, mode_key, "map"))

res.stop()
