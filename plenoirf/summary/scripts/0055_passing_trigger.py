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


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

zenith_bin = res.zenith_binning(key="once")
zenith_assignment = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0019_zenith_bin_assignment")
)

trigger = res.trigger

accepting_height_above_observation_level_m = (
    trigger["modus"]["accepting_altitude_asl_m"]
    - res.SITE["observation_level_asl_m"]
)
rejecting_height_above_observation_level_m = (
    trigger["modus"]["rejecting_altitude_asl_m"]
    - res.SITE["observation_level_asl_m"]
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


SIZE_BIN_EDGES = np.array(
    sorted(list(set(np.round(np.geomspace(80, 8_000, 100)))))
)


def has_focus_above_threshold(
    focus_response_pe, zenith_corrected_threshold_pe
):
    num_events, num_foci = focus_response_pe.shape
    assert zenith_corrected_threshold_pe.shape[0] == num_events

    is_above = np.zeros(shape=focus_response_pe.shape, dtype=bool)
    for eee in range(num_events):
        is_above[eee] = (
            focus_response_pe[eee, :] >= zenith_corrected_threshold_pe[eee]
        )

    return is_above


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
        fig = sebplt.figure(style=sebplt.FIGURE_1_1)
        ax_c = sebplt.add_axes(fig=fig, span=[0.25, 0.27, 0.55, 0.65])
        # ax_h = sebplt.add_axes(fig=fig, span=[0.25, 0.11, 0.55, 0.1])
        ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
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
        sebplt.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
        ax_c.set_xlabel("accepting over rejecting  / 1")
        ax_c.set_ylabel("rejecting focus / m")
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

    return zdfocrat, ratio_bin


def print_info(
    uids,
    focus_response_pe,
    accepting_over_rejecting,
    zenith_corrected_threshold_pe,
    trigger_focus_bin_edges,
    pk,
):
    mask_above_threshold = has_focus_above_threshold(
        focus_response_pe=focus_response_pe,
        zenith_corrected_threshold_pe=zenith_corrected_threshold_pe,
    )

    num_events = mask_above_threshold.shape[0]
    num_foci = focus_response_pe.shape[1]
    assert num_events == focus_response_pe.shape[0]
    assert num_events == zenith_corrected_threshold_pe.shape[0]

    edges = trigger_focus_bin_edges

    for zd in range(zenith_bin["num"]):
        zk = f"zd{zd:d}"

        mask_zenith = snt.logic.make_mask_of_right_in_left(
            left_indices=uids,
            right_indices=zenith_assignment[zk][pk],
        )
        num_zenith = sum(mask_zenith)

        foci_ratio = np.zeros(shape=num_foci, dtype=float)
        for ff in range(num_foci):
            mask = np.logical_and(mask_above_threshold[:, ff], mask_zenith)
            foci_ratio[ff] = sum(mask) / num_zenith

        fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
        ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
        sebplt.ax_add_histogram(
            ax=ax,
            bin_edges=trigger_focus_bin_edges,
            bincounts=foci_ratio,
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
        ax.set_xlabel("ratio over threshold / 1")
        ax.set_ylabel("sum trigger threshold / p.e.")
        ax.set_xlim([min(edges), max(edges)])
        ax.set_ylim([1e-6, 1.0])
        fig.savefig(
            opj(res.paths["out_dir"], f"{zk:s}_{pk:s}_focus_layer.jpg")
        )
        sebplt.close(fig)


for pk in res.PARTICLES:
    print(pk)
    pk_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={
                "trigger": "__all__",
                "instrument_pointing": "__all__",
            }
        )
    event_table = snt.logic.cut_and_sort_table_on_indices(
        table=event_table,
        common_indices=snt.logic.intersection(
            event_table["trigger"]["uid"],
            event_table["instrument_pointing"]["uid"],
        ),
    )

    num_events = event_table["trigger"].shape[0]

    (accepting_focus, rejecting_focus) = (
        irf.light_field_trigger.assign_accepting_and_rejecting_focus_based_on_pointing_zenith(
            pointing_zenith_rad=event_table["instrument_pointing"][
                "zenith_rad"
            ],
            accepting_height_above_observation_level_m=accepting_height_above_observation_level_m,
            rejecting_height_above_observation_level_m=rejecting_height_above_observation_level_m,
            trigger_foci_bin_edges_m=trigger["foci_bin"]["edges"],
        )
    )
    # rejecting_focus = accepting_focus - 1  # one focus layer below

    assert accepting_focus.shape[0] == num_events
    assert rejecting_focus.shape[0] == num_events

    focus_response_pe = (
        irf.light_field_trigger.copy_focus_response_into_matrix(
            trigger_table=event_table["trigger"]
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
            pointing_zenith_rad=event_table["instrument_pointing"][
                "zenith_rad"
            ],
            trigger=trigger,
            accepting_response_pe=accepting_response_pe,
        )
    )

    accepting_over_rejecting = accepting_response_pe / rejecting_response_pe
    is_ratio_over_threshold = (
        accepting_over_rejecting >= threshold_accepting_over_rejecting
    )

    zenith_corrected_threshold_pe = irf.light_field_trigger.get_trigger_threshold_corrected_for_pointing_zenith(
        pointing_zenith_rad=event_table["instrument_pointing"]["zenith_rad"],
        trigger=trigger,
        nominal_threshold_pe=trigger["threshold_pe"],
    )

    print_info(
        uids=event_table["trigger"]["uid"],
        focus_response_pe=focus_response_pe,
        accepting_over_rejecting=accepting_over_rejecting,
        zenith_corrected_threshold_pe=zenith_corrected_threshold_pe,
        trigger_focus_bin_edges=trigger["foci_bin"]["edges"],
        pk=pk,
    )

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
        uids=event_table["trigger"]["uid"],
        focus_response_pe=focus_response_pe,
        accepting_response_pe=accepting_response_pe,
        zenith_corrected_threshold_pe=zenith_corrected_threshold_pe,
        trigger_focus_bin_edges=trigger["foci_bin"]["edges"],
        pk=pk,
        small_size_ratio_threschold=_small_size_ratio_threschold,
    )

    # export
    # ------
    pk_ratescan_dir = opj(pk_dir, "ratescan")
    os.makedirs(pk_ratescan_dir, exist_ok=True)

    for irs in range(len(trigger["ratescan_thresholds_pe"])):
        nominal_threshold_pe = trigger["ratescan_thresholds_pe"][irs]

        zenith_corrected_threshold_pe = irf.light_field_trigger.get_trigger_threshold_corrected_for_pointing_zenith(
            pointing_zenith_rad=event_table["instrument_pointing"][
                "zenith_rad"
            ],
            trigger=trigger,
            nominal_threshold_pe=nominal_threshold_pe,
        )

        is_size_over_threshold = (
            accepting_response_pe >= zenith_corrected_threshold_pe
        )
        is_pasttrigger = np.logical_and(
            is_size_over_threshold, is_ratio_over_threshold
        )

        _num_is_size = sum(is_size_over_threshold)
        _num_past = sum(is_pasttrigger)
        _loss = (_num_is_size - _num_past) / _num_is_size

        print(f"{pk:s}, num size: {_num_is_size:d}, loss: {_loss*1e2:.2f}%")

        filename = f"{trigger['ratescan_thresholds_pe'][irs]:d}pe.json"
        uids_pasttrigger = event_table["trigger"]["uid"][is_pasttrigger]

        json_utils.write(
            opj(pk_ratescan_dir, filename),
            {"uid": uids_pasttrigger},
        )

        if nominal_threshold_pe == trigger["threshold_pe"]:
            json_utils.write(
                opj(pk_dir, "uid.json"),
                uids_pasttrigger,
            )

            pk_only_accepting_dir = opj(pk_dir, "only_accepting_not_rejecting")
            os.makedirs(pk_only_accepting_dir, exist_ok=True)

            json_utils.write(
                opj(pk_only_accepting_dir, "uid.json"),
                event_table["trigger"]["uid"][is_size_over_threshold],
            )

res.stop()
