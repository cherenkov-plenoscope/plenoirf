#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import json_utils
import numpy as np
import sebastians_matplotlib_addons as sebplt


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

trigger = res.trigger

TRIGGER_ACCEPTING_ALTITUDE_ASL_M = 19_856
TRIGGER_REJECTING_ALTITUDE_ASL_M = 13_851

accepting_height_above_observation_level_m = (
    TRIGGER_ACCEPTING_ALTITUDE_ASL_M - res.SITE["observation_level_asl_m"]
)
rejecting_height_above_observation_level_m = (
    TRIGGER_REJECTING_ALTITUDE_ASL_M - res.SITE["observation_level_asl_m"]
)


def get_trigger_threshold(zenith_rad, trigger, nominal_threshold_pe):
    zenith_factor = np.interp(
        x=zenith_rad,
        xp=trigger["threshold_factor_vs_pointing_zenith"]["zenith_rad"],
        fp=trigger["threshold_factor_vs_pointing_zenith"]["factor"],
    )
    assert np.all(zenith_factor >= 1.0)
    return np.round(zenith_factor * nominal_threshold_pe).astype(int)


def assign_accepting_and_rejecting_focus_based_on_pointing_zenith(
    pointing_zenith_rad,
    accepting_height_above_observation_level_m,
    rejecting_height_above_observation_level_m,
    trigger_foci_bin_edges_m,
):
    cos_pointing_zd_rad = np.cos(pointing_zenith_rad)

    accepting_depth_m = (
        accepting_height_above_observation_level_m / cos_pointing_zd_rad
    )
    rejecting_depth_m = (
        rejecting_height_above_observation_level_m / cos_pointing_zd_rad
    )

    accepting_focus = np.digitize(
        x=accepting_depth_m, bins=trigger_foci_bin_edges_m
    )
    rejecting_focus = np.digitize(
        x=rejecting_depth_m, bins=trigger_foci_bin_edges_m
    )

    return accepting_focus, rejecting_focus


def copy_focus_response_into_matrix(trigger_table):
    num_events = trigger_table.shape[0]
    num_foci = 0
    for key in trigger_table.dtype.names:
        if "focus_" in key and "_response_pe" in key:
            num_foci += 1

    focus_response_pe = np.zeros(shape=(num_events, num_foci), dtype=int)
    for f in range(num_foci):
        focus_response_pe[:, f] = trigger_table[f"focus_{f:02d}_response_pe"]

    return focus_response_pe


def find_accepting_and_rejecting_response(
    accepting_focus,
    rejecting_focus,
    focus_response_pe,
):
    num_events = focus_response_pe.shape[0]
    assert accepting_focus.shape[0] == num_events
    assert rejecting_focus.shape[0] == num_events
    num_foci = focus_response_pe.shape[1]

    accepting_response_pe = np.zeros(shape=num_events, dtype=int)
    rejecting_response_pe = np.zeros(shape=num_events, dtype=int)

    for focus in range(num_foci):
        accepting_mask = accepting_focus == focus
        accepting_response_pe[accepting_mask] = focus_response_pe[
            accepting_mask, focus
        ]

        rejecting_mask = rejecting_focus == focus
        rejecting_response_pe[rejecting_mask] = focus_response_pe[
            rejecting_mask, focus
        ]

    return accepting_response_pe, rejecting_response_pe


fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
x_zd_rad = np.linspace(0, np.deg2rad(45), 1337)
ax.plot(
    np.rad2deg(x_zd_rad),
    get_trigger_threshold(
        zenith_rad=x_zd_rad,
        trigger=trigger,
        nominal_threshold_pe=trigger["threshold_pe"],
    ),
    color="black",
)
ax.set_xlabel(r"zenith / (1$^{\circ}$)")
ax.set_ylabel("threshold / p.e.")
ax.set_xlim([0, 50])
fig.savefig(
    opj(
        res.paths["out_dir"],
        "threshold_vs_zenith.jpg",
    )
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
            [
                event_table["trigger"]["uid"],
                event_table["instrument_pointing"]["uid"],
            ]
        ),
    )

    num_events = event_table["trigger"].shape[0]

    (accepting_focus, rejecting_focus) = (
        assign_accepting_and_rejecting_focus_based_on_pointing_zenith(
            pointing_zenith_rad=event_table["instrument_pointing"][
                "zenith_rad"
            ],
            accepting_height_above_observation_level_m=accepting_height_above_observation_level_m,
            rejecting_height_above_observation_level_m=rejecting_height_above_observation_level_m,
            trigger_foci_bin_edges_m=trigger["foci_bin"]["edges"],
        )
    )
    assert accepting_focus.shape[0] == num_events
    assert rejecting_focus.shape[0] == num_events

    focus_response_pe = copy_focus_response_into_matrix(
        trigger_table=event_table["trigger"]
    )
    assert focus_response_pe.shape[0] == num_events
    assert focus_response_pe.shape[1] == trigger["foci_bin"]["num"]

    (accepting_response_pe, rejecting_response_pe) = (
        find_accepting_and_rejecting_response(
            accepting_focus=accepting_focus,
            rejecting_focus=rejecting_focus,
            focus_response_pe=focus_response_pe,
        )
    )
    assert accepting_response_pe.shape[0] == num_events
    assert rejecting_response_pe.shape[0] == num_events

    threshold_accepting_over_rejecting = np.interp(
        x=accepting_response_pe,
        xp=trigger["modus"]["accepting"]["response_pe"],
        fp=trigger["modus"]["accepting"]["threshold_accepting_over_rejecting"],
        left=None,
        right=None,
        period=None,
    )
    assert threshold_accepting_over_rejecting.shape[0] == num_events

    accepting_over_rejecting = accepting_response_pe / rejecting_response_pe
    is_ratio_over_threshold = (
        accepting_over_rejecting >= threshold_accepting_over_rejecting
    )

    # export
    # ------
    pk_ratescan_dir = opj(pk_dir, "ratescan")
    os.makedirs(pk_ratescan_dir, exist_ok=True)

    for irs in range(len(trigger["ratescan_thresholds_pe"])):
        nominal_threshold_pe = trigger["ratescan_thresholds_pe"][irs]

        zenith_corrected_threshold_pe = get_trigger_threshold(
            zenith_rad=event_table["instrument_pointing"]["zenith_rad"],
            trigger=trigger,
            nominal_threshold_pe=nominal_threshold_pe,
        )

        is_size_over_threshold = (
            accepting_response_pe >= zenith_corrected_threshold_pe
        )
        is_pasttrigger = np.logical_and(
            is_size_over_threshold, is_ratio_over_threshold
        )

        filename = f"{trigger['ratescan_thresholds_pe'][irs]:d}pe.json"
        uids_pasttrigger = event_table["trigger"]["uid"][is_pasttrigger]

        json_utils.write(
            opj(pk_ratescan_dir, filename),
            {"uid": uids_pasttrigger},
        )

        if nominal_threshold_pe == trigger["threshold_pe"]:
            json_utils.write(
                opj(pk_dir, "uid.json"),
                {"uid": uids_pasttrigger},
            )

res.stop()
