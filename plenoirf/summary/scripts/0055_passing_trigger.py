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

# zenith_bin = res.zenith_binning(key="twice")
triggerfoci_bin = res.trigger_image_object_distance_binning()

trigger_modus = res.analysis["trigger"][res.site_key]["modus"]
trigger_threshold_vs_pointing_zenith = res.analysis["trigger"][res.site_key][
    "threshold_vs_pointing_zenith"
]

TRIGGER_ACCEPTING_ALTITUDE_ASL_M = 19_856
TRIGGER_REJECTING_ALTITUDE_ASL_M = 13_851

accepting_height_above_observation_level_m = (
    TRIGGER_ACCEPTING_ALTITUDE_ASL_M - res.SITE["observation_level_asl_m"]
)
rejecting_height_above_observation_level_m = (
    TRIGGER_REJECTING_ALTITUDE_ASL_M - res.SITE["observation_level_asl_m"]
)


def trigger_threshold(zenith_rad):
    return np.interp(
        x=zenith_rad,
        xp=trigger_threshold_vs_pointing_zenith["zenith_rad"],
        fp=trigger_threshold_vs_pointing_zenith["threshold_pe"],
    )


fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
x_zd_rad = np.linspace(0, np.deg2rad(45), 1337)
ax.plot(
    np.rad2deg(x_zd_rad),
    trigger_threshold(zenith_rad=x_zd_rad),
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

    num_events = len(event_table["trigger"])

    uids_pasttrigger = []
    for i in range(num_events):
        if np.mod(i, 1000) == 0:
            print(i, "of", num_events)

        pointing_zd_rad = event_table["instrument_pointing"][i]["zenith_rad"]
        cos_pointing_zd_rad = np.cos(pointing_zd_rad)

        accepting_depth_m = (
            accepting_height_above_observation_level_m / cos_pointing_zd_rad
        )
        rejecting_depth_m = (
            rejecting_height_above_observation_level_m / cos_pointing_zd_rad
        )
        trigger_modus["accepting_focus"] = (
            irf.utils.get_index_of_closest_match(
                x=triggerfoci_bin["centers"],
                y=accepting_depth_m,
            )
        )
        trigger_modus["rejecting_focus"] = (
            irf.utils.get_index_of_closest_match(
                x=triggerfoci_bin["centers"],
                y=rejecting_depth_m,
            )
        )

        _uids_pasttrigger = irf.analysis.light_field_trigger_modi.make_indices(
            trigger_table=event_table["trigger"][i],
            threshold=trigger_threshold(zenith_rad=pointing_zd_rad),
            modus=trigger_modus,
        )
        uids_pasttrigger += list(_uids_pasttrigger)

    json_utils.write(opj(pk_dir, "uid.json"), uids_pasttrigger)

res.stop()
