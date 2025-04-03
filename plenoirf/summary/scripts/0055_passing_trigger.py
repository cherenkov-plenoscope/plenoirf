#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import json_utils
import numpy as np


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

zenith_bin = res.zenith_binning(key="twice")
triggerfoci_bin = res.trigger_image_object_distance_binning()

trigger_modus = res.analysis["trigger"][res.site_key]["modus"]
trigger_threshold = res.analysis["trigger"][res.site_key]["threshold_pe"]

TRIGGER_ACCEPTING_ALTITUDE_ASL_M = 19_856
TRIGGER_REJECTING_ALTITUDE_ASL_M = 13_851

accepting_height_above_observation_level_m = (
    TRIGGER_ACCEPTING_ALTITUDE_ASL_M - res.SITE["observation_level_asl_m"]
)
rejecting_height_above_observation_level_m = (
    TRIGGER_REJECTING_ALTITUDE_ASL_M - res.SITE["observation_level_asl_m"]
)


def get_index_of_closest_match(x, y):
    return int(np.argmin(np.abs(x - y)))


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

    uids_pasttrigger = []
    for zd in range(zenith_bin["num"]):
        zenith_start_rad = zenith_bin["edges"][zd]
        zenith_stop_rad = zenith_bin["edges"][zd + 1]
        zenith_mask = np.logical_and(
            event_table["instrument_pointing"]["zenith_rad"]
            >= zenith_start_rad,
            event_table["instrument_pointing"]["zenith_rad"] < zenith_stop_rad,
        )
        zenith_rad = zenith_bin["centers"][zd]

        accepting_depth_m = (
            accepting_height_above_observation_level_m / np.cos(zenith_rad)
        )
        rejecting_depth_m = (
            rejecting_height_above_observation_level_m / np.cos(zenith_rad)
        )

        trigger_modus["accepting_focus"] = get_index_of_closest_match(
            x=triggerfoci_bin["centers"],
            y=accepting_depth_m,
        )
        trigger_modus["rejecting_focus"] = get_index_of_closest_match(
            x=triggerfoci_bin["centers"],
            y=rejecting_depth_m,
        )

        msg = f"Zenith [{np.rad2deg(zenith_start_rad):4.1f}, "
        msg += f"{np.rad2deg(zenith_rad):4.1f}, "
        msg += f"{np.rad2deg(zenith_stop_rad):4.1f}]deg, "
        msg += f"accept. {1e-3*accepting_depth_m:.1f}km "
        msg += f"[{trigger_modus['accepting_focus']:d}], "
        msg += f"reject. {1e-3*rejecting_depth_m:.1f}km, "
        msg += f"[{trigger_modus['rejecting_focus']:d}]."
        print(msg)

        _uids_pasttrigger = irf.analysis.light_field_trigger_modi.make_indices(
            trigger_table=event_table["trigger"][zenith_mask],
            threshold=trigger_threshold,
            modus=trigger_modus,
        )
        uids_pasttrigger += list(_uids_pasttrigger)

    json_utils.write(opj(pk_dir, "uid.json"), uids_pasttrigger)

res.stop()
