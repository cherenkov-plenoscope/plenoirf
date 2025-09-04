#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

MAX_CHERENKOV_IN_NSB_PE = res.analysis["night_sky_background"][
    "max_num_true_cherenkov_photons"
]
TIME_SLICE_DURATION = 0.5e-9
NUM_TIME_SLICES_IN_LIGHTFIELDSEQUENCE = 100
NUM_TIME_SLICES_PER_EVENT = (
    NUM_TIME_SLICES_IN_LIGHTFIELDSEQUENCE
    - res.config["sum_trigger"]["integration_time_slices"]
)
EXPOSURE_TIME_PER_EVENT = NUM_TIME_SLICES_PER_EVENT * TIME_SLICE_DURATION

passing_trigger = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)
trigger = res.trigger

nsb = {
    "num_exposures": 0,
    "num_triggers_vs_threshold": np.zeros(
        len(trigger["ratescan_thresholds_pe"]),
        dtype=int,
    ),
}
num_all_events = 0
for pk in res.PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        airshower_table = arc.query(
            levels_and_columns={"trigger": ["uid", "num_cherenkov_pe"]}
        )

    # The true num of Cherenkov-photons in the light-field-sequence must be
    # below a critical threshold.
    uid_nsb = airshower_table["trigger"]["uid"][
        airshower_table["trigger"]["num_cherenkov_pe"]
        <= MAX_CHERENKOV_IN_NSB_PE
    ]
    uid_nsb = set(uid_nsb)

    num_all_events += len(airshower_table["trigger"]["uid"])
    nsb["num_exposures"] += len(uid_nsb)

    for tt, threshold in enumerate(trigger["ratescan_thresholds_pe"]):
        uid_trigger = set.intersection(
            set(passing_trigger[pk]["ratescan"][f"{threshold:d}pe"]["uid"]),
            set(uid_nsb),
        )
        nsb["num_triggers_vs_threshold"][tt] += len(uid_trigger)


num_exposures = nsb["num_exposures"]
num_triggers_vs_threshold = nsb["num_triggers_vs_threshold"]

mean_rate = num_triggers_vs_threshold / (
    num_exposures * EXPOSURE_TIME_PER_EVENT
)

relative_uncertainty = irf.utils._divide_silent(
    numerator=np.sqrt(num_triggers_vs_threshold),
    denominator=num_triggers_vs_threshold,
    default=np.nan,
)

json_utils.write(
    opj(res.paths["out_dir"], "night_sky_background_rates.json"),
    {
        "comment": (
            "Trigger rate for night-sky-background"
            "VS trigger-ratescan-thresholds"
        ),
        "unit": "s$^{-1}$",
        "mean": mean_rate,
        "relative_uncertainty": relative_uncertainty,
    },
)

res.stop()
