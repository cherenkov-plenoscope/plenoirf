#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
import json_utils

paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)

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
TRIGGER = res.analysis["trigger"][res.site_key]

trigger_thresholds = TRIGGER["ratescan_thresholds_pe"]
num_trigger_thresholds = len(trigger_thresholds)
trigger_modus = TRIGGER["modus"]
nsb = {
    "num_exposures": 0,
    "num_triggers_vs_threshold": np.zeros(num_trigger_thresholds, dtype=int),
}
for pk in res.PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        airshower_table = arc.query(levels_and_columns={"trigger": "__all__"})

    # The true num of Cherenkov-photons in the light-field-sequence must be
    # below a critical threshold.
    uid_nsb = airshower_table["trigger"]["uid"][
        airshower_table["trigger"]["num_cherenkov_pe"]
        <= MAX_CHERENKOV_IN_NSB_PE
    ]
    nsb_table = snt.logic.cut_level_on_indices(
        level=airshower_table["trigger"],
        indices=uid_nsb,
        index_key="uid",
    )

    for tt, threshold in enumerate(trigger_thresholds):
        uid_trigger = irf.analysis.light_field_trigger_modi.make_indices(
            trigger_table=nsb_table,
            threshold=threshold,
            modus=trigger_modus,
        )
        nsb["num_exposures"] += len(uid_nsb)
        nsb["num_triggers_vs_threshold"][tt] += len(uid_trigger)


num_exposures = nsb["num_exposures"]
num_triggers_vs_threshold = nsb["num_triggers_vs_threshold"]

mean = num_triggers_vs_threshold / (num_exposures * EXPOSURE_TIME_PER_EVENT)
relative_uncertainty = irf.utils._divide_silent(
    numerator=np.sqrt(num_triggers_vs_threshold),
    denominator=num_triggers_vs_threshold,
    default=np.nan,
)

json_utils.write(
    os.path.join(paths["out_dir"], "night_sky_background_rates.json"),
    {
        "comment": (
            "Trigger rate for night-sky-background"
            "VS trigger-ratescan-thresholds"
        ),
        "trigger": TRIGGER,
        "unit": "s$^{-1}$",
        "mean": mean,
        "relative_uncertainty": relative_uncertainty,
    },
)
