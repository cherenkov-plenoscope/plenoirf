#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

PARTICLES = irf_config["config"]["particles"]
SITES = irf_config["config"]["sites"]

MAX_CHERENKOV_IN_NSB_PE = sum_config["night_sky_background"][
    "max_num_true_cherenkov_photons"
]
TIME_SLICE_DURATION = 0.5e-9
NUM_TIME_SLICES_IN_LIGHTFIELDSEQUENCE = 100
NUM_TIME_SLICES_PER_EVENT = (
    NUM_TIME_SLICES_IN_LIGHTFIELDSEQUENCE
    - irf_config["config"]["sum_trigger"]["integration_time_slices"]
)
EXPOSURE_TIME_PER_EVENT = NUM_TIME_SLICES_PER_EVENT * TIME_SLICE_DURATION
TRIGGER = sum_config["trigger"]

nsb = {}
for sk in SITES:
    trigger_thresholds = TRIGGER[sk]["ratescan_thresholds_pe"]
    num_trigger_thresholds = len(trigger_thresholds)
    trigger_modus = TRIGGER[sk]["modus"]

    nsb[sk] = {
        "num_exposures": 0,
        "num_triggers_vs_threshold": np.zeros(
            num_trigger_thresholds, dtype=np.int
        ),
    }
    for pk in PARTICLES:
        airshower_table = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                sk,
                pk,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        # The true num of Cherenkov-photons in the light-field-sequence must be
        # below a critical threshold.
        idx_nsb = airshower_table["trigger"][spt.IDX][
            airshower_table["trigger"]["num_cherenkov_pe"]
            <= MAX_CHERENKOV_IN_NSB_PE
        ]
        nsb_table = spt.cut_level_on_indices(
            level=airshower_table["trigger"],
            indices=idx_nsb,
        )

        for tt, threshold in enumerate(trigger_thresholds):
            idx_trigger = irf.analysis.light_field_trigger_modi.make_indices(
                trigger_table=nsb_table,
                threshold=threshold,
                modus=trigger_modus,
            )
            nsb[sk]["num_exposures"] += len(idx_nsb)
            nsb[sk]["num_triggers_vs_threshold"][tt] += len(idx_trigger)

for sk in SITES:
    num_exposures = nsb[sk]["num_exposures"]
    num_triggers_vs_threshold = nsb[sk]["num_triggers_vs_threshold"]

    mean = num_triggers_vs_threshold / (
        num_exposures * EXPOSURE_TIME_PER_EVENT
    )
    relative_uncertainty = irf.utils._divide_silent(
        numerator=np.sqrt(num_triggers_vs_threshold),
        denominator=num_triggers_vs_threshold,
        default=np.nan,
    )

    site_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(site_dir, exist_ok=True)
    json_utils.write(
        os.path.join(site_dir, "night_sky_background_rates.json"),
        {
            "comment": (
                "Trigger rate for night-sky-background"
                "VS trigger-ratescan-thresholds"
            ),
            "trigger": sum_config["trigger"],
            "unit": "s$^{-1}$",
            "mean": mean,
            "relative_uncertainty": relative_uncertainty,
        },
    )
