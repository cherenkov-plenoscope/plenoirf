#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import json_utils
import propagate_uncertainties as pu

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

zenith_bin = res.zenith_binning(key="once")
zenith_assignment = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0019_zenith_bin_assignment")
)

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
EXPOSURE_TIME_PER_EVENT_AU = 0.0

passing_trigger = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)
trigger = res.trigger

num_thresholds = len(trigger["ratescan_thresholds_pe"])

trigger_ratescan = {
    "only_accepting_not_rejecting": lambda particle_key, threshold_pe: passing_trigger[
        particle_key
    ][
        "only_accepting_not_rejecting"
    ][
        "ratescan"
    ][
        f"{threshold_pe:d}pe"
    ][
        "uid"
    ],
    "applying_rejection_refocussing": lambda particle_key, threshold_pe: passing_trigger[
        particle_key
    ][
        "ratescan"
    ][
        f"{threshold_pe:d}pe"
    ][
        "uid"
    ],
}


for trigger_modus in trigger_ratescan:

    nsb = {}
    for zd in range(zenith_bin["num"]):
        zk = f"zd{zd:d}"
        nsb[zk] = {
            "trigger_modus": trigger_modus,
            "num_exposures": 0,
            "num_exposures_au": 0.0,
            "num_trigger": np.zeros(num_thresholds, dtype=int),
            "num_trigger_au": np.zeros(num_thresholds, dtype=float),
            "rate": np.nan * np.ones(shape=num_thresholds),
            "rate_au": np.nan * np.ones(shape=num_thresholds),
            "exposure_time_s": np.nan,
            "exposure_time_s_au": np.nan,
        }

    for pk in res.PARTICLES:
        with res.open_event_table(particle_key=pk) as arc:
            trigger_table = arc.query(
                levels_and_columns={"trigger": ["uid", "num_cherenkov_pe"]}
            )["trigger"]

        # The true num of Cherenkov photons in the light field sequence must be
        # below a critical threshold.
        mask_nsb = trigger_table["num_cherenkov_pe"] <= MAX_CHERENKOV_IN_NSB_PE
        uid_nsb = trigger_table["uid"][mask_nsb]
        uid_nsb = set(uid_nsb)

        for zd in range(zenith_bin["num"]):
            zk = f"zd{zd:d}"

            uid_nsb_zd = set.intersection(uid_nsb, zenith_assignment[zk][pk])

            nsb[zk]["num_exposures"] += len(uid_nsb_zd)

            for tt in range(num_thresholds):
                print(trigger_modus, pk, zd, tt)

                threshold_pe = trigger["ratescan_thresholds_pe"][tt]
                uid_trigger_zd = set.intersection(
                    set(
                        trigger_ratescan[trigger_modus](
                            particle_key=pk, threshold_pe=threshold_pe
                        )
                    ),
                    set(uid_nsb_zd),
                )
                nsb[zk]["num_trigger"][tt] += len(uid_trigger_zd)

    for zd in range(zenith_bin["num"]):
        zk = f"zd{zd:d}"

        nsb[zk]["num_exposures_au"] = np.sqrt(nsb[zk]["num_exposures"])
        for t in range(num_thresholds):
            nsb[zk]["num_trigger_au"][t] = np.sqrt(nsb[zk]["num_trigger"][t])

        nsb[zk]["exposure_time_s"], nsb[zk]["exposure_time_s_au"] = (
            pu.multiply(
                x=EXPOSURE_TIME_PER_EVENT,
                x_au=EXPOSURE_TIME_PER_EVENT_AU,
                y=nsb[zk]["num_exposures"],
                y_au=nsb[zk]["num_exposures_au"],
            )
        )

        for t in range(num_thresholds):
            nsb[zk]["rate"][t], nsb[zk]["rate_au"][t] = pu.divide(
                x=nsb[zk]["num_trigger"][t],
                x_au=nsb[zk]["num_trigger_au"][t],
                y=nsb[zk]["exposure_time_s"],
                y_au=nsb[zk]["exposure_time_s_au"],
            )

    json_utils.write(
        opj(
            res.paths["out_dir"],
            f"night_sky_background_rates_{trigger_modus:s}.json",
        ),
        nsb,
    )

res.stop()
