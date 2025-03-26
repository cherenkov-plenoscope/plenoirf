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

for pk in res.PARTICLES:
    pk_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={
                "reconstructed_trajectory": "__all__",
                "features": "__all__",
            }
        )

    event_frame = irf.reconstruction.trajectory_quality.make_rectangular_table(
        event_table=event_table,
        instrument_pointing_model=res.config["pointing"]["model"],
    )

    # estimate_quality
    # ----------------

    quality = irf.reconstruction.trajectory_quality.estimate_trajectory_quality(
        event_frame=event_frame,
        quality_features=irf.reconstruction.trajectory_quality.QUALITY_FEATURES,
    )

    json_utils.write(
        opj(pk_dir, "trajectory_quality.json"),
        {
            "comment": (
                "Quality of reconstructed trajectory. "
                "0 is worst, 1 is best."
            ),
            "uid": event_frame["uid"],
            "unit": "1",
            "quality": quality,
        },
    )

    # apply cut
    # ---------
    mask = quality >= res.analysis["quality"]["min_trajectory_quality"]
    uids_passed = event_frame["uid"][mask]

    json_utils.write(opj(pk_dir, "uid.json"), uids_passed)

res.stop()
