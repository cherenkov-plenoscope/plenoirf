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

for pk in res.PARTICLES:
    pk_dir = os.path.join(paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    event_table = res.read_event_table(particle_key=pk)

    event_frame = irf.reconstruction.trajectory_quality.make_rectangular_table(
        event_table=event_table,
    )

    # estimate_quality
    # ----------------

    quality = irf.reconstruction.trajectory_quality.estimate_trajectory_quality(
        event_frame=event_frame,
        quality_features=irf.reconstruction.trajectory_quality.QUALITY_FEATURES,
    )

    json_utils.write(
        os.path.join(pk_dir, "trajectory_quality.json"),
        {
            "comment": (
                "Quality of reconstructed trajectory. "
                "0 is worst, 1 is best."
            ),
            snt.IDX: event_frame[snt.IDX],
            "unit": "1",
            "quality": quality,
        },
    )

    # apply cut
    # ---------
    mask = quality >= res.analysis["quality"]["min_trajectory_quality"]
    idx_passed = event_frame[snt.IDX][mask]

    json_utils.write(os.path.join(pk_dir, "idx.json"), idx_passed)
