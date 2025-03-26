#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import json_utils


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

max_relative_leakage = res.analysis["quality"]["max_relative_leakage"]
min_reconstructed_photons = res.analysis["quality"][
    "min_reconstructed_photons"
]

for pk in res.PARTICLES:
    pk_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(levels_and_columns={"features": "__all__"})

    uids_pastquality = irf.analysis.cuts.cut_quality(
        feature_table=event_table["features"],
        max_relative_leakage=max_relative_leakage,
        min_reconstructed_photons=min_reconstructed_photons,
    )

    json_utils.write(opj(pk_dir, "uid.json"), uids_pastquality)

res.stop()
