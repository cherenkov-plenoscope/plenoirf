#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
import json_utils


paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)

max_relative_leakage = res.analysis["quality"]["max_relative_leakage"]
min_reconstructed_photons = res.analysis["quality"][
    "min_reconstructed_photons"
]

for pk in res.PARTICLES:
    pk_dir = os.path.join(paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    event_table = snt.read(
        path=os.path.join(
            paths["plenoirf_dir"],
            "response",
            res.instrument_key,
            res.site_key,
            pk,
            "event_table.tar",
        )
    )

    idx_pastquality = irf.analysis.cuts.cut_quality(
        feature_table=event_table["features"],
        max_relative_leakage=max_relative_leakage,
        min_reconstructed_photons=min_reconstructed_photons,
    )

    json_utils.write(os.path.join(pk_dir, "idx.json"), idx_pastquality)
