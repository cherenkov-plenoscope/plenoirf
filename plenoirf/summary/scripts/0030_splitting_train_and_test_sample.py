#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
import sklearn
import json_utils


paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)

for pk in res.PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(levels_and_columns={"primary": ["uid"]})

    train_uids, test_uids = sklearn.model_selection.train_test_split(
        event_table["primary"]["uid"],
        test_size=res.analysis["train_and_test"]["test_size"],
        random_state=res.analysis["random_seed"],
    )

    json_utils.write(
        os.path.join(paths["out_dir"], pk + ".json"),
        {
            "comment": (
                "Split into train-sample and test-sample to "
                "validate machine-learning."
            ),
            "train": train_uids,
            "test": test_uids,
        },
    )
