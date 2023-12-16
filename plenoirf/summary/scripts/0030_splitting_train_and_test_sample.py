#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
import sklearn
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
summary_random_seed = sum_config["random_seed"]
test_size = sum_config["train_and_test"]["test_size"]


for sk in SITES:
    for pk in PARTICLES:
        site_dir = os.path.join(pa["out_dir"], sk)
        os.makedirs(site_dir, exist_ok=True)

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                sk,
                pk,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        train_idxs, test_idxs = sklearn.model_selection.train_test_split(
            event_table["primary"][spt.IDX],
            test_size=test_size,
            random_state=summary_random_seed,
        )

        json_utils.write(
            os.path.join(site_dir, pk + ".json"),
            {
                "comment": (
                    "Split into train-sample and test-sample to "
                    "validate machine-learning."
                ),
                "train": train_idxs,
                "test": test_idxs,
            },
        )
