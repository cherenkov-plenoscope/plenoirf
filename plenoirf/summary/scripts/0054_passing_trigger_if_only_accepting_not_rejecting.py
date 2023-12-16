#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
import json_utils
import numpy as np
import copy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

for sk in irf_config["config"]["sites"]:
    trigger_modus = sum_config["trigger"][sk]["modus"]
    trigger_threshold = sum_config["trigger"][sk]["threshold_pe"]

    tm = {}
    tm["accepting_focus"] = trigger_modus["accepting_focus"]
    tm["rejecting_focus"] = trigger_modus["rejecting_focus"]
    tm["accepting"] = {}
    tm["accepting"]["threshold_accepting_over_rejecting"] = np.zeros(
        len(trigger_modus["accepting"]["response_pe"])
    )
    tm["accepting"]["response_pe"] = trigger_modus["accepting"]["response_pe"]

    for pk in irf_config["config"]["particles"]:
        sk_pk_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(sk_pk_dir, exist_ok=True)

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

        idx_pasttrigger = irf.analysis.light_field_trigger_modi.make_indices(
            trigger_table=event_table["trigger"],
            threshold=trigger_threshold,
            modus=tm,
        )

        json_utils.write(
            path=os.path.join(sk_pk_dir, "idx.json"),
            out_dict=idx_pasttrigger,
        )
