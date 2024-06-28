#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
import json_utils
import numpy as np


paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)

trigger_modus = res.analysis["trigger"][res.site_key]["modus"]
trigger_threshold = res.analysis["trigger"][res.site_key]["threshold_pe"]

tm = {}
tm["accepting_focus"] = trigger_modus["accepting_focus"]
tm["rejecting_focus"] = trigger_modus["rejecting_focus"]
tm["accepting"] = {}
tm["accepting"]["threshold_accepting_over_rejecting"] = np.zeros(
    len(trigger_modus["accepting"]["response_pe"])
)
tm["accepting"]["response_pe"] = trigger_modus["accepting"]["response_pe"]

for pk in res.PARTICLES:
    pk_dir = os.path.join(paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.read_table(levels_and_columns={"trigger": "__all__"})

    idx_pasttrigger = irf.analysis.light_field_trigger_modi.make_indices(
        trigger_table=event_table["trigger"],
        threshold=trigger_threshold,
        modus=tm,
    )

    json_utils.write(os.path.join(pk_dir, "idx.json"), idx_pasttrigger)
