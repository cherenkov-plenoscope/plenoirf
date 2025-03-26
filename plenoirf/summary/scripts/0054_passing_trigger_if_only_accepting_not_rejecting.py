#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import json_utils
import numpy as np


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

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
    pk_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(levels_and_columns={"trigger": "__all__"})

    uids_pasttrigger = irf.analysis.light_field_trigger_modi.make_indices(
        trigger_table=event_table["trigger"],
        threshold=trigger_threshold,
        modus=tm,
    )

    json_utils.write(opj(pk_dir, "uid.json"), uids_pasttrigger)

res.stop()
