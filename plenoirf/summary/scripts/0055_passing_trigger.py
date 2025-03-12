#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
import json_utils


paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)


trigger_modus = res.analysis["trigger"][res.site_key]["modus"]
trigger_threshold = res.analysis["trigger"][res.site_key]["threshold_pe"]

for pk in res.PARTICLES:
    pk_dir = os.path.join(paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.read_table(levels_and_columns={"trigger": "__all__"})

    uids_pasttrigger = irf.analysis.light_field_trigger_modi.make_indices(
        trigger_table=event_table["trigger"],
        threshold=trigger_threshold,
        modus=trigger_modus,
    )

    json_utils.write(os.path.join(pk_dir, "uid.json"), uids_pasttrigger)
