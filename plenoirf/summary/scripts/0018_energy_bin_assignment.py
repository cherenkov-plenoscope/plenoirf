#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
import json_utils
import numpy as np

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()


for ek in ["trigger_acceptance_onregion"]:
    energy_bin = res.energy_binning(key="trigger_acceptance_onregion")

    for pk in res.PARTICLES:
        ek_pk_dir = os.path.join(res.paths["out_dir"], ek, pk)
        os.makedirs(ek_pk_dir, exist_ok=True)

        with res.open_event_table(particle_key=pk) as arc:
            event_table = arc.query(
                levels_and_columns={
                    "primary": ["uid", "energy_GeV"],
                }
            )

        for ebin in range(energy_bin["num"]):
            energy_start_GeV = energy_bin["edges"][ebin]
            energy_stop_GeV = energy_bin["edges"][ebin + 1]

            mask = np.logical_and(
                event_table["primary"]["energy_GeV"] >= energy_start_GeV,
                event_table["primary"]["energy_GeV"] < energy_stop_GeV,
            )
            uid = event_table["primary"]["uid"][mask]

            json_utils.write(os.path.join(ek_pk_dir, f"{ebin:d}.json"), uid)

res.stop()
