#!/usr/bin/python
import sys
import plenoirf as irf
import os
from os.path import join as opj
import json_utils
import sparse_numeric_table as snt
import sebastians_matplotlib_addons as sebplt

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt)
energy_key = "trigger_acceptance_onregion"
energy_bin = res.energy_binning(energy_key)
zenith_key = "once"
zenith_bin = res.zenith_binning(key=zenith_key)

for pk in res.PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        table = arc.query(
            levels_and_columns={
                "primary": ["uid", "energy_GeV"],
                "instrument_pointing": ["uid", "zenith_rad"],
            },
        )
        table = snt.logic.sort_table_on_common_indices(
            table=table, common_indices=table["primary"]["uid"], inplace=True
        )

    thrown = np.histogram2d(
        x=table["primary"]["energy_GeV"],
        y=table["instrument_pointing"]["zenith_rad"],
        bins=(energy_bin["edges"], zenith_bin["edges"]),
    )[0]

    os.makedirs(opj(res.paths["out_dir"], pk), exist_ok=True)
    json_utils.write(
        opj(res.paths["out_dir"], pk, "num_thrown_energy_vs_zenith.json"),
        {
            "counts": thrown,
            "energy_key": energy_key,
            "zenith_key": zenith_key,
        },
    )

res.stop()
