#!/usr/bin/python
import sys
import flux_sensitivity
import numpy as np
import json_utils
import plenoirf as irf
import binning_utils
import os
from os.path import join as opj


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

observation_times = json_utils.read(
    opj(
        res.paths["analysis_dir"],
        "0539_diffsens_observation_times",
        "observation_times.json",
    )
)["observation_times"]
num_observation_times = len(observation_times)

fls = (
    irf.other_instruments.fermi_lat.flux_sensitivity_vs_observation_time_vs_energy()
)

# conver units
# ------------
odnde = {}
assert fls["dnde"]["unit"] == "cm-2 MeV-1 ph s-1"
odnde["dVdE_per_m2_per_GeV_per_s"] = fls["dnde"]["value"] * 1e4 * 1e3
assert fls["energy_bin_edges"]["unit"] == "MeV"
odnde["energy_bin_edges_GeV"] = fls["energy_bin_edges"]["value"] * 1e-3
assert fls["observation_times"]["unit"] == "s"
odnde["observation_times_s"] = fls["observation_times"]["value"]
num_energy_bins = len(odnde["energy_bin_edges_GeV"]) - 1

# map onto our observation-times
# ------------------------------

out = {}
out["dVdE_per_m2_per_GeV_per_s"] = np.zeros(
    shape=(num_energy_bins, num_observation_times)
)
out["dVdE_per_m2_per_GeV_per_s_au"] = np.zeros(
    shape=out["dVdE_per_m2_per_GeV_per_s"].shape
)
out["observation_times_s"] = observation_times
out["energy_bin_edges_GeV"] = odnde["energy_bin_edges_GeV"]
out["comment"] = "Based on Fermi-LAT 3FGL"

for ebin in range(num_energy_bins):
    for obstix in range(num_observation_times):
        out["dVdE_per_m2_per_GeV_per_s"][ebin, obstix] = np.interp(
            fp=odnde["dVdE_per_m2_per_GeV_per_s"][:, ebin],
            xp=odnde["observation_times_s"],
            x=observation_times[obstix],
            left=float("nan"),
            right=float("nan"),
        )

json_utils.write(opj(res.paths["out_dir"], "flux_sensitivity.json"), out)

res.stop()
