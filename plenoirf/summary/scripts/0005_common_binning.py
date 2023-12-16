#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import json_utils
import binning_utils
import solid_angle_utils


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

os.makedirs(pa["out_dir"], exist_ok=True)
irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

PARTICLES = irf_config["config"]["particles"]

# energy
# ------
energy = {}
for scenario_key in sum_config["energy_binning"]["fine"]:
    edges, num_bins = irf.utils.power10space_bin_edges(
        binning=sum_config["energy_binning"],
        fine=sum_config["energy_binning"]["fine"][scenario_key],
    )

    assert len(edges) >= 2
    assert np.all(np.gradient(edges) > 0.0)

    energy[scenario_key] = {
        "key": scenario_key,
        "edges": edges,
        "num_bins": num_bins,
        "centers": binning_utils.centers(edges),
        "widths": binning_utils.widths(edges),
        "start": edges[0],
        "stop": edges[-1],
        "limits": [edges[0], edges[-1]],
        "unit": "GeV",
    }

json_utils.write(os.path.join(pa["out_dir"], "energy.json"), energy)

# max scatter angle
# -----------------
NUM_MAX_SCATTER_ANGLES = 20

msa = {}
for pk in PARTICLES:
    max_scatter_angle_deg = PARTICLES[pk]["max_scatter_angle_deg"]
    max_scatter_angle_rad = np.deg2rad(max_scatter_angle_deg)
    max_scatter_solid_angle_sr = solid_angle_utils.cone.solid_angle(
        half_angle_rad=max_scatter_angle_rad
    )
    _sc = {}
    _sc["start"] = 0.0
    _sc["stop"] = max_scatter_solid_angle_sr
    _sc["limits"] = [_sc["start"], _sc["stop"]]
    _sc["unit"] = "sr"
    _sc["num_bins"] = NUM_MAX_SCATTER_ANGLES

    solid_angle_step_sr = max_scatter_solid_angle_sr / NUM_MAX_SCATTER_ANGLES
    _sc["edges"] = np.zeros(NUM_MAX_SCATTER_ANGLES + 1)
    for i in range(NUM_MAX_SCATTER_ANGLES + 1):
        _sc["edges"][i] = solid_angle_step_sr * i
    _sc["centers"] = binning_utils.centers(_sc["edges"])
    _sc["widths"] = binning_utils.widths(_sc["edges"])
    msa[pk] = _sc

json_utils.write(os.path.join(pa["out_dir"], "scatter.json"), msa)
