#!/usr/bin/python
import sys
import plenoirf as irf
import os
import json_utils
import cosmic_fluxes

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

STOP_ENERGY = 1e4

cosmic_rays = {
    "proton": {
        "original": cosmic_fluxes.proton_aguilar2015precision(),
        "extrapolation": {
            "stop_energy": 1e4,
            "spectral_index": -2.8,
            "num_points": 10,
        },
    },
    "helium": {
        "original": cosmic_fluxes.helium_patrignani2017helium(),
        "extrapolation": None,
    },
    "electron": {
        "original": cosmic_fluxes.electron_positron_aguilar2014precision(),
        "extrapolation": {
            "stop_energy": 1e4,
            "spectral_index": -3.2,
            "num_points": 10,
        },
    },
}

for ck in cosmic_rays:
    if cosmic_rays[ck]["extrapolation"]:
        out = cosmic_fluxes.extrapolate_with_power_law(
            original=cosmic_rays[ck]["original"],
            stop_energy_GeV=cosmic_rays[ck]["extrapolation"]["stop_energy"],
            spectral_index=cosmic_rays[ck]["extrapolation"]["spectral_index"],
            num_points=cosmic_rays[ck]["extrapolation"]["num_points"],
        )
    else:
        out = cosmic_rays[ck]["original"]

    assert out["energy"]["values"][-1] >= STOP_ENERGY

    with open(os.path.join(pa["out_dir"], ck + ".json"), "wt") as fout:
        fout.write(json_utils.dumps(out, indent=4))
