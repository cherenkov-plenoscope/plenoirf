#!/usr/bin/python
import sys
import plenoirf as irf
import os
from os.path import join as opj
import json_utils
import cosmic_fluxes

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

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

    json_utils.write(opj(res.paths["out_dir"], ck + ".json"), out, indent=4)

res.stop()
