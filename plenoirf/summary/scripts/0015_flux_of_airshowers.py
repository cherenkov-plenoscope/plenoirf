#!/usr/bin/python
import sys
import copy
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

raw_cosmic_ray_fluxes = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0010_flux_of_cosmic_rays")
)

energy_bin = res.energy_binning(key="interpolation")

fraction_of_flux_below_geomagnetic_cutoff = res.analysis["airshower_flux"][
    "fraction_of_flux_below_geomagnetic_cutoff"
]
relative_uncertainty_below_geomagnetic_cutoff = res.analysis["airshower_flux"][
    "relative_uncertainty_below_geomagnetic_cutoff"
]


def _rigidity_to_total_energy(rigidity_GV, electric_charge_qe):
    return rigidity_GV * np.abs(electric_charge_qe)


# interpolate
# -----------
cosmic_ray_fluxes = {}
for pk in res.COSMIC_RAYS:
    cosmic_ray_fluxes[pk] = {}
    cosmic_ray_fluxes[pk]["differential_flux"] = np.interp(
        x=energy_bin["centers"],
        xp=raw_cosmic_ray_fluxes[pk]["energy"]["values"],
        fp=raw_cosmic_ray_fluxes[pk]["differential_flux"]["values"],
    )

# earth's geomagnetic cutoff
# --------------------------
shower_fluxes = {}
for pk in res.COSMIC_RAYS:
    shower_fluxes[pk] = {}
    cutoff_energy = _rigidity_to_total_energy(
        rigidity_GV=res.SITE["geomagnetic_cutoff_rigidity_GV"],
        electric_charge_qe=res.COSMIC_RAYS[pk]["electric_charge_qe"],
    )

    shower_fluxes[pk]["differential_flux"] = np.zeros(energy_bin["num"])
    shower_fluxes[pk]["differential_flux_au"] = np.zeros(energy_bin["num"])

    for ebin in range(energy_bin["num"]):
        if energy_bin["centers"][ebin] < cutoff_energy:
            shower_fluxes[pk]["differential_flux"][ebin] = (
                cosmic_ray_fluxes[pk]["differential_flux"][ebin]
                * fraction_of_flux_below_geomagnetic_cutoff
            )
            shower_fluxes[pk]["differential_flux_au"][ebin] = (
                shower_fluxes[pk]["differential_flux"][ebin]
                * relative_uncertainty_below_geomagnetic_cutoff
            )
        else:
            shower_fluxes[pk]["differential_flux"][ebin] = cosmic_ray_fluxes[
                pk
            ]["differential_flux"][ebin]
            shower_fluxes[pk]["differential_flux_au"][ebin] = 0.0

# export
# ------
for pk in res.COSMIC_RAYS:
    pk_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)
    json_utils.write(
        opj(pk_dir, "differential_flux.json"),
        {
            "comment": (
                "The flux of air-showers seen by / relevant for the "
                "instrument. The geomagnetic cutoff for the specific site "
                "is already applied."
            ),
            "values": shower_fluxes[pk]["differential_flux"],
            "absolute_uncertainty": shower_fluxes[pk]["differential_flux_au"],
            "unit": raw_cosmic_ray_fluxes[pk]["differential_flux"]["unit"],
        },
    )

res.stop()
