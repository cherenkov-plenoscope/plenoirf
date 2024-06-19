#!/usr/bin/python
import sys
import copy
import numpy as np
import plenoirf as irf
import os
import json_utils
import magnetic_deflection

paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)

raw_cosmic_ray_fluxes = json_utils.tree.read(
    os.path.join(paths["analysis_dir"], "0010_flux_of_cosmic_rays")
)

energy_bin = json_utils.read(
    os.path.join(paths["analysis_dir"], "0005_common_binning", "energy.json")
)["interpolation"]

deflection_table = magnetic_deflection.read_deflection(
    work_dir=os.path.join(paths["plenoirf_dir"], "magnetic_deflection"),
)

SITES = res.SITES
PARTICLES = res.PARTICLES
COSMIC_RAYS = res.COSMIC_RAYS

fraction_of_flux_below_geomagnetic_cutoff = res.analysis["airshower_flux"][
    "fraction_of_flux_below_geomagnetic_cutoff"
]
relative_uncertainty_below_geomagnetic_cutoff = res.analysis["airshower_flux"][
    "relative_uncertainty_below_geomagnetic_cutoff"
]


def _rigidity_to_total_energy(rigidity_GV):
    return rigidity_GV * 1.0


# interpolate
# -----------
cosmic_ray_fluxes = {}
for pk in COSMIC_RAYS:
    cosmic_ray_fluxes[pk] = {}
    cosmic_ray_fluxes[pk]["differential_flux"] = np.interp(
        x=energy_bin["centers"],
        xp=raw_cosmic_ray_fluxes[pk]["energy"]["values"],
        fp=raw_cosmic_ray_fluxes[pk]["differential_flux"]["values"],
    )

# earth's geomagnetic cutoff
# --------------------------
shower_fluxes = {}
for sk in SITES:
    shower_fluxes[sk] = {}
    for pk in COSMIC_RAYS:
        shower_fluxes[sk][pk] = {}
        cutoff_energy = _rigidity_to_total_energy(
            rigidity_GV=irf_config["config"]["sites"][sk][
                "geomagnetic_cutoff_rigidity_GV"
            ]
        )

        shower_fluxes[sk][pk]["differential_flux"] = np.zeros(
            energy_bin["num_bins"]
        )
        shower_fluxes[sk][pk]["differential_flux_au"] = np.zeros(
            energy_bin["num_bins"]
        )

        for ebin in range(energy_bin["num_bins"]):
            if energy_bin["centers"][ebin] < cutoff_energy:
                shower_fluxes[sk][pk]["differential_flux"][ebin] = (
                    cosmic_ray_fluxes[pk]["differential_flux"][ebin]
                    * fraction_of_flux_below_geomagnetic_cutoff
                )
                shower_fluxes[sk][pk]["differential_flux_au"][ebin] = (
                    shower_fluxes[sk][pk]["differential_flux"][ebin]
                    * relative_uncertainty_below_geomagnetic_cutoff
                )
            else:
                shower_fluxes[sk][pk]["differential_flux"][
                    ebin
                ] = cosmic_ray_fluxes[pk]["differential_flux"][ebin]
                shower_fluxes[sk][pk]["differential_flux_au"][ebin] = 0.0


# zenith compensation
# -------------------
air_shower_fluxes_zc = {}
for sk in SITES:
    air_shower_fluxes_zc[sk] = {}
    for pk in COSMIC_RAYS:
        air_shower_fluxes_zc[sk][pk] = {}
        primary_zenith_deg = np.interp(
            x=energy_bin["centers"],
            xp=deflection_table[sk][pk]["particle_energy_GeV"],
            fp=deflection_table[sk][pk]["particle_zenith_deg"],
        )
        scaling = np.cos(np.deg2rad(primary_zenith_deg))
        zc_flux = scaling * shower_fluxes[sk][pk]["differential_flux"]
        air_shower_fluxes_zc[sk][pk]["differential_flux"] = zc_flux

        zc_flux_au = scaling * shower_fluxes[sk][pk]["differential_flux_au"]
        air_shower_fluxes_zc[sk][pk]["differential_flux_au"] = zc_flux_au

# export
# ------
for sk in SITES:
    for pk in COSMIC_RAYS:
        sk_pk_dir = os.path.join(paths["out_dir"], sk, pk)
        os.makedirs(sk_pk_dir, exist_ok=True)
        json_utils.write(
            os.path.join(sk_pk_dir, "differential_flux.json"),
            {
                "comment": (
                    "The flux of air-showers seen by/ relevant for the "
                    "instrument. Respects geomagnetic cutoff "
                    "and zenith-compensation when primary is "
                    "deflected in earth's magnetic-field."
                ),
                "values": air_shower_fluxes_zc[sk][pk]["differential_flux"],
                "absolute_uncertainty": air_shower_fluxes_zc[sk][pk][
                    "differential_flux_au"
                ],
                "unit": raw_cosmic_ray_fluxes[pk]["differential_flux"]["unit"],
            },
        )
