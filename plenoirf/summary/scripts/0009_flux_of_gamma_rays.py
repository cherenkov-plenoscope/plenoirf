#!/usr/bin/python
import sys
import plenoirf as irf
import os
import json_utils
import cosmic_fluxes

paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)

energy_bin = res.energy_binning(key="interpolation")

# load catalog
# ------------
fermi_3fgl = cosmic_fluxes.fermi_3fgl_catalog()

# export catalog locally
# ----------------------
json_utils.write(
    os.path.join(paths["out_dir"], "fermi_3fgl_catalog.json"), fermi_3fgl
)

# make reference source
# ---------------------
(
    differential_flux_per_m2_per_s_per_GeV,
    name,
) = irf.summary.make_gamma_ray_reference_flux(
    fermi_3fgl=fermi_3fgl,
    gamma_ray_reference_source=res.analysis["gamma_ray_reference_source"],
    energy_supports_GeV=energy_bin["centers"],
)

json_utils.write(
    os.path.join(os.path.join(paths["out_dir"], "reference_source.json")),
    {
        "name": name,
        "differential_flux": {
            "values": differential_flux_per_m2_per_s_per_GeV,
            "unit": "m$^{-2}$ s$^{-1}$ (GeV)$^{-1}$",
        },
        "energy": {
            "values": energy_bin["centers"],
            "unit": "GeV",
        },
        "energy_implicit": {"fine": "interpolation", "supports": "centers"},
    },
)
