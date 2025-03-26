#!/usr/bin/python
import sys
import plenoirf as irf
import os
import json_utils
import cosmic_fluxes

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

energy_bin = res.energy_binning(key="interpolation")

fermi_3fgl = cosmic_fluxes.fermi_3fgl_catalog()

json_utils.write(
    os.path.join(res.paths["out_dir"], "fermi_3fgl_catalog.json"), fermi_3fgl
)

(
    differential_flux_per_m2_per_s_per_GeV,
    name,
) = irf.summary.make_gamma_ray_reference_flux(
    fermi_3fgl=fermi_3fgl,
    gamma_ray_reference_source=res.analysis["gamma_ray_reference_source"],
    energy_supports_GeV=energy_bin["centers"],
)

json_utils.write(
    os.path.join(res.paths["out_dir"], "reference_source.json"),
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

res.stop()
