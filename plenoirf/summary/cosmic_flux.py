import numpy as np
import cosmic_fluxes
import os


def make_gamma_ray_reference_flux(
    fermi_3fgl,
    gamma_ray_reference_source,
    energy_supports_GeV,
):
    _grrs = gamma_ray_reference_source
    if _grrs["type"] == "3fgl":
        for _source in fermi_3fgl:
            if _source["source_name"] == _grrs["name_3fgl"]:
                _reference_gamma_source = _source
        gamma_dF_per_m2_per_s_per_GeV = cosmic_fluxes.flux_of_fermi_source(
            fermi_source=_reference_gamma_source, energy=energy_supports_GeV
        )
        source_name = _grrs["name_3fgl"]

        return gamma_dF_per_m2_per_s_per_GeV, source_name

    elif _grrs["type"] == "generic_power_law":
        _gpl = _grrs["generic_power_law"]
        gamma_dF_per_m2_per_s_per_GeV = cosmic_fluxes._power_law(
            energy=energy_supports_GeV,
            flux_density=_gpl["flux_density_per_m2_per_s_per_GeV"],
            spectral_index=_gpl["spectral_index"],
            pivot_energy=_gpl["pivot_energy_GeV"],
        )
        source_name = "".join(
            [
                "$F = F_0 \\left( \\frac{E}{E_0}\\right) ^{\\gamma}$, ",
                "$F_0$ = {:1.2f} m$^{-2}$ (GeV)$^{-1}$ s$^{-1}$, ".format(
                    _gpl["flux_density_per_m2_per_s_per_GeV"]
                ),
                "$E_0$ = {:1.2f} GeV, ".format(_gpl["pivot_energy_GeV"]),
                "$\\gamma = {:1.2f}$".format(_gpl["spectral_index"]),
            ]
        )
        return gamma_dF_per_m2_per_s_per_GeV, source_name

    else:
        raise KeyError("'type' must either be '3fgl', or 'generic_power_law'.")
