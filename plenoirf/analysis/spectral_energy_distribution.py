import numpy as np

one_eV_in_J = 1.602176634e-19
one_erg_in_J = 1e-7
one_erg_in_eV = one_erg_in_J / one_eV_in_J

PLENOIRF_SED_STYLE = {
    "x_energy_in_eV": 1e9,
    "y_inverse_energy_in_eV": 1e9,
    "y_inverse_area_in_m2": 1.0,
    "y_inverse_time_in_s": 1.0,
    "y_scale_energy_in_eV": 1e9,
    "y_scale_energy_power": 0.0,
    "x_label": "energy",
    "x_unit": "GeV",
    "y_label": "(differential flux)",
    "y_unit": "m$^{-2}$ s$^{-1}$ (GeV)$^{-1}$",
}

FERMI_SED_STYLE = {
    "x_energy_in_eV": 1e6,
    "y_inverse_energy_in_eV": one_erg_in_eV,
    "y_inverse_area_in_m2": 1e-4,
    "y_inverse_time_in_s": 1.0,
    "y_scale_energy_in_eV": one_erg_in_eV,
    "y_scale_energy_power": 2.0,
    "x_label": "energy",
    "x_unit": "MeV",
    "y_label": "energy$^{2}$ $\\times$ (differential flux)",
    "y_unit": "erg$^{2}$ (cm)$^{-2}$ s$^{-1}$ erg$^{-1}$",
}

SCIENCE_SED_STYLE = {
    "x_energy_in_eV": 1e6,
    "y_inverse_energy_in_eV": 1e6,
    "y_inverse_area_in_m2": 1e-4,
    "y_inverse_time_in_s": 1.0,
    "y_scale_energy_in_eV": 1e6,
    "y_scale_energy_power": 2.0,
    "x_label": "energy",
    "x_unit": "MeV",
    "y_label": "energy$^{2}$ $\\times$ (differential flux)",
    "y_unit": "(MeV)$^{2}$ (cm)$^{-2}$ s$^{-1}$ (MeV)$^{-1}$",
}

CHERENKOV_TELESCOPE_ARRAY_SED_STYLE = {
    "x_energy_in_eV": 1e12,
    "y_inverse_energy_in_eV": one_erg_in_eV,
    "y_inverse_area_in_m2": 1e-4,
    "y_inverse_time_in_s": 1.0,
    "y_scale_energy_in_eV": one_erg_in_eV,
    "y_scale_energy_power": 2.0,
    "x_label": "energy",
    "x_unit": "TeV",
    "y_label": "energy$^{2}$ $\\times$ (differential flux)",
    "y_unit": "erg$^{2}$ (cm)$^{-2}$ s$^{-1}$ erg$^{-1}$",
}
