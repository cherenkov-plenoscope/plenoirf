#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as sebplt
import json_utils
import propagate_uncertainties as pru


paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)

acceptance = json_utils.tree.read(
    os.path.join(
        paths["analysis_dir"], "0100_trigger_acceptance_for_cosmic_particles"
    )
)

energy_binning = json_utils.read(
    os.path.join(paths["analysis_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance"]
fine_energy_bin = energy_binning["interpolation"]

# cosmic-ray-flux
# ----------------
airshower_fluxes = json_utils.tree.read(
    os.path.join(paths["analysis_dir"], "0015_flux_of_airshowers")
)

# gamma-ray-flux of reference source
# ----------------------------------
gamma_source = json_utils.read(
    os.path.join(
        paths["analysis_dir"],
        "0009_flux_of_gamma_rays",
        "reference_source.json",
    )
)
gamma_dKdE = gamma_source["differential_flux"]["values"]
gamma_dKdE_au = np.zeros(gamma_dKdE.shape)

comment_differential = (
    "Differential trigger-rate, entire field-of-view. "
    "VS trigger-ratescan-thresholds"
)
comment_integral = (
    "Integral trigger-rate, entire field-of-view. "
    "VS trigger-ratescan-thresholds"
)

"""
A / m^{2}
Q / m^{2} sr

R / s^{-1}
dRdE / s^{-1} (GeV)^{-1}

F / s^{-1} m^{-2} (sr)^{-1}
dFdE / s^{-1} m^{-2} (sr)^{-1} (GeV)^{-1}

K / s^{-1} m^{-2}
dKdE / s^{-1} m^{-2} (GeV)^{-1}
"""

trigger_thresholds = np.array(
    res.analysis["trigger"][res.site_key]["ratescan_thresholds_pe"]
)
analysis_trigger_threshold = res.analysis["trigger"][res.site_key][
    "threshold_pe"
]
num_trigger_thresholds = len(trigger_thresholds)

# gamma-ray
# ---------
gamma_dir = os.path.join(paths["out_dir"], "gamma")
os.makedirs(gamma_dir, exist_ok=True)

_A = acceptance["gamma"]["point"]["mean"]
_A_au = acceptance["gamma"]["point"]["absolute_uncertainty"]

R = np.zeros(num_trigger_thresholds)
R_au = np.zeros(R.shape)
dRdE = np.zeros(shape=(num_trigger_thresholds, fine_energy_bin["num"]))
dRdE_au = np.zeros(shape=dRdE.shape)
for tt in range(num_trigger_thresholds):
    A = np.interp(
        x=fine_energy_bin["centers"],
        xp=energy_bin["centers"],
        fp=_A[tt, :],
    )
    A_au = np.interp(
        x=fine_energy_bin["centers"],
        xp=energy_bin["centers"],
        fp=_A_au[tt, :],
    )

    dRdE[tt, :], dRdE_au[tt, :] = pru.multiply(
        x=gamma_dKdE,
        x_au=gamma_dKdE_au,
        y=A,
        y_au=A_au,
    )

    R[tt], R_au[tt] = irf.utils.integrate_rate_where_known(
        dRdE=dRdE[tt, :],
        dRdE_au=dRdE_au[tt, :],
        E_edges=fine_energy_bin["edges"],
    )

json_utils.write(
    os.path.join(gamma_dir, "differential_rate.json"),
    {
        "comment": comment_differential + ", " + gamma_source["name"],
        "unit": "s$^{-1} (GeV)$^{-1}$",
        "mean": dRdE,
        "absolute_uncertainty": dRdE_au,
    },
)
json_utils.write(
    os.path.join(gamma_dir, "integral_rate.json"),
    {
        "comment": comment_integral + ", " + gamma_source["name"],
        "unit": "s$^{-1}$",
        "mean": R,
        "absolute_uncertainty": R_au,
    },
)

# cosmic-rays
# -----------
for ck in airshower_fluxes:
    ck_dir = os.path.join(paths["out_dir"], ck)
    os.makedirs(ck_dir, exist_ok=True)

    _Q = acceptance[ck]["diffuse"]["mean"]
    _Q_au = acceptance[ck]["diffuse"]["absolute_uncertainty"]

    R = np.zeros(num_trigger_thresholds)
    R_au = np.zeros(R.shape)
    dRdE = np.zeros(shape=(num_trigger_thresholds, fine_energy_bin["num"]))
    dRdE_au = np.zeros(shape=dRdE.shape)

    cosmic_dFdE = airshower_fluxes[ck]["differential_flux"]["values"]
    cosmic_dFdE_au = airshower_fluxes[ck]["differential_flux"][
        "absolute_uncertainty"
    ]

    for tt in range(num_trigger_thresholds):
        Q = np.interp(
            x=fine_energy_bin["centers"],
            xp=energy_bin["centers"],
            fp=_Q[tt, :],
        )
        Q_au = np.interp(
            x=fine_energy_bin["centers"],
            xp=energy_bin["centers"],
            fp=_Q_au[tt, :],
        )

        dRdE[tt, :], dRdE_au[tt, :] = pru.multiply(
            x=cosmic_dFdE,
            x_au=cosmic_dFdE_au,
            y=Q,
            y_au=Q_au,
        )

        R[tt], R_au[tt] = irf.utils.integrate_rate_where_known(
            dRdE=dRdE[tt, :],
            dRdE_au=dRdE_au[tt, :],
            E_edges=fine_energy_bin["edges"],
        )

    json_utils.write(
        os.path.join(ck_dir, "differential_rate.json"),
        {
            "comment": comment_differential,
            "unit": "s$^{-1} (GeV)$^{-1}$",
            "mean": dRdE,
            "absolute_uncertainty": dRdE_au,
        },
    )
    json_utils.write(
        os.path.join(ck_dir, "integral_rate.json"),
        {
            "comment": comment_integral,
            "unit": "s$^{-1}$",
            "mean": R,
            "absolute_uncertainty": R_au,
        },
    )
