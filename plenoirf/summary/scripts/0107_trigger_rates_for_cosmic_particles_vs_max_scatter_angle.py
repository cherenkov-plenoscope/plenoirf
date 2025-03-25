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
        paths["analysis_dir"],
        "0102_trigger_acceptance_for_cosmic_particles_vs_max_scatter_angle",
    )
)

energy_bin = json_utils.read(
    os.path.join(paths["analysis_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

# cosmic-ray-flux
# ----------------
airshower_fluxes = json_utils.tree.read(
    os.path.join(paths["analysis_dir"], "0017_flux_of_airshowers_rebin")
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

source_key = "diffuse"

# cosmic-rays
# -----------
for ck in res.COSMIC_RAYS:
    ck_dir = os.path.join(paths["out_dir"], ck)
    os.makedirs(ck_dir, exist_ok=True)

    Q = acceptance[ck][source_key]["mean"]
    Q_au = acceptance[ck][source_key]["absolute_uncertainty"]

    num_max_scatter_angles = Q.shape[0]

    R = np.zeros(num_max_scatter_angles)
    R_au = np.zeros(R.shape)
    dRdE = np.zeros(shape=(num_max_scatter_angles, energy_bin["num"]))
    dRdE_au = np.zeros(shape=dRdE.shape)

    cosmic_dFdE = airshower_fluxes[ck]["differential_flux"]
    cosmic_dFdE_au = airshower_fluxes[ck]["absolute_uncertainty"]

    for sc in range(num_max_scatter_angles):
        for eb in range(energy_bin["num"]):
            dRdE[sc, eb], dRdE_au[sc, eb] = pru.multiply(
                x=cosmic_dFdE[eb],
                x_au=cosmic_dFdE_au[eb],
                y=Q[sc, eb],
                y_au=Q_au[sc, eb],
            )

        R[sc], R_au[sc] = irf.utils.integrate_rate_where_known(
            dRdE=dRdE[sc, :],
            dRdE_au=dRdE_au[sc, :],
            E_edges=energy_bin["edges"],
        )

    json_utils.write(
        os.path.join(ck_dir, "differential.json"),
        {
            "comment": "Differential rate VS max. scatter angle VS energy",
            "unit": "s$^{-1} (GeV)$^{-1}$",
            "dRdE": dRdE,
            "dRdE_au": dRdE_au,
        },
    )
    json_utils.write(
        os.path.join(ck_dir, "integral.json"),
        {
            "comment": "Intrgral rate VS max. scatter angle.",
            "unit": "s$^{-1}$",
            "R": R,
            "R_au": R_au,
        },
    )
