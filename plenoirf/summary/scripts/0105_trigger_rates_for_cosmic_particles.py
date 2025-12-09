#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils
import propagate_uncertainties as pru
import rename_after_writing as rnw


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

energy_bin = res.energy_binning(key="10_bins_per_decade")
zenith_bin = res.zenith_binning("3_bins_per_45deg")
fine_energy_bin = res.energy_binning(key="60_bins_per_decade")

acceptance = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"],
        "0100_trigger_acceptance_for_cosmic_particles",
    )
)

# cosmic-ray-flux
# ----------------
airshower_fluxes = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0015_flux_of_airshowers")
)

# gamma-ray-flux of reference source
# ----------------------------------
gamma_source = json_utils.read(
    opj(
        res.paths["analysis_dir"],
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

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    # gamma-ray
    # ---------
    gamma_dir = opj(res.paths["out_dir"], zk, "gamma")
    os.makedirs(gamma_dir, exist_ok=True)

    _A = acceptance[zk]["gamma"]["point"]["mean"]
    _A_au = acceptance[zk]["gamma"]["point"]["absolute_uncertainty"]

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
        opj(gamma_dir, "differential_rate.json"),
        {
            "comment": comment_differential + ", " + gamma_source["name"],
            "unit": "s$^{-1} (GeV)$^{-1}$",
            "mean": dRdE,
            "absolute_uncertainty": dRdE_au,
        },
    )
    json_utils.write(
        opj(gamma_dir, "integral_rate.json"),
        {
            "comment": comment_integral + ", " + gamma_source["name"],
            "unit": "s$^{-1}$",
            "mean": R,
            "absolute_uncertainty": R_au,
        },
    )

    # cosmic-rays
    # -----------
    for ck in res.COSMIC_RAYS:
        ck_dir = opj(res.paths["out_dir"], zk, ck)
        os.makedirs(ck_dir, exist_ok=True)

        _Q = acceptance[zk][ck]["diffuse"]["mean"]
        _Q_au = acceptance[zk][ck]["diffuse"]["absolute_uncertainty"]

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
            opj(ck_dir, "differential_rate.json"),
            {
                "comment": comment_differential,
                "unit": "s$^{-1} (GeV)$^{-1}$",
                "mean": dRdE,
                "absolute_uncertainty": dRdE_au,
            },
        )
        json_utils.write(
            opj(ck_dir, "integral_rate.json"),
            {
                "comment": comment_integral,
                "unit": "s$^{-1}$",
                "mean": R,
                "absolute_uncertainty": R_au,
            },
        )


def find_index_of_analysis_trigger_threshold(
    trigger_thresholds, script_resources
):
    res = script_resources
    _index = -1
    for i in range(len(trigger_thresholds)):
        if (
            trigger_thresholds[i]
            == res.analysis["trigger"][res.site_key]["threshold_pe"]
        ):
            _index = i
    return _index


# summarize rates

tti = find_index_of_analysis_trigger_threshold(
    trigger_thresholds=trigger_thresholds, script_resources=res
)

inte = {}
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    inte[zk] = {}
    for pk in res.PARTICLES:
        zk_pk_dir = opj(res.paths["out_dir"], zk, pk)
        with open(opj(zk_pk_dir, "integral_rate.json"), "rt") as fin:
            inr = json_utils.loads(fin.read())
        inte[zk][pk] = {
            "R": float(inr["mean"][tti]),
            "R_au": float(inr["absolute_uncertainty"][tti]),
        }

    R_tot = []
    R_tot_au = []
    for pk in res.PARTICLES:
        R_tot.append(inte[zk][pk]["R"])
        R_tot_au.append(inte[zk][pk]["R_au"])

    Rt, Rt_au = pru.sum(x=R_tot, x_au=R_tot_au)
    inte[zk]["__total__"] = {"R": float(Rt), "R_au": float(Rt_au)}

with rnw.open(
    opj(res.paths["out_dir"], "summary.zenith.particle.json"), "wt"
) as fout:
    fout.write(json_utils.dumps(inte, indent=4))


SNR = {}
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    SNR[zk] = {}

    Rbg, Rbg_au = pru.add(
        x=inte[zk]["proton"]["R"],
        x_au=inte[zk]["proton"]["R_au"],
        y=inte[zk]["helium"]["R"],
        y_au=inte[zk]["helium"]["R_au"],
    )

    snr, snr_au = pru.divide(
        x=inte[zk]["gamma"]["R"],
        x_au=inte[zk]["gamma"]["R_au"],
        y=Rbg,
        y_au=Rbg_au,
    )
    SNR[zk] = {"snr_u": snr * 1e6, "snr_u_au": snr_au * 1e6}

with rnw.open(opj(res.paths["out_dir"], "snr.json"), "wt") as fout:
    fout.write(json_utils.dumps(SNR, indent=4))

res.stop()
