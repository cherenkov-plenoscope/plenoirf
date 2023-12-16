import numpy as np
import spectral_energy_distribution_units as sed
from ..analysis import spectral_energy_distribution as sed_styles
from ._sensitivity_vs_observation_time import (
    sensitivity_vs_observation_time as _sens_vs_obs,
)


COLOR = "mediumblue"
LABEL = "CTA-South"


def sensitivity_vs_observation_time(energy_GeV=25.0):
    return _sens_vs_obs(energy_GeV=energy_GeV, instrument=LABEL)


def differential_sensitivity(observation_time=1800):
    enrgy_bin_edges_TeV = [
        0.02,
        0.0316,
        0.0501,
        0.0794,
        0.126,
        0.2,
        0.316,
        0.501,
        0.794,
        1.26,
        2,
        3.16,
        5.01,
        7.94,
        12.6,
        20,
        31.6,
        50.1,
        79.4,
        126,
        200,
    ]

    num_bins = len(enrgy_bin_edges_TeV) - 1

    energy_TeV = []
    for ee in range(num_bins):
        energy_TeV.append(enrgy_bin_edges_TeV[ee])
        energy_TeV.append(enrgy_bin_edges_TeV[ee + 1])
    energy_TeV = np.array(energy_TeV)

    # E^{2} x Flux Sensitivity (erg cm^{-2} s^{-1})

    if observation_time == 1800:
        E2_senitivity_erg_per_cm2_per_s = [
            6.16e-11,
            3.23e-11,
            1.58e-11,
            7.61e-12,
            4.94e-12,
            3.51e-12,
            2.79e-12,
            2.45e-12,
            2.14e-12,
            1.97e-12,
            1.98e-12,
            2.28e-12,
            2.91e-12,
            3.96e-12,
            5.83e-12,
            8.74e-12,
            1.39e-11,
            2.11e-11,
            3.26e-11,
            5.12e-11,
        ]
    elif observation_time == 5 * 3600:
        E2_senitivity_erg_per_cm2_per_s = [
            2.79e-11,
            1.22e-11,
            5.78e-12,
            2.59e-12,
            1.53e-12,
            1.01e-12,
            7.2e-13,
            5.72e-13,
            4.59e-13,
            4.03e-13,
            3.54e-13,
            3.64e-13,
            4.5e-13,
            5.24e-13,
            6.81e-13,
            9.67e-13,
            1.5e-12,
            2.28e-12,
            3.51e-12,
            5.46e-12,
        ]
    elif observation_time == 50 * 3600:
        E2_senitivity_erg_per_cm2_per_s = [
            2.18e-11,
            6.94e-12,
            1.83e-12,
            7.97e-13,
            4.63e-13,
            2.89e-13,
            1.98e-13,
            1.44e-13,
            9.67e-14,
            8.12e-14,
            6.63e-14,
            6.3e-14,
            7.07e-14,
            6.28e-14,
            8.27e-14,
            1.18e-13,
            1.88e-13,
            2.89e-13,
            4.59e-13,
            7.14e-13,
        ]
    else:
        assert False, "No such observation_time"

    sens = []
    for ss in range(num_bins):
        sens.append(E2_senitivity_erg_per_cm2_per_s[ss])
        sens.append(E2_senitivity_erg_per_cm2_per_s[ss])

    CTA_STYLE = sed_styles.CHERENKOV_TELESCOPE_ARRAY_SED_STYLE

    assert CTA_STYLE["x_energy_in_eV"] == 1e12
    assert CTA_STYLE["y_inverse_energy_in_eV"] == sed_styles.one_erg_in_eV
    assert CTA_STYLE["y_inverse_area_in_m2"] == 1e-4
    assert CTA_STYLE["y_inverse_time_in_s"] == 1.0
    assert CTA_STYLE["y_scale_energy_in_eV"] == sed_styles.one_erg_in_eV
    assert CTA_STYLE["y_scale_energy_power"] == 2.0

    energy, dfdE = sed.convert_units_with_style(
        x=energy_TeV,
        y=sens,
        input_style=CTA_STYLE,
        target_style=sed_styles.PLENOIRF_SED_STYLE,
    )

    return {
        "energy": {
            "values": energy,
            "unit": "GeV",
            "unit": "GeV",
        },
        "differential_flux": {
            "values": dfdE,
            "unit": "m$^{-2}$ s$^{-1}$ GeV$^{-1}$",
        },
        "comment": "www.cta-observatory.org/science/cta-performance/ (prod3b-v2)",
        "title": "CTA-South, observation-time: {:d}s".format(observation_time),
    }


def energy_resolution():
    raw = np.array(
        [
            [2.6315e-2, 2.7293e-1],
            [2.8905e-2, 2.6694e-1],
            [3.1485e-2, 2.6203e-1],
            [3.3983e-2, 2.5646e-1],
            [3.6735e-2, 2.4895e-1],
            [3.9328e-2, 2.4203e-1],
            [4.1073e-2, 2.3661e-1],
            [4.3372e-2, 2.3115e-1],
            [4.5190e-2, 2.2535e-1],
            [4.6719e-2, 2.2107e-1],
            [4.9003e-2, 2.1486e-1],
            [5.1481e-2, 2.1178e-1],
            [5.2770e-2, 2.0518e-1],
            [5.6059e-2, 1.9744e-1],
            [5.8800e-2, 1.9110e-1],
            [6.1645e-2, 1.8495e-1],
            [6.5546e-2, 1.7987e-1],
            [6.7856e-2, 1.7352e-1],
            [7.3544e-2, 1.6781e-1],
            [7.8361e-2, 1.6203e-1],
            [8.3247e-2, 1.5725e-1],
            [9.2863e-2, 1.5071e-1],
            [1.0279e-1, 1.4361e-1],
            [1.1514e-1, 1.3716e-1],
            [1.2643e-1, 1.3196e-1],
            [1.4184e-1, 1.2574e-1],
            [1.6136e-1, 1.2041e-1],
            [1.8150e-1, 1.1480e-1],
            [2.0343e-1, 1.1006e-1],
            [2.3348e-1, 1.0518e-1],
            [2.7218e-1, 9.8577e-2],
            [3.1721e-1, 9.5208e-2],
            [3.5675e-1, 9.0963e-2],
            [4.1744e-1, 8.7080e-2],
            [4.9198e-1, 8.3014e-2],
            [5.7783e-1, 7.8939e-2],
            [6.9094e-1, 7.4572e-2],
            [8.0674e-1, 7.0930e-2],
            [9.2339e-1, 6.8206e-2],
            [1.1340e0, 6.4778e-2],
            [1.2879e0, 6.3281e-2],
            [1.5696e0, 6.1505e-2],
            [1.8251e0, 6.0554e-2],
            [2.1468e0, 5.9962e-2],
            [2.5123e0, 5.9045e-2],
            [3.0303e0, 5.8893e-2],
            [3.6852e0, 5.8505e-2],
            [4.8449e0, 5.9302e-2],
            [5.8198e0, 5.9302e-2],
            [7.4042e0, 5.8701e-2],
            [8.9044e0, 5.9137e-2],
            [1.0439e1, 6.0385e-2],
            [1.3033e1, 6.1434e-2],
            [1.4959e1, 6.4083e-2],
            [1.8014e1, 6.6591e-2],
            [2.1087e1, 6.8351e-2],
            [2.4517e1, 7.0372e-2],
            [2.8560e1, 7.2201e-2],
            [3.3501e1, 7.4896e-2],
            [3.9217e1, 7.5891e-2],
            [4.6521e1, 7.7544e-2],
            [5.6706e1, 7.9589e-2],
            [6.6359e1, 8.0600e-2],
            [8.2141e1, 8.2694e-2],
            [9.5762e1, 8.3271e-2],
        ]
    )

    reco_energy_TeV = raw[:, 0]
    resolution_68 = raw[:, 1]

    res = {
        "reconstructed_energy": {
            "values": 1e3 * reco_energy_TeV,
            "label": "reco. energy",
            "unit": "GeV",
        },
        "energy_resolution_68": {
            "values": resolution_68,
            "label": "energy (reco. - true) / true (68% containment)",
            "unit": "1",
        },
        "reference": {
            "url": "https://www.cta-observatory.org/science/cta-performance",
            "date": "2021-06-09",
            "production": "prod3b-v2",
            "zenith_distance": "20deg",
            "comment": (
                "The energy resolution Delta E / E is obtained from the "
                "distribution of (ER – ET) / ET, where R and T refer to "
                "the reconstructed and true energies of gamma-ray events "
                "recorded by CTA. Delta E/E is the half-width of the interval "
                "around 0 which contains 68% of the distribution. "
                "The plot shows the energy resolution as a function of "
                "reconstructed energy  (the result depends only weakly on "
                "the assumed gamma-ray spectrum; for the results here "
                "we use dNgamma/dE ~E-2.62). The full energy migration "
                "matrix is provided, in each of the the IRF files, in "
                "two versions: one filled with all gamma events surviving "
                "the gamma/hadron separation cuts, suitable for cases in "
                "which there is no a priori knowledge of the true "
                "direction of incoming gamma rays (e.g. for the observation "
                "of diffuse sources), and another one filled after "
                "applying a cut on the angle between the true and the "
                "reconstructed gamma-ray direction "
                "(for observations of point-like objects) - "
                "the angular cut is the same used for the calculation of "
                "the point source sensitivity."
            ),
        },
    }

    return res


def angular_resolution():
    raw = np.array(
        [
            [3.4905e-2, 2.4849e-1],
            [3.5639e-2, 2.4497e-1],
            [3.6630e-2, 2.4116e-1],
            [3.7703e-2, 2.3723e-1],
            [3.8807e-2, 2.3327e-1],
            [3.9943e-2, 2.2933e-1],
            [4.1113e-2, 2.2536e-1],
            [4.2328e-2, 2.2146e-1],
            [4.3329e-2, 2.1776e-1],
            [4.4585e-2, 2.1409e-1],
            [4.5657e-2, 2.1052e-1],
            [4.6723e-2, 2.0719e-1],
            [4.7813e-2, 2.0385e-1],
            [4.8930e-2, 2.0052e-1],
            [5.0072e-2, 1.9718e-1],
            [5.1241e-2, 1.9384e-1],
            [5.2437e-2, 1.9049e-1],
            [5.3662e-2, 1.8721e-1],
            [5.5036e-2, 1.8372e-1],
            [5.6578e-2, 1.7994e-1],
            [5.8235e-2, 1.7614e-1],
            [6.0516e-2, 1.7208e-1],
            [6.2947e-2, 1.6794e-1],
            [6.4826e-2, 1.6446e-1],
            [6.7264e-2, 1.6092e-1],
            [6.9670e-2, 1.5742e-1],
            [7.2342e-2, 1.5405e-1],
            [7.5335e-2, 1.5069e-1],
            [7.8680e-2, 1.4728e-1],
            [8.2412e-2, 1.4381e-1],
            [8.6530e-2, 1.4042e-1],
            [9.0517e-2, 1.3708e-1],
            [9.5465e-2, 1.3352e-1],
            [1.0036e-1, 1.2997e-1],
            [1.0543e-1, 1.2651e-1],
            [1.1083e-1, 1.2300e-1],
            [1.1693e-1, 1.1935e-1],
            [1.2329e-1, 1.1574e-1],
            [1.2989e-1, 1.1234e-1],
            [1.3725e-1, 1.0884e-1],
            [1.4544e-1, 1.0534e-1],
            [1.5509e-1, 1.0212e-1],
            [1.6718e-1, 9.8505e-2],
            [1.7923e-1, 9.5070e-2],
            [1.9384e-1, 9.1556e-2],
            [2.1025e-1, 8.8057e-2],
            [2.2872e-1, 8.4726e-2],
            [2.4954e-1, 8.1375e-2],
            [2.7384e-1, 7.8067e-2],
            [3.0313e-1, 7.4610e-2],
            [3.3753e-1, 7.1318e-2],
            [3.7912e-1, 6.7896e-2],
            [4.2832e-1, 6.4674e-2],
            [4.8392e-1, 6.1650e-2],
            [5.4674e-1, 5.8970e-2],
            [6.1773e-1, 5.6423e-2],
            [6.9795e-1, 5.4225e-2],
            [7.8858e-1, 5.2119e-2],
            [8.9101e-1, 5.0345e-2],
            [1.0068e0, 4.8961e-2],
            [1.1376e0, 4.8208e-2],
            [1.2855e0, 4.7833e-2],
            [1.4526e0, 4.7611e-2],
            [1.6414e0, 4.7328e-2],
            [1.8547e0, 4.6973e-2],
            [2.0958e0, 4.6613e-2],
            [2.3682e0, 4.6223e-2],
            [2.6760e0, 4.5432e-2],
            [3.0237e0, 4.4434e-2],
            [3.4166e0, 4.3402e-2],
            [3.8605e0, 4.2247e-2],
            [4.3621e0, 4.1062e-2],
            [4.9288e0, 3.9869e-2],
            [5.5692e0, 3.8677e-2],
            [6.2928e0, 3.7472e-2],
            [7.1104e0, 3.6291e-2],
            [8.0342e0, 3.5102e-2],
            [9.0781e0, 3.3913e-2],
            [1.0258e1, 3.2816e-2],
            [1.1591e1, 3.1960e-2],
            [1.3097e1, 3.1115e-2],
            [1.4799e1, 3.0327e-2],
            [1.6722e1, 2.9704e-2],
            [1.8895e1, 2.9176e-2],
            [2.1351e1, 2.8653e-2],
            [2.4126e1, 2.8125e-2],
            [2.7262e1, 2.7754e-2],
            [3.0806e1, 2.7490e-2],
            [3.4811e1, 2.7223e-2],
            [3.9336e1, 2.6948e-2],
            [4.4449e1, 2.6768e-2],
            [5.0227e1, 2.6603e-2],
            [5.6757e1, 2.6439e-2],
            [6.4135e1, 2.6355e-2],
            [7.2474e1, 2.6550e-2],
            [8.1897e1, 2.6772e-2],
            [9.2547e1, 2.7150e-2],
            [9.9532e1, 2.7282e-2],
        ]
    )

    reco_energy_TeV = raw[:, 0]
    direction_resolution_deg = raw[:, 1]

    res = {
        "reconstructed_energy": {
            "values": 1e3 * reco_energy_TeV,
            "label": "reco. energy",
            "unit": "GeV",
        },
        "angular_resolution_68": {
            "values": np.deg2rad(direction_resolution_deg),
            "unit": "rad",
            "label": "angular resolution 68%",
        },
        "reference": {
            "url": "https://www.cta-observatory.org/science/cta-performance",
            "date": "2021-06-09",
            "production": "prod3b-v2",
            "zenith_distance": "20deg",
            "comment": (
                "The angular resolution vs. reconstructed energy "
                "curve shows the angle within which 68% of "
                "reconstructed gamma rays fall, relative to "
                "their true direction. Gamma-hadron separation "
                "cuts are applied for the MC events used to "
                "determine the angular resolution. Note that "
                "this analysis is not optimised to provide the "
                "best possible angular resolution, but rather "
                "the best point-source sensitivity (compatible "
                "with the minimum required angular resolution). "
                "Dedicated analysis cuts can provide, relative "
                "to the IRFs provided here, improved angular "
                "(or spectral) resolution at the expense of "
                "collection area, enabling e.g. a better study "
                "of the morphology or spectral "
                "characteristics of bright sources."
            ),
        },
    }

    return res
