import numpy as np
import spectral_energy_distribution_units as sed
from ..analysis import spectral_energy_distribution as sed_styles


def sensitivity_vs_observation_time(energy_GeV=25.0, instrument="cta_south"):
    if energy_GeV == 25.0:
        cta = np.array(
            [
                [1.54448e1, 6.15336e-9],
                [2.70159e1, 3.52307e-9],
                [4.72564e1, 2.01004e-9],
                [8.26598e1, 1.16304e-9],
                [1.51192e2, 7.59478e-10],
                [2.52859e2, 4.64194e-10],
                [4.42192e2, 3.32821e-10],
                [7.73277e2, 2.42859e-10],
                [1.35223e3, 1.80991e-10],
                [2.36467e3, 1.33000e-10],
                [4.13503e3, 1.00170e-10],
                [7.23083e3, 7.54429e-11],
                [1.26442e4, 5.74223e-11],
                [2.21098e4, 4.47950e-11],
                [3.86615e4, 3.48218e-11],
                [6.76010e4, 2.82353e-11],
                [1.18198e5, 2.37972e-11],
                [2.06657e5, 2.07013e-11],
                [3.61304e5, 1.87182e-11],
                [6.31644e5, 1.77787e-11],
                [1.10428e6, 1.65922e-11],
                [1.93045e6, 1.64963e-11],
                [3.37468e6, 1.65167e-11],
                [5.89939e6, 1.65371e-11],
                [1.03133e7, 1.60419e-11],
                [1.80291e7, 1.59492e-11],
                [3.15173e7, 1.59689e-11],
                [5.50965e7, 1.59886e-11],
                [9.63193e7, 1.55098e-11],
                [1.68375e8, 1.58600e-11],
                [2.94335e8, 1.62180e-11],
                [5.14544e8, 1.60677e-11],
                [8.99491e8, 1.60876e-11],
                [1.57243e9, 1.61074e-11],
                [2.74882e9, 1.61273e-11],
                [4.80530e9, 1.61472e-11],
                [7.03254e9, 1.61608e-11],
            ]
        )

        fermi = np.array(
            [
                [1.38090e1, 8.70556e-5],
                [2.41550e1, 4.90454e-5],
                [4.22516e1, 2.82409e-5],
                [7.39021e1, 1.70497e-5],
                [1.29273e2, 9.51841e-6],
                [2.26128e2, 5.35790e-6],
                [3.95531e2, 3.15127e-6],
                [6.91860e2, 1.80748e-6],
                [1.21024e3, 1.00265e-6],
                [2.11687e3, 5.94580e-7],
                [3.62533e3, 3.38095e-7],
                [6.42264e3, 1.87740e-7],
                [1.13295e4, 1.11653e-7],
                [1.98172e4, 6.48271e-8],
                [3.57111e4, 4.11470e-8],
                [6.15767e4, 1.78582e-8],
                [1.07062e5, 1.36805e-8],
                [1.90551e5, 7.89615e-9],
                [3.21777e5, 4.03152e-9],
                [5.67650e5, 2.24790e-9],
                [9.72135e5, 1.29960e-9],
                [1.73689e6, 7.16254e-10],
                [3.03804e6, 4.25385e-10],
                [5.31425e6, 2.38275e-10],
                [9.29559e6, 1.37536e-10],
                [1.62600e7, 7.80828e-11],
                [2.85500e7, 4.95109e-11],
                [4.83210e7, 2.90515e-11],
                [1.74532e8, 7.88401e-12],
                [3.55583e8, 3.88780e-12],
                [5.96381e8, 2.46769e-12],
                [1.05790e9, 1.44023e-12],
                [1.83739e9, 7.12110e-13],
                [3.21409e9, 3.92323e-13],
                [7.82200e9, 1.89423e-13],
            ]
        )
    else:
        assert False, "No such energy_GeV"

    style = {
        "x_energy_in_eV": 1e9,
        "y_inverse_energy_in_eV": sed_styles.one_erg_in_eV,
        "y_inverse_area_in_m2": 1e-4,
        "y_inverse_time_in_s": 1.0,
        "y_scale_energy_in_eV": sed_styles.one_erg_in_eV,
        "y_scale_energy_power": 2.0,
    }

    PIVOT_ENERGY_GEV = 25.0

    if instrument == "Fermi-LAT":
        _energy_GeV = PIVOT_ENERGY_GEV * np.ones(fermi.shape[0])
        _sens_E2_rg_per_cm2_per_s = fermi[:, 1]
        obstime_s = fermi[:, 0]
    elif instrument == "CTA-South":
        _energy_GeV = PIVOT_ENERGY_GEV * np.ones(cta.shape[0])
        _sens_E2_rg_per_cm2_per_s = cta[:, 1]
        obstime_s = cta[:, 0]

    energy, dfdE = sed.convert_units_with_style(
        x=_energy_GeV,
        y=_sens_E2_rg_per_cm2_per_s,
        input_style=style,
        target_style=sed_styles.PLENOIRF_SED_STYLE,
    )

    assert energy[0] == PIVOT_ENERGY_GEV

    return {
        "pivot_energy": {
            "values": PIVOT_ENERGY_GEV,
            "unit": "GeV",
        },
        "differential_flux": {
            "values": dfdE,
            "unit": "m$^{-2}$ s$^{-1}$ GeV$^{-1}$",
        },
        "observation_time": {
            "values": obstime_s,
            "unit": "s",
        },
        "reference": {
            "key": "funk2013comparison",
            "title": "Comparison of Fermi-LAT and CTA in the region between 10--100 GeV",
            "author": "Funk, Stefan and Hinton, JA and CTA Consortium and others",
            "journal": "Astroparticle Physics",
            "volume": "43",
            "pages": "348--355",
            "year": "2013",
            "publisher": "Elsevier",
        },
    }
