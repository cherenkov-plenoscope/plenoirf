import numpy as np
from .. import utils


def momentum_to_cx_cy_wrt_aperture(
    momentum_x_GeV_per_c,
    momentum_y_GeV_per_c,
    momentum_z_GeV_per_c,
    plenoscope_pointing,
):
    assert plenoscope_pointing["zenith_deg"] == 0.0
    assert plenoscope_pointing["azimuth_deg"] == 0.0
    WRT_APERTURE = -1.0
    momentum = np.array(
        [
            momentum_x_GeV_per_c,
            momentum_y_GeV_per_c,
            momentum_z_GeV_per_c,
        ]
    ).T
    momentum_norm = np.linalg.norm(momentum, axis=1)
    for m in range(len(momentum_norm)):
        momentum[m, :] /= momentum_norm[m]
    return WRT_APERTURE * momentum[:, 0], WRT_APERTURE * momentum[:, 1]


def histogram_theta_square(theta_deg, theta_square_bin_edges_deg2):
    num_airshower = theta_deg.shape[0]
    if num_airshower > 0:
        theta_square_hist = np.histogram(
            theta_deg**2, bins=theta_square_bin_edges_deg2
        )[0]
        theta_square_hist_relunc = utils._divide_silent(
            np.sqrt(theta_square_hist), theta_square_hist, np.nan
        )
    else:
        theta_square_hist = np.zeros(
            len(theta_square_bin_edges_deg2) - 1, dtype=np.int64
        )
        theta_square_hist_relunc = np.nan * np.ones(
            len(theta_square_bin_edges_deg2) - 1, dtype=np.float64
        )
    return theta_square_hist, theta_square_hist_relunc


def estimate_containment_radius(theta_deg, psf_containment_factor):
    num_airshower = theta_deg.shape[0]
    if num_airshower > 0:
        theta_containment_deg = np.quantile(
            theta_deg,
            q=psf_containment_factor,
            interpolation="nearest",
        )
        theta_containment_deg_relunc = 1.0 / np.sqrt(num_airshower)
    else:
        theta_containment_deg = np.nan
        theta_containment_deg_relunc = np.nan
    return theta_containment_deg, theta_containment_deg_relunc
