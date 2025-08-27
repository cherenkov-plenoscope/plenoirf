import numpy as np


def generate_paxel_intensity_median_hypot(event_frame):
    slope = np.hypot(
        event_frame["features/paxel_intensity_median_x"],
        event_frame["features/paxel_intensity_median_y"],
    )
    return np.sqrt(slope)


def generate_diff_image_and_light_front(event_frame):
    ev = event_frame
    f_raw = np.hypot(
        ev["features/image_infinity_cx_mean"] - ev["features/light_front_cx"],
        ev["features/image_infinity_cy_mean"] - ev["features/light_front_cy"],
    )
    return f_raw


def generate_image_infinity_std_density(event_frame):
    ev = event_frame
    std = np.hypot(
        ev["features/image_infinity_cx_std"],
        ev["features/image_infinity_cx_std"],
    )
    return np.log10(ev["features/num_photons"]) / std**2.0


def generate_A(event_frame):
    ev = event_frame
    shift = np.hypot(
        ev["features/image_half_depth_shift_cx"],
        ev["features/image_half_depth_shift_cy"],
    )
    return (
        ev["features/num_photons"]
        * shift
        / ev["features/image_smallest_ellipse_half_depth"]
    )


def generate_B(event_frame):
    ev = event_frame
    return (
        ev["features/num_photons"]
        / ev["features/image_smallest_ellipse_object_distance"] ** 2.0
    )


def generate_C(event_frame):
    ev = event_frame
    return ev["features/paxel_intensity_peakness_std_over_mean"] / np.log10(
        ev["features/image_smallest_ellipse_object_distance"]
    )


def generate_reco_core_radius_m(event_frame):
    ev = event_frame
    return np.hypot(
        ev["reconstructed_trajectory/x_m"],
        ev["reconstructed_trajectory/y_m"],
    )


def generate_reco_theta_rad(event_frame):
    ev = event_frame
    return np.hypot(
        ev["reconstructed_trajectory/cx_rad"],
        ev["reconstructed_trajectory/cy_rad"],
    )


def init_combined_features_structure():
    out = {}
    out["trajectory_reco_core_radius_m"] = {
        "generator": generate_reco_core_radius_m,
        "dtype": "<f4",
        "unit": "m",
        "transformation": {
            "function": "x",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    }
    out["trajectory_reco_theta_rad"] = {
        "generator": generate_reco_theta_rad,
        "dtype": "<f4",
        "unit": "rad",
        "transformation": {
            "function": "x",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    }
    out["combi_diff_image_and_light_front"] = {
        "generator": generate_diff_image_and_light_front,
        "dtype": "<f4",
        "unit": "rad",
        "transformation": {
            "function": "sqrt(x)",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    }
    out["combi_paxel_intensity_median_hypot"] = {
        "generator": generate_paxel_intensity_median_hypot,
        "dtype": "<f4",
        "unit": "$m^{1/2}$",
        "transformation": {
            "function": "log(x)",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    }
    out["combi_image_infinity_std_density"] = {
        "generator": generate_image_infinity_std_density,
        "dtype": "<f4",
        "unit": "$sr^{-1}$",
        "transformation": {
            "function": "log(x)",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    }
    out["combi_A"] = {
        "generator": generate_A,
        "dtype": "<f4",
        "unit": "$sr m^{-1}$",
        "transformation": {
            "function": "log(x)",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    }
    out["combi_B"] = {
        "generator": generate_B,
        "dtype": "<f4",
        "unit": "$m^{-2}$",
        "transformation": {
            "function": "log(x)",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    }
    out["combi_C"] = {
        "generator": generate_C,
        "dtype": "<f4",
        "unit": "$1$",
        "transformation": {
            "function": "x",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    }
    return out
