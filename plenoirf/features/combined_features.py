import numpy as np

COMBINED_FEATURES = {}

# -----------------------------------------------------------------------------


def generate_diff_image_and_light_front(features):
    f_raw = np.hypot(
        features["image_infinity_cx_mean"] - features["light_front_cx"],
        features["image_infinity_cy_mean"] - features["light_front_cy"],
    )
    return f_raw


COMBINED_FEATURES["combi_diff_image_and_light_front"] = {
    "generator": generate_diff_image_and_light_front,
    "dtype": "<f8",
    "unit": "rad",
    "transformation": {
        "function": "sqrt(x)",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
}

# -----------------------------------------------------------------------------


def generate_paxel_intensity_median_hypot(features):
    slope = np.hypot(
        features["paxel_intensity_median_x"],
        features["paxel_intensity_median_y"],
    )
    return np.sqrt(slope)


COMBINED_FEATURES["combi_paxel_intensity_median_hypot"] = {
    "generator": generate_paxel_intensity_median_hypot,
    "dtype": "<f8",
    "unit": "$m^{1/2}$",
    "transformation": {
        "function": "log(x)",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
}

# -----------------------------------------------------------------------------


def generate_image_infinity_std_density(features):
    std = np.hypot(
        features["image_infinity_cx_std"], features["image_infinity_cx_std"]
    )
    return np.log10(features["num_photons"]) / std**2.0


COMBINED_FEATURES["combi_image_infinity_std_density"] = {
    "generator": generate_image_infinity_std_density,
    "dtype": "<f8",
    "unit": "$sr^{-1}$",
    "transformation": {
        "function": "log(x)",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
}

# -----------------------------------------------------------------------------


def generate_A(features):
    shift = np.hypot(
        features["image_half_depth_shift_cx"],
        features["image_half_depth_shift_cy"],
    )
    return (
        features["num_photons"]
        * shift
        / features["image_smallest_ellipse_half_depth"]
    )


COMBINED_FEATURES["combi_A"] = {
    "generator": generate_A,
    "dtype": "<f8",
    "unit": "$sr m^{-1}$",
    "transformation": {
        "function": "log(x)",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
}

# -----------------------------------------------------------------------------


def generate_B(features):
    return (
        features["num_photons"]
        / features["image_smallest_ellipse_object_distance"] ** 2.0
    )


COMBINED_FEATURES["combi_B"] = {
    "generator": generate_B,
    "dtype": "<f8",
    "unit": "$m^{-2}$",
    "transformation": {
        "function": "log(x)",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
}

# -----------------------------------------------------------------------------


def generate_C(features):
    return features["paxel_intensity_peakness_std_over_mean"] / np.log10(
        features["image_smallest_ellipse_object_distance"]
    )


COMBINED_FEATURES["combi_C"] = {
    "generator": generate_C,
    "dtype": "<f8",
    "unit": "$1$",
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
}
