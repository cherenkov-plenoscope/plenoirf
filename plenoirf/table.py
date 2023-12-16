import numpy as np
from atmospheric_cherenkov_response.production import table as acr_table

# basics
# ------

STRUCTURE = {}
for key in acr_table.STRUCTURE:
    STRUCTURE[key] = acr_table.STRUCTURE[key].copy()


# specifics
# ---------

STRUCTURE["trigger"] = {
    "num_cherenkov_pe": {"dtype": "<i8", "comment": ""},
    "response_pe": {"dtype": "<i8", "comment": ""},
    "focus_00_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_01_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_02_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_03_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_04_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_05_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_06_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_07_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_08_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_09_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_10_response_pe": {"dtype": "<i8", "comment": ""},
    "focus_11_response_pe": {"dtype": "<i8", "comment": ""},
}

STRUCTURE["pasttrigger"] = {}

STRUCTURE["cherenkovclassification"] = {
    "num_true_positives": {"dtype": "<i8", "comment": ""},
    "num_false_negatives": {"dtype": "<i8", "comment": ""},
    "num_false_positives": {"dtype": "<i8", "comment": ""},
    "num_true_negatives": {"dtype": "<i8", "comment": ""},
}

STRUCTURE["features"] = {}
STRUCTURE["features"]["num_photons"] = {
    "dtype": "<i8",
    "comment": "The number of photon-eqivalents that are identified to be dense cluster(s) of Cherenkov-photons",
    "transformation": {
        "function": "log(x)",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "1",
}
STRUCTURE["features"]["paxel_intensity_peakness_std_over_mean"] = {
    "dtype": "<f8",
    "comment": "A measure for the intensity distribution on the aperture-plane. The larger the value, the less evenly the intensity is distributed on the plane.",
    "transformation": {
        "function": "log(x)",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "1",
}
STRUCTURE["features"]["paxel_intensity_peakness_max_over_mean"] = {
    "dtype": "<f8",
    "comment": "A measure for the intensity distribution on the aperture-plane. The larger the value, the more the intensity is concentrated in a small area on the aperture-plane.",
    "transformation": {
        "function": "log(x)",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "1",
}

paxel_intensity_median_str = "Median intersection-positions in {:s} of reconstructed Cherenkov-photons on the aperture-plane"
STRUCTURE["features"]["paxel_intensity_median_x"] = {
    "dtype": "<f8",
    "comment": paxel_intensity_median_str.format("x"),
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "m",
}
STRUCTURE["features"]["paxel_intensity_median_y"] = {
    "dtype": "<f8",
    "comment": paxel_intensity_median_str.format("y"),
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "m",
}

_watershed_str = "A measure for the areal distribution of reconstructed Cherenkov-photons on the aperture-plane."
STRUCTURE["features"]["aperture_num_islands_watershed_rel_thr_2"] = {
    "dtype": "<i8",
    "comment": _watershed_str,
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "1",
}
STRUCTURE["features"]["aperture_num_islands_watershed_rel_thr_4"] = {
    "dtype": "<i8",
    "comment": _watershed_str,
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "1",
}
STRUCTURE["features"]["aperture_num_islands_watershed_rel_thr_8"] = {
    "dtype": "<i8",
    "comment": _watershed_str,
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "1",
}

_light_front_c_str = "Incident-direction in {:s} of reconstructed Cherenkov-photon-plane passing through the aperture-plane."
STRUCTURE["features"]["light_front_cx"] = {
    "dtype": "<f8",
    "comment": _light_front_c_str.format("x"),
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "rad",
}
STRUCTURE["features"]["light_front_cy"] = {
    "dtype": "<f8",
    "comment": _light_front_c_str.format("y"),
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "rad",
}

_image_infinity_c_mean_str = "Mean incident-direction in {:s} of reconstructed Cherenkov-photons in the image focussed to infinity."
STRUCTURE["features"]["image_infinity_cx_mean"] = {
    "dtype": "<f8",
    "comment": _image_infinity_c_mean_str.format("x"),
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "rad",
}
STRUCTURE["features"]["image_infinity_cy_mean"] = {
    "dtype": "<f8",
    "comment": _image_infinity_c_mean_str.format("y"),
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "rad",
}
STRUCTURE["features"]["image_infinity_cx_std"] = {
    "dtype": "<f8",
    "comment": "",
    "transformation": {
        "function": "log(x)",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "rad",
}
STRUCTURE["features"]["image_infinity_cy_std"] = {
    "dtype": "<f8",
    "comment": "",
    "transformation": {
        "function": "log(x)",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "rad",
}
STRUCTURE["features"]["image_infinity_num_photons_on_edge_field_of_view"] = {
    "dtype": "<i8",
    "comment": "Number of photon-eqivalents on the edge of the field-of-view in an image focused on infinity.",
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "p.e.",
}
STRUCTURE["features"]["image_smallest_ellipse_object_distance"] = {
    "dtype": "<f8",
    "comment": "The object-distance in front of the aperture where the refocused image of the airshower yields the Hillas-ellipse with the smallest solid angle. See also 'image_smallest_ellipse_solid_angle'.",
    "transformation": {
        "function": "log(x)",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "m",
}
STRUCTURE["features"]["image_smallest_ellipse_solid_angle"] = {
    "dtype": "<f8",
    "comment": "The solid angle of the smallest Hillas-ellipse in all refocused images. See also 'image_smallest_ellipse_object_distance'.",
    "transformation": {
        "function": "log(x)",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "sr",
}
STRUCTURE["features"]["image_smallest_ellipse_half_depth"] = {
    "dtype": "<f8",
    "comment": "The range in object-distance for the Hillas-ellipse to double its solid angle when refocusing starts at the smallest ellipse.",
    "transformation": {
        "function": "log(x)",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "m",
}

image_half_depth_shift_c_str = "How much the mean intensity in the image shifts in {:s} when refocussing from smallest to double solid angle of ellipse."
STRUCTURE["features"]["image_half_depth_shift_cx"] = {
    "dtype": "<f8",
    "comment": image_half_depth_shift_c_str.format("cx"),
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "rad",
}
STRUCTURE["features"]["image_half_depth_shift_cy"] = {
    "dtype": "<f8",
    "comment": image_half_depth_shift_c_str.format("cy"),
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "rad",
}
STRUCTURE["features"][
    "image_smallest_ellipse_num_photons_on_edge_field_of_view"
] = {
    "dtype": "<i8",
    "comment": "Number of photon-eqivalents on the edge of the field-of-view in an image focused to the smallest Hillas-ellipse.",
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "p.e.",
}
STRUCTURE["features"]["image_num_islands"] = {
    "dtype": "<i8",
    "comment": "The number of individual dense clusters of reconstructed Cherenkov-photons in the image-space.",
    "transformation": {
        "function": "x",
        "shift": "mean(x)",
        "scale": "std(x)",
        "quantile_range": [0.01, 0.99],
    },
    "unit": "1",
}

_traj = {}

_traj_xy_comment = (
    "Primary particle's core-position w.r.t. principal aperture-plane."
)
_traj["x_m"] = {
    "dtype": "<f8",
    "comment": _traj_xy_comment,
    "unit": "m",
}
_traj["y_m"] = {
    "dtype": "<f8",
    "comment": _traj_xy_comment,
    "unit": "m",
}

_traj_cxy_comment = "Primary particle's direction w.r.t. pointing."
_traj["cx_rad"] = {
    "dtype": "<f8",
    "comment": _traj_cxy_comment,
    "unit": "rad",
}
_traj["cy_rad"] = {
    "dtype": "<f8",
    "comment": _traj_cxy_comment,
    "unit": "rad",
}

_traj_cxy_comment_fuzzy = (
    "Primary particle's direction w.r.t. "
    + "pointing according to fuzzy-estimator."
)
_traj["fuzzy_cx_rad"] = {
    "dtype": "<f8",
    "comment": _traj_cxy_comment_fuzzy,
    "unit": "rad",
}
_traj["fuzzy_cy_rad"] = {
    "dtype": "<f8",
    "comment": _traj_cxy_comment_fuzzy,
    "unit": "rad",
}

_traj["fuzzy_main_axis_support_cx_rad"] = {
    "dtype": "<f8",
    "comment": "",
    "unit": "rad",
}
_traj["fuzzy_main_axis_support_cy_rad"] = {
    "dtype": "<f8",
    "comment": "",
    "unit": "rad",
}
_traj["fuzzy_main_axis_support_uncertainty_rad"] = {
    "dtype": "<f8",
    "comment": "",
    "unit": "rad",
}

_traj["fuzzy_main_axis_azimuth_rad"] = {
    "dtype": "<f8",
    "comment": "",
    "unit": "rad",
}
_traj["fuzzy_main_axis_azimuth_uncertainty_rad"] = {
    "dtype": "<f8",
    "comment": "",
    "unit": "rad",
}

STRUCTURE["reconstructed_trajectory"] = _traj
