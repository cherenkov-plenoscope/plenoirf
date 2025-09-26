import numpy as np
from . import default


def generate_paxel_intensity_median_hypot_x_y(event_frame):
    return np.hypot(
        event_frame["features/paxel_intensity_median_x"],
        event_frame["features/paxel_intensity_median_y"],
    )


def generate_image_half_depth_shift_hypot_cx_cy(event_frame):
    return np.hypot(
        event_frame["features/image_half_depth_shift_cx"],
        event_frame["features/image_half_depth_shift_cy"],
    )


def generate_diff_image_and_light_front(event_frame):
    ev = event_frame
    dcx = ev["features/image_infinity_cx_mean"] - ev["features/light_front_cx"]
    dcy = ev["features/image_infinity_cy_mean"] - ev["features/light_front_cy"]
    return np.hypot(dcx, dcy)


def generate_diff_image_and_trajectory_reconstruction(event_frame):
    ev = event_frame
    dcx = (
        ev["features/image_infinity_cx_mean"]
        - ev["reconstructed_trajectory/cx_rad"]
    )
    dcy = (
        ev["features/image_infinity_cy_mean"]
        - ev["reconstructed_trajectory/cy_rad"]
    )
    return np.hypot(dcx, dcy)


def generate_diff_light_front_and_trajectory_reconstruction(event_frame):
    ev = event_frame
    dcx = ev["reconstructed_trajectory/cx_rad"] - ev["features/light_front_cx"]
    dcy = ev["reconstructed_trajectory/cy_rad"] - ev["features/light_front_cy"]
    return np.hypot(dcx, dcy)


def generate_A(event_frame):
    ev = event_frame
    shift = np.hypot(
        ev["features/image_half_depth_shift_cx"],
        ev["features/image_half_depth_shift_cy"],
    )
    return (
        np.log10(ev["features/num_photons"])
        * shift
        / ev["features/image_smallest_ellipse_half_depth"]
    )


def generate_B(event_frame):
    ev = event_frame
    return (
        np.log10(ev["features/num_photons"])
        / ev["features/image_smallest_ellipse_object_distance"] ** 2.0
    )


def generate_C(event_frame):
    ev = event_frame
    return (
        ev["features/image_smallest_ellipse_half_depth"]
        * ev["features/image_smallest_ellipse_object_distance"]
    )


def _shower_volume(shower_sr, half_depth, object_distance):
    full_sphere_sr = 4 * np.pi
    shower_fraction_sphere = shower_sr / full_sphere_sr
    area_of_focus_sphere_m2 = np.pi * object_distance**2
    shower_area_m2 = shower_fraction_sphere * area_of_focus_sphere_m2
    shower_volume_m3 = shower_area_m2 * half_depth
    return shower_volume_m3


def genetate_shower_volume(event_frame):
    ev = event_frame
    return _shower_volume(
        shower_sr=ev["features/image_smallest_ellipse_solid_angle"],
        half_depth=ev["features/image_smallest_ellipse_half_depth"],
        object_distance=ev["features/image_smallest_ellipse_object_distance"],
    )


def generate_shower_shift_volume(event_frame):
    ev = event_frame
    shower_sr = np.abs(ev["features/image_half_depth_shift_cx"]) * np.abs(
        ev["features/image_half_depth_shift_cy"]
    )
    return _shower_volume(
        shower_sr=shower_sr,
        half_depth=ev["features/image_smallest_ellipse_half_depth"],
        object_distance=ev["features/image_smallest_ellipse_object_distance"],
    )


def genetate_shower_density(event_frame):
    shower_volume_m3 = genetate_shower_volume(event_frame=event_frame)
    return event_frame["features/num_photons"] / shower_volume_m3


def generate_reconstructed_trajectory_hypot_x_y(event_frame):
    ev = event_frame
    return np.hypot(
        ev["reconstructed_trajectory/x_m"],
        ev["reconstructed_trajectory/y_m"],
    )


def generate_reconstructed_trajectory_hypot_cx_cy(event_frame):
    ev = event_frame
    return np.hypot(
        ev["reconstructed_trajectory/cx_rad"],
        ev["reconstructed_trajectory/cy_rad"],
    )


def generate_paxel_intensity_peakness_mean_over_std(event_frame):
    return 1.0 / event_frame["features/paxel_intensity_peakness_std_over_mean"]


def generate_reconstructed_trajectory_x_m(event_frame):
    return event_frame["reconstructed_trajectory/x_m"]


def generate_reconstructed_trajectory_y_m(event_frame):
    return event_frame["reconstructed_trajectory/y_m"]


def generate_reconstructed_trajectory_cx_rad(event_frame):
    return event_frame["reconstructed_trajectory/cx_rad"]


def generate_reconstructed_trajectory_cy_rad(event_frame):
    return event_frame["reconstructed_trajectory/cy_rad"]


def init_combined_features_structure():
    fx_median = default.transformation_fx_median()
    fx_containment_percentile_90 = (
        default.transformation_fx_containment_percentile_90()
    )

    out = {}

    # reconstructed_trajectory
    # ------------------------
    out["reconstructed_trajectory_x_m"] = {
        "generator": generate_reconstructed_trajectory_x_m,
        "dtype": "<f4",
        "unit": "1",
        "transformation": {
            "function": "log10(abs(x)) * sign(x)",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }

    out["reconstructed_trajectory_y_m"] = {
        "generator": generate_reconstructed_trajectory_y_m,
        "dtype": "<f4",
        "unit": "1",
        "transformation": {
            "function": "log10(abs(x)) * sign(x)",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }

    out["reconstructed_trajectory_cx_rad"] = {
        "generator": generate_reconstructed_trajectory_cx_rad,
        "dtype": "<f4",
        "unit": "1",
        "transformation": {
            "function": "x",
            "shift": "0",
            "scale": fx_containment_percentile_90,
        },
    }

    out["reconstructed_trajectory_cy_rad"] = {
        "generator": generate_reconstructed_trajectory_cy_rad,
        "dtype": "<f4",
        "unit": "1",
        "transformation": {
            "function": "x",
            "shift": "0",
            "scale": fx_containment_percentile_90,
        },
    }

    out["reconstructed_trajectory_hypot_x_y"] = {
        "generator": generate_reconstructed_trajectory_hypot_x_y,
        "dtype": "<f4",
        "unit": "m",
        "transformation": {
            "function": "log10(x)",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }

    out["reconstructed_trajectory_hypot_cx_cy"] = {
        "generator": generate_reconstructed_trajectory_hypot_cx_cy,
        "dtype": "<f4",
        "unit": "m",
        "transformation": {
            "function": "x",
            "shift": "0",
            "scale": fx_containment_percentile_90,
        },
    }

    out["paxel_intensity_peakness_mean_over_std"] = {
        "generator": generate_paxel_intensity_peakness_mean_over_std,
        "dtype": "<f4",
        "unit": "1",
        "transformation": {
            "function": "log(x)",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }

    out["image_half_depth_shift_hypot_cx_cy"] = {
        "generator": generate_image_half_depth_shift_hypot_cx_cy,
        "dtype": "<f4",
        "unit": "rad",
        "transformation": {
            "function": "log10(x)**(-3)",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }

    out["combi_diff_image_and_light_front"] = {
        "generator": generate_diff_image_and_light_front,
        "dtype": "<f4",
        "unit": "rad",
        "transformation": {
            "function": "log(x)",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }
    out["combi_diff_image_and_trajectory_reconstruction"] = {
        "generator": generate_diff_image_and_trajectory_reconstruction,
        "dtype": "<f4",
        "unit": "rad",
        "transformation": {
            "function": "log(x)",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }
    out["combi_diff_light_front_and_trajectory_reconstruction"] = {
        "generator": generate_diff_light_front_and_trajectory_reconstruction,
        "dtype": "<f4",
        "unit": "rad",
        "transformation": {
            "function": "log(x)",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }
    out["combi_paxel_intensity_median_hypot_x_y"] = {
        "generator": generate_paxel_intensity_median_hypot_x_y,
        "dtype": "<f4",
        "unit": "$m$",
        "transformation": {
            "function": "x",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }
    out["combi_A"] = {
        "generator": generate_A,
        "dtype": "<f4",
        "unit": "$sr m^{-1}$",
        "transformation": {
            "function": "log(x)**(-2)",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }
    out["combi_B"] = {
        "generator": generate_B,
        "dtype": "<f4",
        "unit": "$m^{-2}$",
        "transformation": {
            "function": "log(x)**(-2)",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }
    out["combi_C"] = {
        "generator": generate_C,
        "dtype": "<f4",
        "unit": "$1$",
        "transformation": {
            "function": "log(x)",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }

    out["combi_shower_volume"] = {
        "generator": genetate_shower_volume,
        "dtype": "<f4",
        "unit": "$m{^3}$",
        "transformation": {
            "function": "log(x)**(-3)",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }

    out["combi_shower_shift_volume"] = {
        "generator": generate_shower_shift_volume,
        "dtype": "<f4",
        "unit": "$m{^3}$",
        "transformation": {
            "function": "log(x**(1/2))",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }

    out["combi_shower_density"] = {
        "generator": genetate_shower_density,
        "dtype": "<f4",
        "unit": "$m{^-3}$",
        "transformation": {
            "function": "log(x)",
            "shift": fx_median,
            "scale": fx_containment_percentile_90,
        },
    }
    return out
