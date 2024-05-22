import collections
import numpy as np


def init_event_table_structure():
    t = collections.OrderedDict()
    t["primary"] = init_primary_level_structure()
    t["instrument_pointing"] = init_instrument_pointing_level_structure()

    t["cherenkovsize"] = init_cherenkovsize_level_structure()
    t["cherenkovpool"] = init_cherenkovpool_level_structure()

    t["groundgrid"] = init_groundgrid_level_structure()

    t["cherenkovsizepart"] = init_cherenkovsizepart_level_structure()
    t["cherenkovpoolpart"] = init_cherenkovpoolpart_level_structure()

    t["core"] = init_core_level_structure()

    t["particlepool"] = init_particlepool_level_structure()
    t["particlepoolonaperture"] = init_particlepoolonaperture_level_structure()

    t["instrument"] = init_instrument_level_structure()
    t["trigger"] = init_trigger_level_structure()
    t["pasttrigger"] = init_pasttrigger_level_structure()

    return t


def dtypes(table_structure=None, include_index=False):
    if table_structure is None:
        table_structure = init_event_table_structure()

    dtypes = {}
    for level_key in table_structure:
        dtypes[level_key] = level_structure_to_dtype(
            level_structure=table_structure[level_key],
            include_index=include_index,
        )
    return dtypes


def level_structure_to_dtype(level_structure, include_index=True):
    """
    Returns a list of [(str, str), (str, str), ... ] to
    initialize a numpy recarray.

    Parameters
    ----------
    level_structure : {key1: {"dtype": dtype1}, key2: { ... }}
        A level_structure of the table. E.g. 'primary'.

    Returns
    -------
    dtype : [(key1, dtype1), (key2, dtype2), ... ]
        The dtype.
    """
    out = []
    if include_index:
        out.append(("idx", "<u8"))
    for key in level_structure:
        out.append((key, level_structure[key]["dtype"]))
    return out


def init_primary_level_structure():
    t = collections.OrderedDict()
    t["particle_id"] = {
        "dtype": "<i8",
        "comment": "CORSIKA particle-id to identify the primary particle.",
    }
    t["energy_GeV"] = {
        "dtype": "<f8",
        "comment": "Energy of primary particle.",
    }
    t["theta_rad"] = {
        "dtype": "<f8",
        "comment": "Direction of primary particle's momentum. "
        "CORSIKA's zenith distance angle of the primary particle's momentum.",
    }
    t["phi_rad"] = {
        "dtype": "<f8",
        "comment": "Direction of primary particle's momentum. "
        "CORSIKA's azimuth angle w.r.t. to magnetic north of the "
        "primary particle's momentum.",
    }
    t["azimuth_rad"] = {
        "dtype": "<f8",
        "comment": "Pointing direction of the primary particle w.r.t. "
        "magnetic north.",
    }
    t["zenith_rad"] = {
        "dtype": "<f8",
        "comment": "Pointing direction of the primary particle.",
    }
    t["depth_g_per_cm2"] = {
        "dtype": "<f8",
        "comment": "Starting depth of CORSIKA simulation.",
    }

    t["momentum_x_GeV_per_c"] = {
        "dtype": "<f8",
        "comment": "Taken from CORSIKA's eventheader (EVTH).",
    }
    t["momentum_y_GeV_per_c"] = {"dtype": "<f8", "comment": ""}
    t["momentum_z_GeV_per_c"] = {"dtype": "<f8", "comment": ""}

    t["starting_height_asl_m"] = {
        "dtype": "<f8",
        "comment": "The simulation of the primary particle " "starts here.",
    }
    t["starting_x_m"] = {
        "dtype": "<f8",
        "comment": "See starting_height_asl_m.",
    }
    t["starting_y_m"] = {
        "dtype": "<f8",
        "comment": "See starting_height_asl_m.",
    }

    t["first_interaction_height_asl_m"] = {
        "dtype": "<f8",
        "comment": "See CORSIKA, this is not very "
        "meaningfull for e.g. electrons.",
    }

    t["solid_angle_thrown_sr"] = {
        "dtype": "<f8",
        "comment": "The size of the solid angle from which the direction "
        "of the primary particle is drawn from.",
    }
    t["inner_atmopsheric_magnetic_cutoff"] = {
        "dtype": "<i8",
        "comment": "A boolean flag (0, 1). If 1, the tables for magnetic "
        "deflection were not able to predict the direction of the "
        "primary particle. This is a strong indicator for there is "
        "no primary direction which can produce Cherenkov light at a "
        "given direction.",
    }
    return t


def init_instrument_pointing_level_structure():
    t = collections.OrderedDict()
    t["azimuth_rad"] = {
        "dtype": "<f8",
        "comment": "Azimuth direction of the instrument's optical axis "
        "w.r.t. magnetic north.",
    }
    t["zenith_rad"] = {
        "dtype": "<f8",
        "comment": "Zenith direction of the instrument's optical axis "
        "w.r.t. magnetic north.",
    }
    return t


def init_cherenkovsize_level_structure():
    t = collections.OrderedDict()
    t["num_bunches"] = {"dtype": "<i8", "comment": ""}
    t["num_photons"] = {"dtype": "<f8", "comment": ""}
    return t


def init_cherenkovpool_level_structure():
    t = collections.OrderedDict()
    keys = {}
    keys["cx"] = "1"
    keys["cy"] = "1"
    keys["x"] = "m"
    keys["y"] = "m"
    keys["z_emission"] = "m"
    keys["wavelength"] = "m"
    keys["bunch_size"] = "1"

    for key in keys:
        for percentile in [16, 50, 84]:
            tkey = "{:s}_p{:02d}_{:s}".format(key, percentile, keys[key])
            t[tkey] = {
                "dtype": "<f8",
                "comment": "percentile {:d}".format(percentile),
            }
    return t


def init_groundgrid_level_structure():
    t = collections.OrderedDict()
    # arguments to init GroundGrid
    t["bin_width_m"] = {"dtype": "<f8", "comment": ""}
    t["num_bins_each_axis"] = {"dtype": "<i8", "comment": ""}

    t["center_x_m"] = {
        "dtype": "<f8",
        "comment": "This is random_shift_x_m + cherenkov_pool_median_x_m.",
    }
    t["center_y_m"] = {"dtype": "<f8", "comment": "See center_x_m."}

    t["cherenkov_pool_median_x_m"] = {
        "dtype": "<f8",
        "comment": "The median impact of Cherenkov bunches in 'x'."
        "If there are Cherenkov buncehs, this is taken from "
        "cherenkovpool.x_median_m. If not, this is zero.",
    }
    t["cherenkov_pool_median_y_m"] = {
        "dtype": "<f8",
        "comment": "See cherenkov_pool_median_x_m.",
    }

    t["random_shift_x_m"] = {
        "dtype": "<f8",
        "comment": "A random shift of the grid by plus or "
        "minus half the bin_width_m.",
    }
    t["random_shift_y_m"] = {"dtype": "<f8", "comment": ""}

    t["num_bins_thrown"] = {
        "dtype": "<i8",
        "comment": "The number of all grid-bins which can collect "
        "Cherenkov-photons.",
    }
    t["num_bins_above_threshold"] = {"dtype": "<i8", "comment": ""}
    t["area_thrown_m2"] = {"dtype": "<f8", "comment": ""}

    t["num_photons_overflow"] = {
        "dtype": "<i8",
        "comment": "Number of Cherenkov bunches which did not hit "
        "any bins in the grid.",
    }

    # compare scatter
    num_scatter_bins = 16
    for rbin in range(num_scatter_bins):
        t["scatter_rbin_{:02d}".format(rbin)] = {
            "dtype": "<u4",
            "comment": "Number of bins above threshold and within "
            "a certain range of scatter radii.",
        }
    return t


def init_cherenkovsizepart_level_structure():
    return init_cherenkovsize_level_structure()


def init_cherenkovpoolpart_level_structure():
    return init_cherenkovpool_level_structure()


def init_core_level_structure():
    t = collections.OrderedDict()
    t["bin_idx_x"] = {"dtype": "<i8", "comment": ""}
    t["bin_idx_y"] = {"dtype": "<i8", "comment": ""}
    t["core_x_m"] = {"dtype": "<f8", "comment": ""}
    t["core_y_m"] = {"dtype": "<f8", "comment": ""}
    return t


def init_particlepool_level_structure():
    t = collections.OrderedDict()
    t["num_water_cherenkov"] = {
        "dtype": "<i8",
        "comment": "The number of particles which reach the observation-level "
        "and will emitt Cherenkov-light in water",
    }
    t["num_air_cherenkov"] = {
        "dtype": "<i8",
        "comment": "Same as 'num_water_cherenkov' but for the air at the "
        "instruments altitude.",
    }
    t["num_unknown"] = {
        "dtype": "<i8",
        "comment": "Particles which are not (yet) in our"
        "corsika-particle-zoo.",
    }
    return t


def init_particlepoolonaperture_level_structure():
    t = collections.OrderedDict()
    t["num_air_cherenkov_on_aperture"] = {
        "dtype": "<i8",
        "comment": "Same as 'num_air_cherenkov' but also run through the "
        "instrument's aperture for Cherenkov-light.",
    }
    return t


def init_instrument_level_structure():
    t = collections.OrderedDict()
    t["start_time_of_exposure_s"] = {
        "dtype": "<f8",
        "comment": "The start-time of the instrument's exposure-window"
        "relative to the clock in CORSIKA.",
    }
    return t


def init_trigger_level_structure(num_foci=12):
    t = collections.OrderedDict()
    t["num_cherenkov_pe"] = {"dtype": "<i8", "comment": ""}
    t["response_pe"] = {"dtype": "<i8", "comment": ""}
    for nn in range(num_foci):
        key = "focus_{:02d}_response_pe".format(nn)
        t[key] = {"dtype": "<i8", "comment": ""}
    return t


def init_pasttrigger_level_structure():
    return collections.OrderedDict()  # only the event uid.


def init_cherenkovclassification_level_structure():
    t = collections.OrderedDict()
    t["num_true_positives"] = {"dtype": "<i8", "comment": ""}
    t["num_false_negatives"] = {"dtype": "<i8", "comment": ""}
    t["num_false_positives"] = {"dtype": "<i8", "comment": ""}
    t["num_true_negatives"] = {"dtype": "<i8", "comment": ""}
    return t


def init_features_level_structure():
    t = collections.OrderedDict()
    t["num_photons"] = {
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
    t["paxel_intensity_peakness_std_over_mean"] = {
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
    t["paxel_intensity_peakness_max_over_mean"] = {
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
    t["paxel_intensity_median_x"] = {
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
    t["paxel_intensity_median_y"] = {
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
    t["aperture_num_islands_watershed_rel_thr_2"] = {
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
    t["aperture_num_islands_watershed_rel_thr_4"] = {
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
    t["aperture_num_islands_watershed_rel_thr_8"] = {
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
    t["light_front_cx"] = {
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
    t["light_front_cy"] = {
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
    t["image_infinity_cx_mean"] = {
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
    t["image_infinity_cy_mean"] = {
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
    t["image_infinity_cx_std"] = {
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
    t["image_infinity_cy_std"] = {
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
    t["image_infinity_num_photons_on_edge_field_of_view"] = {
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
    t["image_smallest_ellipse_object_distance"] = {
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
    t["image_smallest_ellipse_solid_angle"] = {
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
    t["image_smallest_ellipse_half_depth"] = {
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
    t["image_half_depth_shift_cx"] = {
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
    t["image_half_depth_shift_cy"] = {
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
    t["image_smallest_ellipse_num_photons_on_edge_field_of_view"] = {
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
    t["image_num_islands"] = {
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
    return t


def init_reconstructed_trajectory_level_structure():
    t = collections.OrderedDict()

    xy_comment = (
        "Primary particle's core-position w.r.t. principal aperture-plane."
    )
    t["x_m"] = {
        "dtype": "<f8",
        "comment": xy_comment,
        "unit": "m",
    }
    t["y_m"] = {
        "dtype": "<f8",
        "comment": xy_comment,
        "unit": "m",
    }

    cxy_comment = "Primary particle's direction w.r.t. pointing."
    t["cx_rad"] = {
        "dtype": "<f8",
        "comment": cxy_comment,
        "unit": "rad",
    }
    t["cy_rad"] = {
        "dtype": "<f8",
        "comment": cxy_comment,
        "unit": "rad",
    }

    cxy_comment_fuzzy = (
        "Primary particle's direction w.r.t. "
        + "pointing according to fuzzy-estimator."
    )
    t["fuzzy_cx_rad"] = {
        "dtype": "<f8",
        "comment": cxy_comment_fuzzy,
        "unit": "rad",
    }
    t["fuzzy_cy_rad"] = {
        "dtype": "<f8",
        "comment": cxy_comment_fuzzy,
        "unit": "rad",
    }

    t["fuzzy_main_axis_support_cx_rad"] = {
        "dtype": "<f8",
        "comment": "",
        "unit": "rad",
    }
    t["fuzzy_main_axis_support_cy_rad"] = {
        "dtype": "<f8",
        "comment": "",
        "unit": "rad",
    }
    t["fuzzy_main_axis_support_uncertainty_rad"] = {
        "dtype": "<f8",
        "comment": "",
        "unit": "rad",
    }

    t["fuzzy_main_axis_azimuth_rad"] = {
        "dtype": "<f8",
        "comment": "",
        "unit": "rad",
    }
    t["fuzzy_main_axis_azimuth_uncertainty_rad"] = {
        "dtype": "<f8",
        "comment": "",
        "unit": "rad",
    }
    return t
