import collections
import numpy as np
import dynamicsizerecarray


def init_table_structure():
    t = collections.OrderedDict()
    t["primary"] = init_primary_level_structure()
    t["pointing"] = init_pointing_level_structure()

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


def to_dtype(level_structure, include_index=True):
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


def init_table_dynamicsizerecarray(table_structure=None):
    if table_structure is None:
        table_structure = init_table_structure()
    out = {}
    for level in table_structure:
        out[level] = dynamicsizerecarray.DynamicSizeRecarray(
            dtype=to_dtype(level_structure=table_structure[level]),
        )
    return out


def init_primary_level_structure():
    t = collections.OrderedDict()
    t["particle_id"] = {"dtype": "<i8", "comment": "CORSIKA particle-id"}
    t["energy_GeV"] = {"dtype": "<f8", "comment": ""}
    t["azimuth_rad"] = {
        "dtype": "<f8",
        "comment": "Direction of the primary particle w.r.t. magnetic north.",
    }
    t["zenith_rad"] = {
        "dtype": "<f8",
        "comment": "Direction of the primary particle.",
    }
    t["depth_g_per_cm2"] = {"dtype": "<f8", "comment": ""}

    t["momentum_x_GeV_per_c"] = {"dtype": "<f8", "comment": ""}
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


def init_pointing_level_structure():
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
    t = {}
    t["bin_idx_x"] = {"dtype": "<i8", "comment": ""}
    t["bin_idx_y"] = {"dtype": "<i8", "comment": ""}
    t["core_x_m"] = {"dtype": "<f8", "comment": ""}
    t["core_y_m"] = {"dtype": "<f8", "comment": ""}
    return t


def init_particlepool_level_structure():
    t = {}
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
    t = {}
    t["num_air_cherenkov_on_aperture"] = {
        "dtype": "<i8",
        "comment": "Same as 'num_air_cherenkov' but also run through the "
        "instrument's aperture for Cherenkov-light.",
    }
    return t


def init_instrument_level_structure():
    t = {}
    t["start_time_of_exposure_s"] = {
        "dtype": "<f8",
        "comment": "The start-time of the instrument's exposure-window"
        "relative to the clock in CORSIKA.",
    }
    return t


def init_trigger_level_structure(num_foci=12):
    t = {}
    t["num_cherenkov_pe"] = {"dtype": "<i8", "comment": ""}
    t["response_pe"] = {"dtype": "<i8", "comment": ""}
    for nn in range(num_foci):
        key = "focus_{:02d}_response_pe".format(nn)
        t[key] = {"dtype": "<i8", "comment": ""}
    return t


def init_pasttrigger_level_structure():
    return {}  # only the event uid.


"""
STRUCTURE["grid"] = {
    "num_bins_thrown": {
        "dtype": "<i8",
        "comment": "The number of all grid-bins which can collect "
        "Cherenkov-photons.",
    },
    "bin_width_m": {"dtype": "<f8", "comment": ""},
    "field_of_view_radius_deg": {"dtype": "<f8", "comment": ""},
    "field_of_view_azimuth_deg": {
        "dtype": "<f8",
        "comment": "Pointing azimuth w.r.t. magnetic north.",
    },
    "field_of_view_zenith_deg": {
        "dtype": "<f8",
        "comment": "Pointing zenith-distance.",
    },
    "pointing_direction_x": {"dtype": "<f8", "comment": ""},
    "pointing_direction_y": {"dtype": "<f8", "comment": ""},
    "pointing_direction_z": {"dtype": "<f8", "comment": ""},
    "random_shift_x_m": {"dtype": "<f8", "comment": ""},
    "random_shift_y_m": {"dtype": "<f8", "comment": ""},
    "magnet_shift_x_m": {"dtype": "<f8", "comment": ""},
    "magnet_shift_y_m": {"dtype": "<f8", "comment": ""},
    "total_shift_x_m": {
        "dtype": "<f8",
        "comment": "Sum of random and magnetic shift.",
    },
    "total_shift_y_m": {
        "dtype": "<f8",
        "comment": "Sum of random and magnetic shift.",
    },
    "num_bins_above_threshold": {"dtype": "<i8", "comment": ""},
    "overflow_x": {"dtype": "<i8", "comment": ""},
    "overflow_y": {"dtype": "<i8", "comment": ""},
    "underflow_x": {"dtype": "<i8", "comment": ""},
    "underflow_y": {"dtype": "<i8", "comment": ""},
    "area_thrown_m2": {"dtype": "<f8", "comment": ""},
    "artificial_core_limitation": {"dtype": "<i8", "comment": "Flag"},
    "artificial_core_limitation_radius_m": {"dtype": "<f8", "comment": ""},
}

STRUCTURE["cherenkovpool"] = {
    "maximum_asl_m": {"dtype": "<f8", "comment": ""},
    "wavelength_median_nm": {"dtype": "<f8", "comment": ""},
    "cx_median_rad": {"dtype": "<f8", "comment": ""},
    "cy_median_rad": {"dtype": "<f8", "comment": ""},
    "x_median_m": {"dtype": "<f8", "comment": ""},
    "y_median_m": {"dtype": "<f8", "comment": ""},
    "bunch_size_median": {"dtype": "<f8", "comment": ""},
}

"""
