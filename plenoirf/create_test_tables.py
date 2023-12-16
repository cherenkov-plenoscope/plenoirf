import numpy as np
import sparse_numeric_table as spt
import corsika_primary as cpw
import pandas
import solid_angle_utils

from . import table
from . import utils


cfg = {
    "particle_id": 14,
    "energy": {
        "lower": 0.25,
        "upper": 1e3,
        "power_slope": -2.0,
    },
    "num_airshower": 1000 * 1000,
    "propagation_probability": {
        "one_grid_cell_over_threshold_to_trigger": 0.025,
    },
    "magnetic_deflection": {
        "energy_GeV": [1, 10, 100, 1000],
        "primary_azimuth_deg": [90, 25, 5, 0],
        "primary_zenith_deg": [16, 4, 1, 0],
        "cherenkov_pool_x_m": [1e4, 1e3, 1e2, 1e1],
        "cherenkov_pool_y_m": [2e4, 2e3, 2e2, 2e1],
    },
    "max_scatter_rad": np.deg2rad(13.0),
    "grid": {
        "num_bins_thrown": 1024 * 1024,
        "bin_width_m": 80,
        "field_of_view_radius_deg": 3.25 * 1.1,
        "num_bins_above_threshold": 10
        * np.array([1, 2, 4, 8, 4, 2, 1, 4, 16, 64, 256]),
        "energy_GeV": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    },
}


def cut_level(level, mask):
    num = np.sum(mask)
    lvl = {}
    lvl["num"] = num
    lvl["energy_GeV"] = level["energy_GeV"][mask]
    lvl["idx"] = level["idx"][mask]
    lvl["ones"] = np.ones(num)
    lvl["magnet_azimuth_rad"] = level["magnet_azimuth_rad"][mask]
    lvl["magnet_zenith_rad"] = level["magnet_zenith_rad"][mask]
    lvl["magnet_cherenkov_pool_x_m"] = level["magnet_cherenkov_pool_x_m"][mask]
    lvl["magnet_cherenkov_pool_y_m"] = level["magnet_cherenkov_pool_y_m"][mask]
    return lvl


"""
primary
cherenkovsize
grid
cherenkovpool

    cherenkovsizepart
    cherenkovpoolpart
    core
    trigger

        pasttrigger

            cherenkovclassification

                features

"""


def create_dummy_table(
    prng,
    config=cfg,
    num_primary=1000,
    structrue=table.STRUCTURE,
):
    ones = np.ones(num_primary)
    t = {}

    # primary
    # -------
    _energies = cpw.random.distributions.draw_power_law(
        prng=prng,
        lower_limit=config["energy"]["lower"],
        upper_limit=config["energy"]["upper"],
        power_slope=config["energy"]["power_slope"],
        num_samples=num_primary,
    )

    lvl = {}
    lvl["num"] = num_primary
    lvl["energy_GeV"] = _energies
    lvl["idx"] = np.arange(num_primary)
    lvl["ones"] = np.ones(num_primary)
    lvl["magnet_azimuth_rad"] = np.deg2rad(
        np.interp(
            x=lvl["energy_GeV"],
            xp=config["magnetic_deflection"]["energy_GeV"],
            fp=config["magnetic_deflection"]["primary_azimuth_deg"],
        )
    )
    lvl["magnet_zenith_rad"] = np.deg2rad(
        np.interp(
            x=lvl["energy_GeV"],
            xp=config["magnetic_deflection"]["energy_GeV"],
            fp=config["magnetic_deflection"]["primary_zenith_deg"],
        )
    )
    lvl["magnet_cherenkov_pool_x_m"] = np.interp(
        x=lvl["energy_GeV"],
        xp=config["magnetic_deflection"]["energy_GeV"],
        fp=config["magnetic_deflection"]["cherenkov_pool_x_m"],
    )
    lvl["magnet_cherenkov_pool_y_m"] = np.interp(
        x=lvl["energy_GeV"],
        xp=config["magnetic_deflection"]["energy_GeV"],
        fp=config["magnetic_deflection"]["cherenkov_pool_y_m"],
    )

    primary = {}
    primary[spt.IDX] = lvl["idx"]
    primary["particle_id"] = config["particle_id"] * lvl["ones"]
    primary["energy_GeV"] = lvl["energy_GeV"]
    primary["magnet_azimuth_rad"] = lvl["magnet_azimuth_rad"]
    primary["magnet_zenith_rad"] = lvl["magnet_zenith_rad"]
    primary["magnet_cherenkov_pool_x_m"] = lvl["magnet_cherenkov_pool_x_m"]
    primary["magnet_cherenkov_pool_y_m"] = lvl["magnet_cherenkov_pool_y_m"]

    _az = []
    _zd = []
    for i in range(lvl["num"]):
        _az_, _zd_ = cpw.random.distributions.draw_azimuth_zenith_in_viewcone(
            prng=prng,
            azimuth_rad=primary["magnet_azimuth_rad"][i],
            zenith_rad=primary["magnet_zenith_rad"][i],
            min_scatter_opening_angle_rad=0.0,
            max_scatter_opening_angle_rad=config["max_scatter_rad"],
        )
        _az.append(_az_)
        _zd.append(_zd_)
    _az = np.array(_az)
    _zd = np.array(_zd)

    primary["zenith_rad"] = _zd
    primary["azimuth_rad"] = _az

    primary["max_scatter_rad"] = config["max_scatter_rad"]
    primary["solid_angle_thrown_sr"] = solid_angle_utils.cone.solid_angle(
        half_angle_rad=primary["max_scatter_rad"]
    )

    primary["depth_g_per_cm2"] = 0.0 * lvl["ones"]

    primary["momentum_x_GeV_per_c"] = 0.0 * lvl["ones"]
    primary["momentum_y_GeV_per_c"] = 0.0 * lvl["ones"]
    primary["momentum_z_GeV_per_c"] = -1.0 * lvl["energy_GeV"]

    primary["first_interaction_height_asl_m"] = prng.uniform(
        low=1e4, high=1e5, size=lvl["num"]
    )
    primary["starting_height_asl_m"] = 120.0e3 * lvl["ones"]

    primary["starting_x_m"] = 0.0 * lvl["ones"]
    primary["starting_y_m"] = 0.0 * lvl["ones"]

    t["primary"] = primary

    # cherenkovsize
    # -------------
    cherenkovsize = {}
    cherenkovsize[spt.IDX] = lvl["idx"]
    cherenkovsize["num_bunches"] = 1e3 * lvl["energy_GeV"] + prng.normal(
        loc=0.0, scale=3e2, size=lvl["num"]
    )
    cherenkovsize["num_bunches"][cherenkovsize["num_bunches"] < 0] = 0
    cherenkovsize["num_bunches"] = cherenkovsize["num_bunches"].astype(
        np.uint64
    )
    cherenkovsize["num_photons"] = 0.97 * cherenkovsize["num_bunches"]
    t["cherenkovsize"] = cherenkovsize

    _plvl = cut_level(level=lvl, mask=cherenkovsize["num_bunches"] > 0)

    # cherenkovpool
    # -------------
    cherenkovpool = {}
    cherenkovpool[spt.IDX] = _plvl["idx"]
    cherenkovpool["maximum_asl_m"] = prng.normal(
        loc=1e4, scale=3e3, size=_plvl["num"]
    )
    cherenkovpool["wavelength_median_nm"] = prng.normal(
        loc=433.0, scale=10.0, size=_plvl["num"]
    )
    cherenkovpool["cx_median_rad"] = prng.normal(
        loc=0.0, scale=config["max_scatter_rad"], size=_plvl["num"]
    )
    cherenkovpool["cy_median_rad"] = prng.normal(
        loc=0.0, scale=config["max_scatter_rad"], size=_plvl["num"]
    )
    cherenkovpool["x_median_m"] = _plvl["magnet_cherenkov_pool_x_m"] + (
        1e3 / _plvl["energy_GeV"]
    ) * prng.normal(loc=0.0, scale=1.0, size=_plvl["num"])
    cherenkovpool["y_median_m"] = _plvl["magnet_cherenkov_pool_y_m"] + (
        1e3 / _plvl["energy_GeV"]
    ) * prng.normal(loc=0.0, scale=1.0, size=_plvl["num"])
    cherenkovpool["bunch_size_median"] = prng.uniform(
        low=0.9, high=1.0, size=_plvl["num"]
    )
    t["cherenkovpool"] = cherenkovpool

    # grid
    # ----
    grid = {}
    grid[spt.IDX] = lvl["idx"]
    grid["num_bins_thrown"] = config["grid"]["num_bins_thrown"] * lvl["ones"]
    grid["bin_width_m"] = config["grid"]["bin_width_m"] * lvl["ones"]

    grid["field_of_view_radius_deg"] = (
        config["grid"]["field_of_view_radius_deg"] * lvl["ones"]
    )
    grid["pointing_direction_x"] = 0.0 * lvl["ones"]
    grid["pointing_direction_y"] = 0.0 * lvl["ones"]
    grid["pointing_direction_z"] = 1.0 * lvl["ones"]
    grid["random_shift_x_m"] = prng.uniform(
        low=-0.5 * config["grid"]["bin_width_m"],
        high=0.5 * config["grid"]["bin_width_m"],
        size=lvl["num"],
    )
    grid["random_shift_y_m"] = prng.uniform(
        low=-0.5 * config["grid"]["bin_width_m"],
        high=0.5 * config["grid"]["bin_width_m"],
        size=lvl["num"],
    )
    grid["magnet_shift_x_m"] = -1.0 * lvl["magnet_cherenkov_pool_x_m"]
    grid["magnet_shift_y_m"] = -1.0 * lvl["magnet_cherenkov_pool_y_m"]
    grid["total_shift_x_m"] = (
        grid["random_shift_x_m"] + grid["magnet_shift_x_m"]
    )
    grid["total_shift_y_m"] = (
        grid["random_shift_y_m"] + grid["magnet_shift_y_m"]
    )
    grid["num_bins_above_threshold"] = np.round(
        np.interp(
            x=lvl["energy_GeV"],
            xp=config["grid"]["energy_GeV"],
            fp=config["grid"]["num_bins_above_threshold"],
        )
        + prng.normal(loc=0.0, scale=5.0, size=lvl["num"])
    )
    grid["num_bins_above_threshold"][grid["num_bins_above_threshold"] < 0] = 0
    grid["num_bins_above_threshold"] = grid["num_bins_above_threshold"].astype(
        np.uint64
    )

    grid["overflow_x"] = 0 * lvl["ones"]
    grid["overflow_y"] = 0 * lvl["ones"]
    grid["underflow_x"] = 0 * lvl["ones"]
    grid["underflow_y"] = 0 * lvl["ones"]

    grid["area_thrown_m2"] = (
        config["grid"]["bin_width_m"] ** 2
        * config["grid"]["num_bins_thrown"]
        * lvl["ones"]
    )
    grid["artificial_core_limitation"] = 0 * lvl["ones"]
    grid["artificial_core_limitation_radius_m"] = 0.0 * lvl["ones"]
    t["grid"] = grid

    _glvl = cut_level(level=lvl, mask=grid["num_bins_above_threshold"] > 0)

    # cherenkovsizepart
    # -------------
    cherenkovsizepart = {}
    cherenkovsizepart[spt.IDX] = _glvl["idx"]
    cherenkovsizepart["num_bunches"] = _glvl["energy_GeV"] * prng.normal(
        loc=10.0, scale=10.0, size=_glvl["num"]
    )
    cherenkovsizepart["num_bunches"][cherenkovsizepart["num_bunches"] < 0] = 0
    cherenkovsizepart["num_bunches"] = cherenkovsizepart["num_bunches"].astype(
        np.uint64
    )

    cherenkovsizepart["num_photons"] = 0.97 * cherenkovsizepart["num_bunches"]

    t["cherenkovsizepart"] = cherenkovsizepart

    _pplvl = cut_level(level=_glvl, mask=cherenkovsizepart["num_bunches"] > 0)

    # cherenkovpoolpart
    # -------------
    cherenkovpoolpart = {}
    cherenkovpoolpart[spt.IDX] = _pplvl["idx"]
    cherenkovpoolpart["maximum_asl_m"] = prng.normal(
        loc=1e4, scale=3e3, size=_pplvl["num"]
    )
    cherenkovpoolpart["wavelength_median_nm"] = prng.normal(
        loc=433.0, scale=10.0, size=_pplvl["num"]
    )
    cherenkovpoolpart["cx_median_rad"] = prng.normal(
        loc=0.0, scale=config["max_scatter_rad"], size=_pplvl["num"]
    )
    cherenkovpoolpart["cy_median_rad"] = prng.normal(
        loc=0.0, scale=config["max_scatter_rad"], size=_pplvl["num"]
    )
    cherenkovpoolpart["x_median_m"] = 0.0 * _pplvl["ones"]
    cherenkovpoolpart["y_median_m"] = 0.0 * _pplvl["ones"]
    cherenkovpoolpart["bunch_size_median"] = prng.uniform(
        low=0.9, high=1.0, size=_pplvl["num"]
    )
    t["cherenkovpoolpart"] = cherenkovpoolpart

    _past_trigger_ridxs = prng.choice(
        np.arange(_pplvl["idx"].shape[0]),
        size=int(
            _pplvl["idx"].shape[0]
            * config["propagation_probability"][
                "one_grid_cell_over_threshold_to_trigger"
            ]
        ),
    )
    past_trigger_mask = np.zeros(_pplvl["idx"].shape[0], dtype=np.bool)
    past_trigger_mask[_past_trigger_ridxs] = 1

    # core
    # ----
    core = {}
    core[spt.IDX] = _pplvl["idx"]
    core["bin_idx_x"] = 0 * _pplvl["ones"]
    core["bin_idx_y"] = 0 * _pplvl["ones"]
    core["core_x_m"] = 0.0 * _pplvl["ones"]
    core["core_y_m"] = 0.0 * _pplvl["ones"]

    t["core"] = core

    # trigger
    # -------
    _add_pe = 10.0 * _pplvl["energy_GeV"]
    trigger = {}
    trigger[spt.IDX] = _pplvl["idx"]
    trigger["num_cherenkov_pe"] = 50.0 * _pplvl["energy_GeV"]
    trigger["response_pe"] = 100.0 * past_trigger_mask + _add_pe
    trigger["focus_00_response_pe"] = 100.0 * past_trigger_mask + _add_pe
    trigger["focus_01_response_pe"] = 100.0 * past_trigger_mask + _add_pe
    trigger["focus_02_response_pe"] = 100.0 * past_trigger_mask + _add_pe
    trigger["focus_03_response_pe"] = 100.0 * past_trigger_mask + _add_pe
    trigger["focus_04_response_pe"] = 100.0 * past_trigger_mask + _add_pe
    trigger["focus_05_response_pe"] = 100.0 * past_trigger_mask + _add_pe
    trigger["focus_06_response_pe"] = 100.0 * past_trigger_mask + _add_pe
    trigger["focus_07_response_pe"] = 100.0 * past_trigger_mask + _add_pe
    trigger["focus_08_response_pe"] = 100.0 * past_trigger_mask + _add_pe
    trigger["focus_09_response_pe"] = 100.0 * past_trigger_mask + _add_pe
    trigger["focus_10_response_pe"] = 100.0 * past_trigger_mask + _add_pe
    trigger["focus_11_response_pe"] = 100.0 * past_trigger_mask + _add_pe
    t["trigger"] = trigger

    # pasttrigger
    # -----------
    assert len(_pplvl["idx"]) == len(past_trigger_mask)
    pasttrigger = {}
    pasttrigger[spt.IDX] = _pplvl["idx"][past_trigger_mask]
    t["pasttrigger"] = pasttrigger

    _ptlvl = cut_level(level=_pplvl, mask=past_trigger_mask)

    # cherenkovclassification
    # -----------------------
    cercls = {}
    cercls[spt.IDX] = _ptlvl["idx"]
    cercls["num_true_positives"] = (
        0.76 * trigger["num_cherenkov_pe"][past_trigger_mask]
    )
    cercls["num_false_negatives"] = (
        0.2 * trigger["num_cherenkov_pe"][past_trigger_mask]
    )
    cercls["num_false_positives"] = (
        0.2 * trigger["num_cherenkov_pe"][past_trigger_mask]
    )
    cercls["num_true_negatives"] = prng.normal(
        loc=1e6, scale=1e3, size=_ptlvl["num"]
    )
    t["cherenkovclassification"] = cercls

    # features
    # --------
    features = {}
    features[spt.IDX] = _ptlvl["idx"]
    features["num_photons"] = (
        1.2 * trigger["num_cherenkov_pe"][past_trigger_mask]
    )
    features["paxel_intensity_peakness_std_over_mean"] = _ptlvl["ones"]
    features["paxel_intensity_peakness_max_over_mean"] = _ptlvl["ones"]
    features["paxel_intensity_median_x"] = _ptlvl["ones"]
    features["paxel_intensity_median_y"] = _ptlvl["ones"]
    features["aperture_num_islands_watershed_rel_thr_2"] = _ptlvl["ones"]
    features["aperture_num_islands_watershed_rel_thr_4"] = _ptlvl["ones"]
    features["aperture_num_islands_watershed_rel_thr_8"] = _ptlvl["ones"]
    features["light_front_cx"] = prng.normal(
        loc=0.0, scale=np.deg2rad(3.25), size=_ptlvl["num"]
    )
    features["light_front_cy"] = prng.normal(
        loc=0.0, scale=np.deg2rad(3.25), size=_ptlvl["num"]
    )
    features["image_infinity_cx_mean"] = features[
        "light_front_cy"
    ] + prng.normal(loc=0.0, scale=np.deg2rad(0.25), size=_ptlvl["num"])
    features["image_infinity_cy_mean"] = features[
        "light_front_cx"
    ] + prng.normal(loc=0.0, scale=np.deg2rad(0.25), size=_ptlvl["num"])
    features["image_infinity_cx_std"] = prng.normal(
        loc=0.0, scale=np.deg2rad(0.1), size=_ptlvl["num"]
    )
    features["image_infinity_cy_std"] = prng.normal(
        loc=0.0, scale=np.deg2rad(0.1), size=_ptlvl["num"]
    )
    features["image_infinity_num_photons_on_edge_field_of_view"] = _ptlvl[
        "ones"
    ]
    features["image_smallest_ellipse_object_distance"] = _ptlvl["ones"]
    features["image_smallest_ellipse_solid_angle"] = _ptlvl["ones"]
    features["image_smallest_ellipse_half_depth"] = _ptlvl["ones"]
    features["image_half_depth_shift_cx"] = prng.normal(
        loc=0.0, scale=np.deg2rad(0.5), size=_ptlvl["num"]
    )
    features["image_half_depth_shift_cy"] = prng.normal(
        loc=0.0, scale=np.deg2rad(0.5), size=_ptlvl["num"]
    )
    features[
        "image_smallest_ellipse_num_photons_on_edge_field_of_view"
    ] = _ptlvl["ones"]
    features["image_num_islands"] = prng.uniform(
        low=1, high=2, size=_ptlvl["num"]
    )

    t["features"] = features

    for level in t:
        for key in table.STRUCTURE[level]:
            t[level][key] = t[level][key].astype(
                table.STRUCTURE[level][key]["dtype"]
            )
        t[level][spt.IDX] = t[level][spt.IDX].astype(spt.IDX_DTYPE)

    for level in t:
        t[level] = pandas.DataFrame(t[level]).to_records(index=False)

    return t
