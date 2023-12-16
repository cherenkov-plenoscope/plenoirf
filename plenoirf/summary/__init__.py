import os
import copy
from os.path import join as opj
import pandas
import numpy as np
import pkg_resources
import subprocess
import sparse_numeric_table as spt
import glob
import json_utils
import atmospheric_cherenkov_response
import merlict_development_kit_python

from .. import features
from .. import reconstruction
from .. import analysis
from .. import table
from .. import provenance
from .. import production
from .. import outer_telescope_array
from . import figure
from .cosmic_flux import make_gamma_ray_reference_flux
from .scripts_multiprocessing import run_parallel


def init(run_dir):
    summary_config = _guess_summary_config(run_dir)

    summary_dir = os.path.join(run_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    with open(opj(summary_dir, "summary_config.json"), "wt") as fout:
        fout.write(json_utils.dumps(summary_config, indent=4))


def argv_since_py(sys_argv):
    argv = []
    for arg in sys_argv:
        if len(argv) > 0:
            argv.append(arg)
        if ".py" in arg:
            argv.append(arg)
    return argv


def paths_from_argv(argv):
    assert len(argv) == 2
    run_dir = argv[1]
    summary_dir = os.path.join(run_dir, "summary")
    script_name = str.split(os.path.basename(argv[0]), ".")[0]
    return {
        "run_dir": run_dir,
        "script_name": script_name,
        "summary_dir": summary_dir,
        "out_dir": os.path.join(summary_dir, script_name),
    }


def production_name_from_run_dir(path):
    return os.path.basename(os.path.normpath(path))


def read_summary_config(summary_dir):
    with open(opj(summary_dir, "summary_config.json"), "rt") as fin:
        config = json_utils.loads(fin.read())
    return config


def read_instrument_response_config(run_dir):
    with open(opj(run_dir, "input", "config.json"), "rt") as f:
        config = json_utils.loads(f.read())
    light_field_sensor_geometry = merlict_development_kit_python.plenoscope_propagator.read_plenoscope_geometry(
        opj(run_dir, "input", "scenery", "scenery.json")
    )

    grid_geometry = atmospheric_cherenkov_response.grid.init_geometry(
        instrument_aperture_outer_diameter=(
            2.0
            * light_field_sensor_geometry[
                "expected_imaging_system_aperture_radius"
            ]
        ),
        bin_width_overhead=config["grid"]["bin_width_overhead"],
        instrument_field_of_view_outer_radius_deg=(
            0.5 * light_field_sensor_geometry["max_FoV_diameter_deg"]
        ),
        instrument_pointing_direction=[0, 0, 1],
        field_of_view_overhead=config["grid"]["field_of_view_overhead"],
        num_bins_radius=config["grid"]["num_bins_radius"],
    )

    with open(opj(run_dir, "input", "scenery", "scenery.json"), "rt") as f:
        plenoscope_scenery = json_utils.loads(f.read())
    _prop_cfg_path = opj(run_dir, "input", "merlict_propagation_config.json")
    with open(_prop_cfg_path, "rt") as f:
        merlict_propagation_config = json_utils.loads(f.read())
    bundle = {
        "config": config,
        "light_field_sensor_geometry": light_field_sensor_geometry,
        "plenoscope_scenery": plenoscope_scenery,
        "grid_geometry": grid_geometry,
        "merlict_propagation_config": merlict_propagation_config,
    }
    return bundle


def run(run_dir):
    json_utils.write(
        path=opj(run_dir, "summary", "provenance.json"),
        out_dict=provenance.make_provenance(),
    )

    script_abspaths = _make_script_abspaths()

    for script_abspath in script_abspaths:
        script_basename = os.path.basename(script_abspath)
        script_name = str.split(script_basename, ".")[0]
        result_path = os.path.join(run_dir, "summary", script_name)
        if os.path.exists(result_path):
            print("[skip] ", script_name)
        else:
            print("[run ] ", script_name)
            subprocess.call(["python", script_abspath, run_dir])


def _make_script_abspaths():
    script_absdir = pkg_resources.resource_filename(
        "plenoirf", os.path.join("summary", "scripts")
    )
    _paths = glob.glob(os.path.join(script_absdir, "*"))
    out = []
    order = []
    for _path in _paths:
        basename = os.path.basename(_path)
        if str.isdigit(basename[0:4]):
            order.append(int(basename[0:4]))
            out.append(_path)
    order = np.array(order)
    argorder = np.argsort(order)
    out_order = [out[arg] for arg in argorder]
    return out_order


def _estimate_num_events_past_trigger(run_dir, irf_config):
    irf_config = read_instrument_response_config(run_dir=run_dir)

    num_events_past_trigger = 10 * 1000
    for site_key in irf_config["config"]["sites"]:
        for particle_key in irf_config["config"]["particles"]:
            event_table = spt.read(
                path=os.path.join(
                    run_dir,
                    "event_table",
                    site_key,
                    particle_key,
                    "event_table.tar",
                ),
                structure=table.STRUCTURE,
            )
            if event_table["pasttrigger"].shape[0] < num_events_past_trigger:
                num_events_past_trigger = event_table["pasttrigger"].shape[0]
    return num_events_past_trigger


def _guess_num_direction_bins(num_events):
    num_bins = int(0.5 * np.sqrt(num_events))
    num_bins = np.max([np.min([num_bins, 2**7]), 2**4])
    return num_bins


def make_ratescan_trigger_thresholds(
    lower_threshold,
    upper_threshold,
    num_thresholds,
    collection_trigger_threshold,
    analysis_trigger_threshold,
):
    assert lower_threshold <= collection_trigger_threshold
    assert upper_threshold >= collection_trigger_threshold

    assert lower_threshold <= analysis_trigger_threshold
    assert upper_threshold >= analysis_trigger_threshold

    tt = np.geomspace(
        lower_threshold,
        upper_threshold,
        num_thresholds,
    )
    tt = np.round(tt)
    tt = tt.tolist()
    tt = tt + [collection_trigger_threshold]
    tt = tt + [analysis_trigger_threshold]
    tt = np.array(tt, dtype=np.int)
    tt = set(tt)
    tt = list(tt)
    tt = np.sort(tt)
    return tt


FERMI_3FGL_CRAB_NEBULA_NAME = "3FGL J0534.5+2201"
FERMI_3FGL_PHD_THESIS_REFERENCE_SOURCE_NAME = "3FGL J2254.0+1608"


def _guess_trigger(
    collection_trigger_threshold_pe,
    analysis_trigger_threshold_pe,
    site_altitude_asl_m,
    trigger_foci_object_distamces_m,
    trigger_accepting_altitude_asl_m=19856,
    trigger_rejecting_altitude_asl_m=13851,
):
    """

    example Namibia:
    - accepting focus 17,556m, rejecting focus 11,551m, site's altitude 2,300m
    """
    _obj = trigger_foci_object_distamces_m
    _trg_acc_alt = trigger_accepting_altitude_asl_m
    _trg_rej_alt = trigger_rejecting_altitude_asl_m
    _site_alt = site_altitude_asl_m

    accep = np.argmin(np.abs(_obj - _trg_acc_alt + _site_alt))
    rejec = np.argmin(np.abs(_obj - _trg_rej_alt + _site_alt))

    modus = {
        "modus": {
            "accepting_focus": accep,
            "rejecting_focus": rejec,
            "accepting": {
                "threshold_accepting_over_rejecting": [
                    1,
                    1,
                    0.8,
                    0.4,
                    0.2,
                    0.1,
                ],
                "response_pe": [1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
            },
        },
        "threshold_pe": analysis_trigger_threshold_pe,
        "ratescan_thresholds_pe": make_ratescan_trigger_thresholds(
            lower_threshold=int(collection_trigger_threshold_pe * 0.86),
            upper_threshold=int(collection_trigger_threshold_pe * 1.6),
            num_thresholds=32,
            collection_trigger_threshold=collection_trigger_threshold_pe,
            analysis_trigger_threshold=analysis_trigger_threshold_pe,
        ),
    }
    return modus


def guess_num_offregions(
    fov_radius_deg,
    gamma_resolution_radius_at_energy_threshold_deg,
    onregion_radius_deg,
    fraction_of_fov_being_useful,
):
    assert gamma_resolution_radius_at_energy_threshold_deg > 0.0
    assert 0 < fraction_of_fov_being_useful < 1
    assert fov_radius_deg > 0.0
    assert onregion_radius_deg > 0.0
    assert fov_radius_deg > onregion_radius_deg
    valid_fov_radius_deg = (
        fov_radius_deg - gamma_resolution_radius_at_energy_threshold_deg
    )
    num = int(
        np.round(
            (valid_fov_radius_deg**2 / onregion_radius_deg**2)
            * fraction_of_fov_being_useful
        )
    )
    return num


def _guess_summary_config(run_dir):
    irf_config = read_instrument_response_config(run_dir=run_dir)

    num_events_past_collection_trigger = _estimate_num_events_past_trigger(
        run_dir=run_dir, irf_config=irf_config
    )

    collection_trigger_threshold_pe = irf_config["config"]["sum_trigger"][
        "threshold_pe"
    ]
    analysis_trigger_threshold_pe = int(
        np.round(1.09 * collection_trigger_threshold_pe)
    )

    fov_radius_deg = (
        0.5 * irf_config["light_field_sensor_geometry"]["max_FoV_diameter_deg"]
    )

    _onoff = {
        "opening_angle_scaling": {
            "reco_num_photons_pe": [1e1, 1e2, 1e3, 1e4, 1e5],
            "scale": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        "ellipticity_scaling": {
            "reco_core_radius_m": [0.0, 2.5e2, 5e2, 1e3],
            "scale": [1.0, 1.0, 1.0, 1.0],
        },
    }

    cfg = {
        "energy_binning": {
            "start": {"decade": -1, "bin": 2},
            "stop": {"decade": 3, "bin": 2},
            "num_bins_per_decade": 5,
            "fine": {
                "trigger_acceptance": 2,
                "trigger_acceptance_onregion": 1,
                "interpolation": 12,
                "point_spread_function": 1,
            },
        },
        "direction_binning": {
            "radial_angle_deg": 35.0,
            "num_bins": _guess_num_direction_bins(
                num_events_past_collection_trigger
            ),
        },
        "night_sky_background": {
            "max_num_true_cherenkov_photons": 0,
        },
        "airshower_flux": {
            "fraction_of_flux_below_geomagnetic_cutoff": 0.05,
            "relative_uncertainty_below_geomagnetic_cutoff": 0.5,
        },
        "gamma_ray_source_direction": {
            "max_angle_relative_to_pointing_deg": fov_radius_deg - 0.5,
        },
        "train_and_test": {"test_size": 0.5},
        "gamma_hadron_seperation": {"gammaness_threshold": 0.5},
        "random_seed": 1,
        "quality": {
            "max_relative_leakage": 0.1,
            "min_reconstructed_photons": 50,
            "min_trajectory_quality": 0.3,
        },
        "point_spread_function": {
            "theta_square": {
                "max_angle_deg": 3.25,
                "num_bins": 256,
            },
            "core_radius": {"max_radius_m": 640, "num_bins": 4},
            "containment_factor": 0.68,
            "pivot_energy_GeV": 2.0,
        },
        "differential_sensitivity": {
            "gamma_ray_effective_area_scenario": "bell_spectrum",
        },
        "on_off_measuremnent": {
            "estimator_for_critical_signal_rate": "LiMaEq17",
            "detection_threshold_std": 5.0,
            "systematic_uncertainties": [1e-2, 1e-3],
            "onregion_types": {
                "small": {
                    "opening_angle_deg": 0.2,
                    "opening_angle_scaling": _onoff["opening_angle_scaling"],
                    "ellipticity_scaling": _onoff["ellipticity_scaling"],
                    "on_over_off_ratio": 1 / 5,
                },
                "medium": {
                    "opening_angle_deg": 0.4,
                    "opening_angle_scaling": _onoff["opening_angle_scaling"],
                    "ellipticity_scaling": _onoff["ellipticity_scaling"],
                    "on_over_off_ratio": 1 / 5,
                },
                "large": {
                    "opening_angle_deg": 0.8,
                    "opening_angle_scaling": _onoff["opening_angle_scaling"],
                    "ellipticity_scaling": _onoff["ellipticity_scaling"],
                    "on_over_off_ratio": 1 / 5,
                },
            },
        },
        "gamma_ray_reference_source": {
            "type": "3fgl",
            "name_3fgl": FERMI_3FGL_CRAB_NEBULA_NAME,
            "generic_power_law": {
                "flux_density_per_m2_per_s_per_GeV": 1e-3,
                "spectral_index": -2.0,
                "pivot_energy_GeV": 1.0,
            },
        },
        "outer_telescope_array_configurations": {
            "ring-mst": {
                "mirror_diameter_m": 11.5,
                "positions": outer_telescope_array.init_telescope_positions_in_annulus(
                    outer_radius=2.5,
                    inner_radius=0.5,
                ),
            },
            "many-sst": {
                "mirror_diameter_m": 4.3,
                "positions": outer_telescope_array.init_telescope_positions_in_annulus(
                    outer_radius=5.5,
                    inner_radius=0.5,
                ),
            },
            "few-magics": {
                "mirror_diameter_m": 17.0,
                "positions": [
                    [1, 1],
                    [-1, 1],
                    [-1, -1],
                    [1, -1],
                ],
            },
        },
    }

    cfg["plot"] = {}
    cfg["plot"]["matplotlib"] = figure.MATPLOTLIB_RCPARAMS_LATEX
    cfg["plot"]["particle_colors"] = figure.PARTICLE_COLORS

    cfg["trigger"] = {}
    SITES = irf_config["config"]["sites"]
    for sk in irf_config["config"]["sites"]:
        cfg["trigger"][sk] = _guess_trigger(
            collection_trigger_threshold_pe=collection_trigger_threshold_pe,
            analysis_trigger_threshold_pe=analysis_trigger_threshold_pe,
            site_altitude_asl_m=SITES[sk]["observation_level_asl_m"],
            trigger_foci_object_distamces_m=irf_config["config"][
                "sum_trigger"
            ]["object_distances_m"],
            trigger_accepting_altitude_asl_m=19856,
            trigger_rejecting_altitude_asl_m=13851,
        )
    return cfg


def read_train_test_frame(
    site_key,
    particle_key,
    run_dir,
    transformed_features_dir,
    trigger_config,
    quality_config,
    train_test,
    level_keys,
):
    sk = site_key
    pk = particle_key

    airshower_table = spt.read(
        path=os.path.join(
            run_dir,
            "event_table",
            sk,
            pk,
            "event_table.tar",
        ),
        structure=table.STRUCTURE,
    )

    airshower_table["transformed_features"] = spt.read(
        path=os.path.join(
            transformed_features_dir,
            sk,
            pk,
            "transformed_features.tar",
        ),
        structure=features.TRANSFORMED_FEATURE_STRUCTURE,
    )["transformed_features"]

    idxs_triggered = analysis.light_field_trigger_modi.make_indices(
        trigger_table=airshower_table["trigger"],
        threshold=trigger_config["threshold_pe"],
        modus=trigger_config["modus"],
    )
    idxs_quality = analysis.cuts.cut_quality(
        feature_table=airshower_table["features"],
        max_relative_leakage=quality_config["max_relative_leakage"],
        min_reconstructed_photons=quality_config["min_reconstructed_photons"],
    )

    EXT_STRUCTRURE = dict(table.STRUCTURE)
    EXT_STRUCTRURE[
        "transformed_features"
    ] = features.TRANSFORMED_FEATURE_STRUCTURE["transformed_features"]

    out = {}
    for kk in ["test", "train"]:
        idxs_valid_kk = spt.intersection(
            [
                idxs_triggered,
                idxs_quality,
                train_test[sk][pk][kk],
            ]
        )
        table_kk = spt.cut_and_sort_table_on_indices(
            table=airshower_table,
            common_indices=idxs_valid_kk,
            level_keys=level_keys,
        )
        out[kk] = spt.make_rectangular_DataFrame(table_kk)

    return out
