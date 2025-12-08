import os
import numpy as np
import warnings

from importlib import resources as importlib_resources

import subprocess
import sparse_numeric_table as snt

import glob
import json_utils

import atmospheric_cherenkov_response
import merlict_development_kit_python
import solid_angle_utils
import binning_utils
import shutil
import copy

from .. import utils
from .. import provenance
from .. import outer_telescope_array
from .. import configuration
from .. import event_table
from . import figure
from . import report
from . import scripts
from . import estimator
from .scripts import run

from .cosmic_flux import make_gamma_ray_reference_flux


def argv_since_py(argv):
    _argv = []
    for arg in argv:
        if len(_argv) > 0:
            _argv.append(arg)
        if ".py" in arg:
            _argv.append(arg)
    return _argv


class ScriptResources:
    """
    Lazy
    """

    def __init__(self, plenoirf_dir, instrument_key, site_key, script_name):
        self.instrument_key = instrument_key
        self.site_key = site_key
        self.script_name = script_name

        self.paths = {}
        self.paths["plenoirf_dir"] = plenoirf_dir
        self.paths["analysis_dir"] = os.path.join(
            plenoirf_dir, "analysis", instrument_key, site_key
        )
        self.paths["out_dir"] = os.path.join(
            self.paths["analysis_dir"], script_name
        )
        try:
            ppp = __path__[0]
            ppp, e = os.path.split(ppp)
            assert e == "summary"
            ppp, e = os.path.split(ppp)
            assert e == "plenoirf"
            ppp, e = os.path.split(ppp)
            assert e == "plenoirf"
            ppp, e = os.path.split(ppp)
            assert e == "packages"
            self.paths["starter_kit_dir"] = ppp
            assert "starter_kit" in os.path.basename(
                self.paths["starter_kit_dir"]
            )
        except Exception as err:
            self.paths["starter_kit_dir"] = None

    def start(self, sebplt=None):
        self.final_out_dir = copy.copy(self.paths["out_dir"])
        self.paths["out_dir"] += ".part"
        os.makedirs(self.paths["out_dir"], exist_ok=True)

        # caching
        # -------
        self.paths["cache_dir"] = os.path.join(
            self.paths["out_dir"], "__cache__"
        )
        old_cache_dir = os.path.join(self.final_out_dir, "__cache__")
        if os.path.exists(old_cache_dir):
            shutil.copytree(
                src=old_cache_dir,
                dst=self.paths["cache_dir"],
                dirs_exist_ok=True,
            )
        else:
            os.makedirs(self.paths["cache_dir"], exist_ok=True)

        # plotting
        # --------
        if sebplt is not None:
            sebplt.matplotlib.rcParams.update(
                self.analysis["plot"]["matplotlib"]
            )

    def stop(self):
        # overwrite existing out_dir
        if os.path.exists(self.final_out_dir):
            shutil.rmtree(path=self.final_out_dir)

        os.rename(self.paths["out_dir"], self.final_out_dir)
        self.paths["out_dir"] = copy.copy(self.paths["out_dir"])
        self.paths["cache_dir"] = os.path.join(
            self.paths["out_dir"], "__cache__"
        )

    @classmethod
    def from_argv(cls, argv):
        argv = argv_since_py(argv)
        assert len(argv) == 4
        script_name = str.split(os.path.basename(argv[0]), ".")[0]
        return cls(
            plenoirf_dir=argv[1],
            instrument_key=argv[2],
            site_key=argv[3],
            script_name=script_name,
        )

    @property
    def config(self):
        if not hasattr(self, "_config"):
            self._config = configuration.read(
                plenoirf_dir=self.paths["plenoirf_dir"]
            )
        return self._config

    @property
    def instrument(self):
        if not hasattr(self, "_instrument"):
            self._instrument = read_instrument_config(
                plenoirf_dir=self.paths["plenoirf_dir"],
                instrument_key=self.instrument_key,
            )
        return self._instrument

    @property
    def SITES(self):
        if not hasattr(self, "_SITES"):
            self._SITES = _init_SITES(config=self.config)
        return self._SITES

    @property
    def SITE(self):
        if not hasattr(self, "_SITE"):
            self._SITE = self.SITES[self.site_key]
        return self._SITE

    @property
    def PARTICLES(self):
        if not hasattr(self, "_PARTICLES"):
            self._PARTICLES = _init_PARTICLES(config=self.config)
        return self._PARTICLES

    @property
    def COSMIC_RAYS(self):
        if not hasattr(self, "_COSMIC_RAYS"):
            self._COSMIC_RAYS = utils.filter_particles_with_electric_charge(
                self.PARTICLES
            )
        return self._COSMIC_RAYS

    @property
    def analysis(self):
        if not hasattr(self, "_analysis"):
            path = os.path.join(
                self.paths["plenoirf_dir"],
                "analysis",
                self.instrument_key,
                "config",
            )
            self._analysis = json_utils.tree.read(path)
        return self._analysis

    def response_path(self, particle_key):
        return os.path.join(
            self.paths["plenoirf_dir"],
            "response",
            self.instrument_key,
            self.site_key,
            particle_key,
            "reduce",
        )

    def event_table_path(self, particle_key):
        return os.path.join(
            self.response_path(particle_key=particle_key),
            "event_table.snt.zip",
        )

    def open_event_table(self, particle_key):
        return snt.open(
            self.event_table_path(particle_key=particle_key),
            mode="r",
        )

    def zenith_binning(self, key):
        return init_zenith_binning_from_analysis_config(
            analysis_config=self.analysis,
            key=key,
        )

    def ax_add_site_marker(self, ax, x=0.1, y=0.1):
        ax.text(
            x,
            y,
            f"site: {self.SITE['name']}",
            # horizontalalignment="center",
            # verticalalignment="center",
            transform=ax.transAxes,
        )

    @property
    def PARTICLE_COLORS(self):
        return self.analysis["plot"]["particle_colors"]

    @property
    def PARTICLE_COLORMAPS(self):
        if not hasattr(self, "_PARTICLE_COLORMAPS"):
            self._PARTICLE_COLORMAPS = figure.make_particle_colormaps(
                particle_colors=self.PARTICLE_COLORS
            )
        return self._PARTICLE_COLORMAPS

    def energy_binning(self, key):
        return init_energy_binning_from_analysis_config(
            analysis_config=self.analysis,
            key=key,
        )

    def scatter_binning(self, particle_key):
        num_scatter_bins = 20
        max_scatter_solid_angle_sr = np.max(
            self.config["particles_scatter_solid_angle"][particle_key][
                "solid_angle_sr"
            ]
        )
        edges = np.linspace(
            start=0.0,
            stop=max_scatter_solid_angle_sr,
            num=num_scatter_bins + 1,
        )
        return binning_utils.Binning(bin_edges=edges)

    def trigger_image_object_distance_binning(self):
        if "object_distances" in self.config["sum_trigger"]:
            return configuration.make_sum_trigger_object_distance_geomspace_binning(
                start_m=self.config["sum_trigger"]["object_distances"][
                    "start_m"
                ],
                stop_m=self.config["sum_trigger"]["object_distances"][
                    "stop_m"
                ],
                num=self.config["sum_trigger"]["object_distances"]["num"],
            )
        else:
            warnings.warn(
                "'object_distances' is not yet in the sum_trigger config."
            )
            return configuration.make_sum_trigger_object_distance_geomspace_binning(
                start_m=min(self.config["sum_trigger"]["object_distances_m"]),
                stop_m=max(self.config["sum_trigger"]["object_distances_m"]),
                num=len(self.config["sum_trigger"]["object_distances_m"]),
            )

    @property
    def trigger(self):
        trg = copy.copy(self.analysis["trigger"][self.site_key])
        trg["foci_bin"] = self.trigger_image_object_distance_binning()
        return trg

    @property
    def production_dirname(self):
        return production_name_from_run_dir(self.paths["plenoirf_dir"])

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @property
    def plenoirf_dir_provenance(self):
        if not hasattr(self, "_plenoirf_dir_provenance"):
            self._plenoirf_dir_provenance = (
                provenance.read_plenoirf_dir_provenance(
                    plenoirf_dir=self.paths["plenoirf_dir"],
                    analysis_dir=self.paths["analysis_dir"],
                )
            )
        return self._plenoirf_dir_provenance


def _init_SITES(config):
    SITES = {}
    for sk in config["sites"]["instruemnt_response"]:
        SITES[sk] = atmospheric_cherenkov_response.sites.init(sk)
    return SITES


def _init_PARTICLES(config):
    PARTICLES = {}
    for pk in config["particles"]:
        PARTICLES[pk] = atmospheric_cherenkov_response.particles.init(pk)
    return PARTICLES


def get_PARTICLES(analysis_config):
    particles = {}
    for particle_key in analysis_config["response"]["particles"]:
        particles[particle_key] = (
            atmospheric_cherenkov_response.particles.init(particle_key)
        )
    return particles


def get_SITES(analysis_config):
    sites = {}
    for site_key in analysis_config["response"]["sites"][
        "instruemnt_response"
    ]:
        sites[site_key] = atmospheric_cherenkov_response.sites.init(site_key)
    return sites


def init(plenoirf_dir, instrument_key, site_key, config=None):
    """
    Initialize the summary
    ======================

    """
    config = configuration.read_if_None(plenoirf_dir, config=config)

    assert instrument_key in config["instruments"]
    assert site_key in config["sites"]["instruemnt_response"]

    analysis_instrument_site_dir = os.path.join(
        plenoirf_dir,
        "analysis",
        instrument_key,
        site_key,
    )
    os.makedirs(analysis_instrument_site_dir, exist_ok=True)

    _config_dir = os.path.join(analysis_instrument_site_dir, "config")
    if not os.path.exists(_config_dir):
        os.makedirs(_config_dir, exist_ok=True)
        analysis_config = _guess_analysis_config_for_instrument(
            plenoirf_dir=plenoirf_dir,
            instrument_key=instrument_key,
            site_key=site_key,
            config=config,
        )
        json_utils.tree.write(
            path=_config_dir,
            tree=analysis_config,
            dirtree={},
            indent=4,
        )
    analysis_config = json_utils.tree.read(_config_dir)

    _event_tables_dir = os.path.join(
        analysis_instrument_site_dir,
        "event_tables_binned_by_pointing_zenith_and_primary_energy",
    )
    # if not os.path.exists(_event_tables_dir):
    os.makedirs(_event_tables_dir, exist_ok=True)

    zenith_bin = init_zenith_binning_from_analysis_config(
        analysis_config=analysis_config,
        key="once",
    )
    energy_bin = init_energy_binning_from_analysis_config(
        analysis_config=analysis_config,
        key="trigger_acceptance",
    )

    for particle_key in ["electron"]:  # config["particles"]:
        _particle_dir = os.path.join(_event_tables_dir, particle_key)
        event_table.binned_by_pointing_zenith_and_primary_energy.init(
            work_dir=_particle_dir,
            zenith_bin_edges=zenith_bin["edges"],
            energy_bin_edges=energy_bin["edges"],
        )

        event_table.binned_by_pointing_zenith_and_primary_energy.populate(
            work_dir=_particle_dir,
            event_table_path=os.path.join(
                plenoirf_dir,
                "response",
                instrument_key,
                site_key,
                particle_key,
                "reduce",
                "event_table.snt.zip",
            ),
        )


def production_name_from_run_dir(path):
    return os.path.basename(os.path.normpath(path))


def read_instrument_config(plenoirf_dir, instrument_key):
    instrument_scenery_path = os.path.join(
        plenoirf_dir,
        "plenoptics",
        "instruments",
        instrument_key,
        "light_field_geometry",
        "input",
        "scenery",
        "scenery.json",
    )

    light_field_sensor_geometry = merlict_development_kit_python.plenoscope_propagator.read_plenoscope_geometry(
        instrument_scenery_path
    )

    with open(instrument_scenery_path, "rt") as f:
        plenoscope_scenery = json_utils.loads(f.read())

    bundle = {
        "light_field_sensor_geometry": light_field_sensor_geometry,
        "scenery": plenoscope_scenery,
    }
    return bundle


def _run_instrument_site(plenoirf_dir, instrument_key, site_key):
    result_dir = os.path.join(
        plenoirf_dir, "analysis", instrument_key, site_key
    )
    os.makedirs(result_dir, exist_ok=True)

    json_utils.write(
        os.path.join(result_dir, "provenance.json"),
        provenance.make_provenance(),
    )

    script_abspaths = _make_script_abspaths()

    for script_abspath in script_abspaths:
        script_basename = os.path.basename(script_abspath)
        script_id = int(script_basename[0:4])
        if script_id > 131:
            print(f"Skipping scipt {script_id:d}.")
            continue
        script_name = str.split(script_basename, ".")[0]
        result_path = os.path.join(result_dir, script_name)
        if os.path.exists(result_path):
            print("[skip] ", script_name)
        else:
            print("[run ] ", script_name)
            subprocess.call(
                [
                    "python",
                    script_abspath,
                    plenoirf_dir,
                    instrument_key,
                    site_key,
                ]
            )


def _make_script_abspaths():
    script_absdir = os.path.join(
        importlib_resources.files("plenoirf"), "summary", "scripts"
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


def _estimate_num_events_past_trigger_for_instrument(
    plenoirf_dir, instrument_key, site_key, config=None
):
    config = configuration.read_if_None(plenoirf_dir, config=config)

    num = float("inf")
    for particle_key in config["particles"]:
        path = os.path.join(
            plenoirf_dir,
            "response",
            instrument_key,
            site_key,
            particle_key,
            "reduce",
            "event_table.snt.zip",
        )

        with snt.open(path, mode="r") as arc:
            tab = arc.query(levels_and_columns={"pasttrigger": ("uid",)})
            if tab["pasttrigger"]["uid"].shape[0] < num:
                num = tab["pasttrigger"]["uid"].shape[0]
    return num


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
    tt = np.array(tt, dtype=np.int64)
    tt = set(tt)
    tt = list(tt)
    tt = np.sort(tt)
    return tt


FERMI_3FGL_CRAB_NEBULA_NAME = "3FGL J0534.5+2201"
FERMI_3FGL_PHD_THESIS_REFERENCE_SOURCE_NAME = "3FGL J2254.0+1608"


def _guess_trigger(
    collection_trigger_threshold_pe,
    analysis_trigger_threshold_pe,
):
    modus = {
        "modus": {
            "accepting_altitude_asl_m": 5_000 + 12_500,
            "rejecting_altitude_asl_m": 5_000 + 7_000,
            "accepting": {
                "threshold_accepting_over_rejecting": [
                    1.21,
                    1.21,
                    1.0,
                    0.6,
                    0.3,
                    0.2,
                ],
                "response_pe": [
                    1e1,
                    2e2,
                    1e3,
                    1e4,
                    1e5,
                    1e6,
                ],
                "correction_for_pointing_zenith": {
                    "zenith_rad": np.deg2rad(
                        [
                            0,
                            22.5,
                            45.0,
                        ]
                    ),
                    "factor": [
                        1.0,
                        1.0,
                        1.0,
                    ],
                },
            },
        },
        "threshold_pe": analysis_trigger_threshold_pe,
        "threshold_factor_vs_pointing_zenith": {
            "zenith_rad": np.deg2rad(
                [
                    0,
                    22.5,
                    45.0,
                ]
            ),
            "factor": np.array(
                [
                    1.00,
                    1.00,
                    1.00,
                ]
            ),
        },
    }

    make_ratescan = True

    if make_ratescan:
        modus["ratescan_thresholds_pe"] = make_ratescan_trigger_thresholds(
            lower_threshold=int(collection_trigger_threshold_pe * 0.86),
            upper_threshold=int(collection_trigger_threshold_pe * 2.5),
            num_thresholds=32,
            collection_trigger_threshold=collection_trigger_threshold_pe,
            analysis_trigger_threshold=analysis_trigger_threshold_pe,
        )
    else:
        modus["ratescan_thresholds_pe"] = [analysis_trigger_threshold_pe]

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


def _guess_analysis_config_for_instrument(
    plenoirf_dir, instrument_key, site_key, config=None
):
    config = configuration.read_if_None(plenoirf_dir, config=config)

    ins_config = read_instrument_config(
        plenoirf_dir=plenoirf_dir,
        instrument_key=instrument_key,
    )

    num_events_past_collection_trigger = (
        _estimate_num_events_past_trigger_for_instrument(
            plenoirf_dir=plenoirf_dir,
            instrument_key=instrument_key,
            site_key=site_key,
            config=config,
        )
    )

    collection_trigger_threshold_pe = config["sum_trigger"]["threshold_pe"]
    analysis_trigger_threshold_pe = int(
        np.round(1.085 * collection_trigger_threshold_pe)
    )

    fov_radius_deg = (
        0.5 * ins_config["light_field_sensor_geometry"]["max_FoV_diameter_deg"]
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
            "stop": {"decade": 3, "bin": 3},
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
            "min_reconstructed_photons": 35,
            "min_trajectory_quality": 0.3,
            "min_aperture_intensity_flatness_mean_over_std": 0.8,  # See 0063_aperture_intensity_distribution.py
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

    cfg["pointing_binning"] = {
        "zenith_binning": {
            "start_half_angle_rad": 0,
            "stop_half_angle_rad": np.deg2rad(45),
            "num_bins": 3,
            "fine": {
                "once": 1,
                "twice": 2,
            },
        }
    }

    cfg["plot"] = {}
    cfg["plot"]["matplotlib"] = figure.MATPLOTLIB_RCPARAMS_LATEX
    cfg["plot"]["particle_colors"] = figure.PARTICLE_COLORS

    cfg["trigger"] = _guess_trigger(
        collection_trigger_threshold_pe=collection_trigger_threshold_pe,
        analysis_trigger_threshold_pe=analysis_trigger_threshold_pe,
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

    with snt.open(
        file=os.path.join(
            run_dir,
            "event_table",
            sk,
            pk,
            "event_table.snt.zip",
        ),
        mode="r",
    ) as arc:
        airshower_table = arc.query()

    with snt.open(
        file=os.path.join(
            transformed_features_dir,
            sk,
            pk,
            "transformed_features.zip",
        ),
    ) as arc:
        _part = arc.query()
        airshower_table["transformed_features"] = _part["transformed_features"]

    uids_triggered = analysis.light_field_trigger_modi.make_indices(
        trigger_table=airshower_table["trigger"],
        threshold=trigger_config["threshold_pe"],
        modus=trigger_config["modus"],
    )
    uids_quality = analysis.cuts.cut_quality(
        feature_table=airshower_table["features"],
        max_relative_leakage=quality_config["max_relative_leakage"],
        min_reconstructed_photons=quality_config["min_reconstructed_photons"],
    )

    EXT_STRUCTRURE = dict(table.STRUCTURE)
    EXT_STRUCTRURE["transformed_features"] = (
        features.TRANSFORMED_FEATURE_STRUCTURE["transformed_features"]
    )

    out = {}
    for kk in ["test", "train"]:
        uids_valid_kk = snt.logic.intersection(
            uids_triggered,
            uids_quality,
            train_test[sk][pk][kk],
        )
        table_kk = snt.logic.cut_and_sort_table_on_indices(
            table=airshower_table,
            common_indices=uids_valid_kk,
            level_keys=level_keys,
        )
        out[kk] = snt.logic.make_rectangular_DataFrame(table_kk)

    return out


def init_energy_binning_from_analysis_config(analysis_config, key):
    edges = utils.power10space_bin_edges(
        binning=analysis_config["energy_binning"],
        fine=analysis_config["energy_binning"]["fine"][key],
    )
    assert len(edges) >= 2
    assert np.all(edges > 0.0)
    assert np.all(np.gradient(edges) > 0.0)
    return binning_utils.Binning(bin_edges=edges)


def init_zenith_binning_from_analysis_config(analysis_config, key):
    zb_cfg = analysis_config["pointing_binning"]["zenith_binning"]

    num_bins = zb_cfg["num_bins"] * zb_cfg["fine"][key]
    bin_edges = solid_angle_utils.cone.half_angle_space(
        start_half_angle_rad=zb_cfg["start_half_angle_rad"],
        stop_half_angle_rad=zb_cfg["stop_half_angle_rad"],
        num=num_bins + 1,
    )
    return init_zenith_binning_from_bin_edges(bin_edges=bin_edges)


def init_zenith_binning_from_bin_edges(bin_edges):
    z = binning_utils.Binning(bin_edges=bin_edges)

    # apply spacing to centers
    for i in range(z["num"]):
        z["centers"][i] = solid_angle_utils.cone.half_angle_space(
            start_half_angle_rad=z["edges"][i],
            stop_half_angle_rad=z["edges"][i + 1],
            num=3,
        )[1]

    # add solid angles
    z["solid_angles"] = np.zeros(z["num"])
    for i in range(z["num"]):
        outer = solid_angle_utils.cone.solid_angle(z["edges"][i + 1])
        inner = solid_angle_utils.cone.solid_angle(z["edges"][i])
        z["solid_angles"][i] = outer - inner

    return z


def make_angle_range_str(start_rad, stop_rad):
    circ_str = r"$^\circ{}$"
    zenith_range_str = (
        f"[{np.rad2deg(start_rad):0.1f}"
        + circ_str
        + ", "
        + f"{np.rad2deg(stop_rad):0.1f}"
        + circ_str
        + ")"
    )
    return zenith_range_str
