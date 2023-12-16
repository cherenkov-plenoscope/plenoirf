import os
from os import path as op
import json_utils
import binning_utils

import magnetic_deflection
from .. import utils
from .. import reconstruction
from .. import instrument_response
from .. import provenance


def absjoin(*args):
    return op.abspath(op.join(*args))


CORSIKA_PRIMARY_PATH = absjoin(
    "build",
    "corsika",
    "modified",
    "corsika-75600",
    "run",
    "corsika75600Linux_QGSII_urqmd",
)

MERLICT_PLENOSCOPE_PROPAGATOR_PATH = absjoin(
    "build", "merlict", "merlict-plenoscope-propagation"
)

EXAMPLE_SUM_TRIGGER = {
    "object_distances_m": [
        5000.0,
        6164.0,
        7600.0,
        9369.0,
        11551.0,
        14240.0,
        17556.0,
        21644.0,
        26683.0,
        32897.0,
        40557.0,
        50000.0,
    ],
    "threshold_pe": 107,
    "integration_time_slices": 10,
    "image": {
        "image_outer_radius_deg": 3.216665,
        "pixel_spacing_deg": 0.06667,
        "pixel_radius_deg": 0.146674,
        "max_number_nearest_lixel_in_pixel": 7,
    },
}

EXAMPLE_CHERENKOV_CLASSIFICATION = {
    "region_of_interest": {
        "time_offset_start_s": -10e-9,
        "time_offset_stop_s": 10e-9,
        "direction_radius_deg": 2.0,
        "object_distance_offsets_m": [
            4000.0,
            2000.0,
            0.0,
            -2000.0,
        ],
    },
    "min_num_photons": 17,
    "neighborhood_radius_deg": 0.075,
    "direction_to_time_mixing_deg_per_s": 0.375e9,
}

ARTIFICIAL_CORE_LIMITATION = {
    "gamma": {
        "energy_GeV": [0.23, 0.8, 3.0, 35, 81, 432, 1000],
        "max_scatter_radius_m": [150, 150, 460, 1100, 1235, 1410, 1660],
    },
    "electron": {
        "energy_GeV": [0.23, 1.0, 10, 100, 1000],
        "max_scatter_radius_m": [150, 150, 500, 1100, 2600],
    },
    "proton": {
        "energy_GeV": [5.0, 25, 250, 1000],
        "max_scatter_radius_m": [200, 350, 700, 1250],
    },
}
ARTIFICIAL_CORE_LIMITATION["helium"] = ARTIFICIAL_CORE_LIMITATION[
    "proton"
].copy()


def make_example_job(
    run_dir,
    num_air_showers=25,
    production_key="_testing",
    particle_key="proton",
    site_key="namibia",
    artificial_core_limitation=None,
    run_id=1,
):
    deflection_table = magnetic_deflection.read_deflection(
        work_dir=op.join(run_dir, "magnetic_deflection"),
        style="dict",
    )
    with open(op.join(run_dir, "input", "config.json"), "rt") as fin:
        config = json_utils.loads(fin.read())

    job = instrument_response.make_job_dict(
        run_dir=run_dir,
        production_key=production_key,
        run_id=run_id,
        site_key=site_key,
        particle_key=particle_key,
        config=config,
        deflection_table=deflection_table,
        num_air_showers=num_air_showers,
        corsika_primary_path=CORSIKA_PRIMARY_PATH,
        merlict_plenoscope_propagator_path=MERLICT_PLENOSCOPE_PROPAGATOR_PATH,
        tmp_dir=op.join(
            run_dir, production_key, site_key, particle_key, "tmp"
        ),
        keep_tmp_dir=True,
        date_dict=provenance.get_time_dict_now(),
    )

    return job


def make_helium_demo_for_tomography(
    run_dir,
    production_key="demo_helium_for_tomography",
    run_id=1,
    num_air_showers=25,
    site_key="namibia",
    max_scatter_radius_m=250,
    max_scatter_angle_deg=1.5,
):
    job = make_example_job(
        run_dir=run_dir,
        num_air_showers=num_air_showers,
        production_key=production_key,
        particle_key="helium",
        site_key=site_key,
        artificial_core_limitation=None,
    )

    energy_start = binning_utils.power10.lower_bin_edge(
        decade=3, bin=1, num_bins_per_decade=5
    )
    energy_stop = binning_utils.power10.lower_bin_edge(
        decade=3, bin=2, num_bins_per_decade=5
    )

    job["particle"] = {
        "particle_id": 402,
        "energy_bin_edges_GeV": [
            energy_start,
            energy_stop,
        ],
        "max_scatter_angle_deg": max_scatter_angle_deg,
        "energy_power_law_slope": -1.5,
        "electric_charge_qe": +2.0,
        "magnetic_deflection_max_off_axis_deg": 1.5,
    }

    job["artificial_core_limitation"] = {
        "energy_GeV": [energy_start, energy_stop],
        "max_scatter_radius_m": [
            max_scatter_radius_m,
            max_scatter_radius_m,
        ],
    }

    job["raw_sensor_response"] = {"skip_num_events": 1}

    return job
