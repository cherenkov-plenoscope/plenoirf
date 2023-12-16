from . import table
from . import unique
from . import utils
from . import production
from . import analysis
from . import reconstruction
from . import outer_telescope_array

import sys
import numpy as np
import os
from os import path as op
import shutil
import time
import glob

import tempfile
import pandas
import json_utils
import tarfile
import corsika_primary as cpw
import plenopy as pl
import sparse_numeric_table as spt
import gamma_ray_reconstruction as gamrec
import json_line_logger as jlogging
import rename_after_writing as rnw
import atmospheric_cherenkov_response
import solid_angle_utils
import merlict_development_kit_python


def make_job_dict(
    run_dir,
    production_key,
    run_id,
    site_key,
    particle_key,
    config,
    deflection_table,
    num_air_showers,
    corsika_primary_path,
    merlict_plenoscope_propagator_path,
    tmp_dir,
    keep_tmp_dir,
    date_dict,
):
    job = {
        "run_id": run_id,
        "production_key": production_key,
        "site_key": site_key,
        "particle_key": particle_key,
        "num_air_showers": num_air_showers,
        "plenoscope_pointing": config["plenoscope_pointing"],
        "particle": config["particles"][particle_key],
        "site": config["sites"][site_key],
        "site_particle_deflection": deflection_table[site_key][particle_key],
        "grid": config["grid"],
        "raw_sensor_response": config["raw_sensor_response"],
        "sum_trigger": config["sum_trigger"],
        "cherenkov_classification": config["cherenkov_classification"],
        "reconstruction": config["reconstruction"],
        "corsika_primary_path": str(corsika_primary_path),
        "merlict_plenoscope_propagator_path": str(
            merlict_plenoscope_propagator_path
        ),
        "plenoscope_scenery_path": os.path.join(
            run_dir, "light_field_geometry", "input", "scenery"
        ),
        "light_field_geometry_path": os.path.join(
            run_dir, "light_field_geometry"
        ),
        "trigger_geometry_path": os.path.join(run_dir, "trigger_geometry"),
        "merlict_plenoscope_propagator_config_path": os.path.join(
            run_dir, "input", "merlict_propagation_config.json"
        ),
        "log_dir": os.path.join(
            run_dir, production_key, site_key, particle_key, "log.map"
        ),
        "past_trigger_dir": os.path.join(
            run_dir, production_key, site_key, particle_key, "past_trigger.map"
        ),
        "particles_dir": os.path.join(
            run_dir, production_key, site_key, particle_key, "particles.map"
        ),
        "past_trigger_reconstructed_cherenkov_dir": os.path.join(
            run_dir,
            production_key,
            site_key,
            particle_key,
            "past_trigger_reconstructed_cherenkov_dir.map",
        ),
        "feature_dir": os.path.join(
            run_dir, production_key, site_key, particle_key, "features.map"
        ),
        "keep_tmp": keep_tmp_dir,
        "tmp_dir": tmp_dir,
        "date": date_dict,
        "artificial_core_limitation": config["artificial_core_limitation"][
            particle_key
        ],
    }
    return job


def reduce(run_dir, production_key, site_key, particle_key, LAZY, logger=None):
    logger = logger if logger else jlogging.LoggerStdout()

    production_dir = os.path.join(run_dir, production_key)
    site_dir = os.path.join(production_dir, site_key)
    site_particle_dir = os.path.join(site_dir, particle_key)
    log_dir = os.path.join(site_particle_dir, "log.map")
    features_dir = os.path.join(site_particle_dir, "features.map")

    # run-time
    # ========
    log_path = os.path.join(site_particle_dir, "runtime.csv")
    if not op.exists(log_path) or not LAZY:
        _lop_paths = glob.glob(os.path.join(log_dir, "*_runtime.jsonl"))
        jlogging.reduce(list_of_log_paths=_lop_paths, out_path=log_path)
    logger.info("Reduce {:s} {:s} run-time.".format(site_key, particle_key))

    # event table
    # ===========
    event_table_path = os.path.join(site_particle_dir, "event_table.tar")
    if not op.exists(event_table_path) or not LAZY:
        _features_paths = glob.glob(
            os.path.join(features_dir, "*_event_table.tar")
        )
        event_table = spt.concatenate_files(
            list_of_table_paths=_features_paths,
            structure=table.STRUCTURE,
        )
        spt.write(
            path=event_table_path,
            table=event_table,
            structure=table.STRUCTURE,
        )
    logger.info("Reduce {:s} {:s} event_table.".format(site_key, particle_key))

    # grid images
    # ===========
    for grid_key in ["grid", "grid_roi_pasttrigger"]:
        grid_path = os.path.join(site_particle_dir, grid_key + ".tar")
        if not op.exists(grid_path) or not LAZY:
            _grid_paths = glob.glob(
                os.path.join(features_dir, "*_" + grid_key + ".tar")
            )
            atmospheric_cherenkov_response.grid.serialization.reduce(
                list_of_grid_paths=_grid_paths, out_path=grid_path
            )
        logger.info(
            "Reduce {:s} {:s} {:s}.".format(site_key, particle_key, grid_key)
        )

    # cherenkov-photon-stream
    # =======================
    loph_abspath = os.path.join(site_particle_dir, "cherenkov.phs.loph.tar")
    tmp_loph_abspath = loph_abspath + ".tmp"
    logger.info(
        "Reduce {:s} {:s} cherenkov phs.".format(site_key, particle_key)
    )
    if not op.exists(loph_abspath) or not LAZY:
        logger.info("compile {:s}".format(loph_abspath))
        _cer_run_paths = glob.glob(
            os.path.join(
                site_particle_dir,
                "past_trigger_reconstructed_cherenkov_dir.map",
                "*_reconstructed_cherenkov.tar",
            )
        )
        _cer_run_paths.sort()
        pl.photon_stream.loph.concatenate_tars(
            in_paths=_cer_run_paths, out_path=tmp_loph_abspath
        )
        rnw.move(tmp_loph_abspath, loph_abspath)


def _append_bunch_ssize(cherenkovsise_dict, cherenkov_bunches):
    cb = cherenkov_bunches
    ase = cherenkovsise_dict
    ase["num_bunches"] = cb.shape[0]
    ase["num_photons"] = np.sum(cb[:, cpw.I.BUNCH.BUNCH_SIZE_1])
    return ase


def _append_bunch_statistics(airshower_dict, cherenkov_bunches):
    cb = cherenkov_bunches
    ase = airshower_dict
    assert cb.shape[0] > 0
    ase["maximum_asl_m"] = cpw.CM2M * np.median(
        cb[:, cpw.I.BUNCH.EMISSOION_ALTITUDE_ASL_CM]
    )
    ase["wavelength_median_nm"] = np.abs(
        np.median(cb[:, cpw.I.BUNCH.WAVELENGTH_NM])
    )
    ase["cx_median_rad"] = np.median(cb[:, cpw.I.BUNCH.CX_RAD])
    ase["cy_median_rad"] = np.median(cb[:, cpw.I.BUNCH.CY_RAD])
    ase["x_median_m"] = cpw.CM2M * np.median(cb[:, cpw.I.BUNCH.X_CM])
    ase["y_median_m"] = cpw.CM2M * np.median(cb[:, cpw.I.BUNCH.Y_CM])
    ase["bunch_size_median"] = np.median(cb[:, cpw.I.BUNCH.BUNCH_SIZE_1])
    return ase


def plenoscope_event_dir_to_tar(event_dir, output_tar_path=None):
    if output_tar_path is None:
        output_tar_path = event_dir + ".tar"
    with tarfile.open(output_tar_path, "w") as tarfout:
        tarfout.add(event_dir, arcname=".")


def _run_id_str(job):
    form = "{:0" + str(unique.RUN_ID_NUM_DIGITS) + "d}"
    return form.format(job["run_id"])


def _init_grid_geometry_from_job(job):
    assert job["plenoscope_pointing"]["zenith_deg"] == 0.0
    assert job["plenoscope_pointing"]["azimuth_deg"] == 0.0
    plenoscope_pointing_direction = np.array([0, 0, 1])  # For now this is fix.

    _scenery_path = op.join(job["plenoscope_scenery_path"], "scenery.json")
    _light_field_sensor_geometry = merlict_development_kit_python.plenoscope_propagator.read_plenoscope_geometry(
        merlict_scenery_path=_scenery_path
    )
    plenoscope_diameter = (
        2.0
        * _light_field_sensor_geometry[
            "expected_imaging_system_aperture_radius"
        ]
    )
    plenoscope_radius = 0.5 * plenoscope_diameter
    plenoscope_field_of_view_radius_deg = (
        0.5 * _light_field_sensor_geometry["max_FoV_diameter_deg"]
    )

    grid_geometry = atmospheric_cherenkov_response.grid.init_geometry(
        instrument_aperture_outer_diameter=plenoscope_diameter,
        bin_width_overhead=job["grid"]["bin_width_overhead"],
        instrument_field_of_view_outer_radius_deg=(
            plenoscope_field_of_view_radius_deg
        ),
        instrument_pointing_direction=plenoscope_pointing_direction,
        field_of_view_overhead=job["grid"]["field_of_view_overhead"],
        num_bins_radius=job["grid"]["num_bins_radius"],
    )
    return grid_geometry


def _run_corsika_and_grid_and_output_to_tmp_dir(
    job,
    prng,
    tmp_dir,
    corsika_primary_steering,
    tabrec,
):
    grid_geometry = _init_grid_geometry_from_job(job=job)
    GRID_SKIP = int(job["grid"]["output_after_num_events"])
    assert GRID_SKIP > 0

    # loop over air-showers
    # ---------------------
    cherenkov_pools_path = op.join(tmp_dir, "cherenkov_pools.tar")
    particle_pools_dat_path = op.join(tmp_dir, "particle_pools.dat")
    particle_pools_tar_path = op.join(tmp_dir, "particle_pools.tar.gz")
    grid_histogram_path = op.join(tmp_dir, "grid.tar")
    grid_roi_histogram_path = op.join(tmp_dir, "grid_roi.tar")

    with cpw.cherenkov.CherenkovEventTapeWriter(
        path=cherenkov_pools_path
    ) as evttar, tarfile.open(
        grid_histogram_path, "w"
    ) as imgtar, tarfile.open(
        grid_roi_histogram_path, "w"
    ) as imgroitar:
        with cpw.CorsikaPrimary(
            corsika_path=job["corsika_primary_path"],
            steering_dict=corsika_primary_steering,
            stdout_path=op.join(tmp_dir, "corsika.stdout"),
            stderr_path=op.join(tmp_dir, "corsika.stderr"),
            particle_output_path=particle_pools_dat_path,
        ) as corsika_run:
            evttar.write_runh(runh=corsika_run.runh)

            for event_idx, corsika_event in enumerate(corsika_run):
                corsika_evth, cherenkov_reader = corsika_event

                cherenkov_bunches = np.vstack([b for b in cherenkov_reader])

                # assert match
                run_id = int(corsika_evth[cpw.I.EVTH.RUN_NUMBER])
                assert run_id == corsika_primary_steering["run"]["run_id"]
                event_id = event_idx + 1
                assert event_id == corsika_evth[cpw.I.EVTH.EVENT_NUMBER]
                uid = unique.make_uid(run_id=run_id, event_id=event_id)
                uid_str = unique.make_uid_str(run_id=run_id, event_id=event_id)

                ide = {spt.IDX: uid}

                primary = corsika_primary_steering["primaries"][event_idx]

                # export primary table
                # --------------------
                prim = ide.copy()
                prim["particle_id"] = primary["particle_id"]
                prim["energy_GeV"] = primary["energy_GeV"]
                prim["azimuth_rad"] = primary["azimuth_rad"]
                prim["zenith_rad"] = primary["zenith_rad"]
                prim["max_scatter_rad"] = primary["max_scatter_rad"]
                prim[
                    "solid_angle_thrown_sr"
                ] = solid_angle_utils.cone.solid_angle(
                    half_angle_rad=prim["max_scatter_rad"]
                )
                prim["depth_g_per_cm2"] = primary["depth_g_per_cm2"]
                prim["momentum_x_GeV_per_c"] = corsika_evth[
                    cpw.I.EVTH.PX_MOMENTUM_GEV_PER_C
                ]
                prim["momentum_y_GeV_per_c"] = corsika_evth[
                    cpw.I.EVTH.PY_MOMENTUM_GEV_PER_C
                ]
                prim["momentum_z_GeV_per_c"] = (
                    -1.0 * corsika_evth[cpw.I.EVTH.PZ_MOMENTUM_GEV_PER_C]
                )
                prim["first_interaction_height_asl_m"] = (
                    -1.0
                    * cpw.CM2M
                    * corsika_evth[cpw.I.EVTH.Z_FIRST_INTERACTION_CM]
                )
                prim["starting_height_asl_m"] = (
                    cpw.CM2M * corsika_evth[cpw.I.EVTH.STARTING_HEIGHT_CM]
                )
                obs_lvl_intersection = atmospheric_cherenkov_response.utils.ray_plane_x_y_intersection(
                    support=[0, 0, prim["starting_height_asl_m"]],
                    direction=[
                        prim["momentum_x_GeV_per_c"],
                        prim["momentum_y_GeV_per_c"],
                        prim["momentum_z_GeV_per_c"],
                    ],
                    plane_z=job["site"]["observation_level_asl_m"],
                )
                prim["starting_x_m"] = -1.0 * obs_lvl_intersection[0]
                prim["starting_y_m"] = -1.0 * obs_lvl_intersection[1]
                prim["magnet_azimuth_rad"] = primary["magnet_azimuth_rad"]
                prim["magnet_zenith_rad"] = primary["magnet_zenith_rad"]
                prim["magnet_cherenkov_pool_x_m"] = primary[
                    "magnet_cherenkov_pool_x_m"
                ]
                prim["magnet_cherenkov_pool_y_m"] = primary[
                    "magnet_cherenkov_pool_y_m"
                ]
                tabrec["primary"].append(prim)

                # cherenkov size
                # --------------
                crsz = ide.copy()
                crsz = _append_bunch_ssize(crsz, cherenkov_bunches)
                tabrec["cherenkovsize"].append(crsz)

                # assign grid
                # -----------
                grid_random_shift_x, grid_random_shift_y = prng.uniform(
                    low=-0.5 * grid_geometry["bin_width"],
                    high=0.5 * grid_geometry["bin_width"],
                    size=2,
                )

                grhi = ide.copy()
                if job["artificial_core_limitation"]:
                    _max_core_scatter_radius = np.interp(
                        x=primary["energy_GeV"],
                        xp=job["artificial_core_limitation"]["energy_GeV"],
                        fp=job["artificial_core_limitation"][
                            "max_scatter_radius_m"
                        ],
                    )
                    grid_bin_idxs_limitation = atmospheric_cherenkov_response.grid.artificial_core_limitation.where_grid_idxs_within_radius(
                        grid_geometry=grid_geometry,
                        radius=_max_core_scatter_radius,
                        center_x=-1.0 * grid_random_shift_x,
                        center_y=-1.0 * grid_random_shift_y,
                    )
                    grhi["artificial_core_limitation"] = 1
                    grhi[
                        "artificial_core_limitation_radius_m"
                    ] = _max_core_scatter_radius
                    grhi["num_bins_thrown"] = len(grid_bin_idxs_limitation[0])
                    grhi["area_thrown_m2"] = (
                        grhi["num_bins_thrown"] * grid_geometry["bin_area"]
                    )
                else:
                    grid_bin_idxs_limitation = None
                    grhi["artificial_core_limitation"] = 0
                    grhi["artificial_core_limitation_radius_m"] = -1.0
                    grhi["num_bins_thrown"] = grid_geometry["total_num_bins"]
                    grhi["area_thrown_m2"] = grid_geometry["total_area"]

                grhi["bin_width_m"] = grid_geometry["bin_width"]
                grhi["field_of_view_radius_deg"] = grid_geometry[
                    "field_of_view_radius_deg"
                ]
                grhi["pointing_direction_x"] = grid_geometry[
                    "pointing_direction"
                ][0]
                grhi["pointing_direction_y"] = grid_geometry[
                    "pointing_direction"
                ][1]
                grhi["pointing_direction_z"] = grid_geometry[
                    "pointing_direction"
                ][2]
                grhi["random_shift_x_m"] = grid_random_shift_x
                grhi["random_shift_y_m"] = grid_random_shift_y
                grhi["magnet_shift_x_m"] = (
                    -1.0 * primary["magnet_cherenkov_pool_x_m"]
                )
                grhi["magnet_shift_y_m"] = (
                    -1.0 * primary["magnet_cherenkov_pool_x_m"]
                )
                grhi["total_shift_x_m"] = (
                    grhi["random_shift_x_m"] + grhi["magnet_shift_x_m"]
                )
                grhi["total_shift_y_m"] = (
                    grhi["random_shift_y_m"] + grhi["magnet_shift_y_m"]
                )

                grid_result = atmospheric_cherenkov_response.grid.assign(
                    cherenkov_bunches=cherenkov_bunches,
                    grid_geometry=grid_geometry,
                    shift_x=grhi["total_shift_x_m"],
                    shift_y=grhi["total_shift_y_m"],
                    threshold_num_photons=job["grid"]["threshold_num_photons"],
                    prng=prng,
                    bin_idxs_limitation=grid_bin_idxs_limitation,
                )
                if event_idx % GRID_SKIP == 0:
                    utils.tar_append(
                        tarout=imgtar,
                        file_name=uid_str + ".f4.gz",
                        file_bytes=atmospheric_cherenkov_response.grid.histogram_to_bytes(
                            grid_result["histogram"]
                        ),
                    )

                # grid statistics
                # ---------------
                grhi["num_bins_above_threshold"] = grid_result[
                    "num_bins_above_threshold"
                ]
                grhi["overflow_x"] = grid_result["overflow_x"]
                grhi["underflow_x"] = grid_result["underflow_x"]
                grhi["overflow_y"] = grid_result["overflow_y"]
                grhi["underflow_y"] = grid_result["underflow_y"]
                tabrec["grid"].append(grhi)

                # cherenkov statistics
                # --------------------
                if cherenkov_bunches.shape[0] > 0:
                    fase = ide.copy()
                    fase = _append_bunch_statistics(
                        airshower_dict=fase,
                        cherenkov_bunches=cherenkov_bunches,
                    )
                    tabrec["cherenkovpool"].append(fase)

                reuse_event = grid_result["random_choice"]
                if reuse_event is not None:
                    reuse_evth = corsika_evth.copy()
                    reuse_evth[cpw.I.EVTH.NUM_REUSES_OF_CHERENKOV_EVENT] = 1.0
                    reuse_evth[cpw.I.EVTH.X_CORE_CM(reuse=1)] = (
                        cpw.M2CM * reuse_event["core_x_m"]
                    )
                    reuse_evth[cpw.I.EVTH.Y_CORE_CM(reuse=1)] = (
                        cpw.M2CM * reuse_event["core_y_m"]
                    )

                    evttar.write_evth(evth=reuse_evth)
                    evttar.write_payload(
                        payload=reuse_event["cherenkov_bunches"]
                    )

                    crszp = ide.copy()
                    crszp = _append_bunch_ssize(
                        crszp, reuse_event["cherenkov_bunches"]
                    )
                    tabrec["cherenkovsizepart"].append(crszp)
                    rase = ide.copy()
                    rase = _append_bunch_statistics(
                        airshower_dict=rase,
                        cherenkov_bunches=reuse_event["cherenkov_bunches"],
                    )
                    tabrec["cherenkovpoolpart"].append(rase)
                    rcor = ide.copy()
                    rcor["bin_idx_x"] = reuse_event["bin_idx_x"]
                    rcor["bin_idx_y"] = reuse_event["bin_idx_y"]
                    rcor["core_x_m"] = reuse_event["core_x_m"]
                    rcor["core_y_m"] = reuse_event["core_y_m"]
                    tabrec["core"].append(rcor)

                    utils.tar_append(
                        tarout=imgroitar,
                        file_name=uid_str + ".f4.gz",
                        file_bytes=atmospheric_cherenkov_response.grid.serialization.histogram_to_bytes(
                            utils.copy_square_selection_from_2D_array(
                                img=grid_result["histogram"],
                                ix=reuse_event["bin_idx_x"],
                                iy=reuse_event["bin_idx_y"],
                                r=outer_telescope_array.NUM_BINS_RADIUS,
                                fill=np.nan,
                            )
                        ),
                    )

    rnw.copy(
        op.join(tmp_dir, "corsika.stdout"),
        op.join(job["log_dir"], _run_id_str(job) + "_corsika.stdout"),
    )
    rnw.copy(
        op.join(tmp_dir, "corsika.stderr"),
        op.join(job["log_dir"], _run_id_str(job) + "_corsika.stderr"),
    )
    cpw.particles.dat_to_tape(
        dat_path=particle_pools_dat_path,
        tape_path=particle_pools_tar_path,
    )
    rnw.copy(
        src=particle_pools_tar_path,
        dst=op.join(
            job["particles_dir"], _run_id_str(job) + "_particles.tar.gz"
        ),
    )
    rnw.copy(
        src=op.join(tmp_dir, "grid.tar"),
        dst=op.join(job["feature_dir"], _run_id_str(job) + "_grid.tar"),
    )

    return cherenkov_pools_path, tabrec


def records_by_uid(records, uid):
    for record in records:
        assert spt.IDX in record
        if record[spt.IDX] == uid:
            return record


def _populate_particlepool(job, tabrec):
    media_refractive_indices = {
        "water": 1.33,
        "air": cpw.particles.identification.refractive_index_atmosphere(
            altitude_asl_m=job["site"]["observation_level_asl_m"]
        ),
    }
    aperture_diameter_m = _init_grid_geometry_from_job(job=job)["bin_width"]
    aperture_radius_m = 0.5 * aperture_diameter_m

    zoo = cpw.particles.identification.Zoo(
        media_refractive_indices=media_refractive_indices
    )

    particle_path = op.join(
        job["particles_dir"], _run_id_str(job) + "_particles.tar.gz"
    )

    with cpw.particles.ParticleEventTapeReader(path=particle_path) as run:
        for event_idx, event in enumerate(run):
            evth, parreader = event

            # assert match
            run_id = int(evth[cpw.I.EVTH.RUN_NUMBER])
            assert run_id == job["run_id"]
            event_id = event_idx + 1
            assert event_id == evth[cpw.I.EVTH.EVENT_NUMBER]
            uid = unique.make_uid(run_id=run_id, event_id=event_id)

            core = records_by_uid(records=tabrec["core"], uid=uid)

            ppp = {spt.IDX: uid}
            ppp["num_water_cherenkov"] = 0
            ppp["num_air_cherenkov"] = 0
            ppp["num_unknown"] = 0

            if core:
                aaa = {spt.IDX: uid}
                aaa["num_air_cherenkov_on_aperture"] = 0

            for parblock in parreader:
                cer = analysis.particles_on_ground.mask_cherenkov_emission(
                    corsika_particles=parblock,
                    corsika_particle_zoo=zoo,
                )
                ppp["num_water_cherenkov"] += np.sum(cer["media"]["water"])
                ppp["num_air_cherenkov"] += np.sum(cer["media"]["air"])
                ppp["num_unknown"] += np.sum(cer["unknown"])

                if core:
                    subparblock = parblock[cer["media"]["air"]]
                    subparblock_r_m = analysis.particles_on_ground.distances_to_point_on_observation_level_m(
                        corsika_particles=subparblock,
                        x_m=core["core_x_m"],
                        y_m=core["core_y_m"],
                    )
                    mask_on_aperture = subparblock_r_m <= aperture_radius_m
                    aaa["num_air_cherenkov_on_aperture"] += np.sum(
                        mask_on_aperture
                    )

            tabrec["particlepool"].append(ppp)
            if core:
                tabrec["particlepoolonaperture"].append(aaa)
    return tabrec


def _run_merlict(job, cherenkov_pools_path, tmp_dir):
    detector_responses_path = op.join(tmp_dir, "detector_responses")
    if not op.exists(detector_responses_path):
        merlict_rc = merlict_development_kit_python.plenoscope_propagator.plenoscope_propagator(
            corsika_run_path=cherenkov_pools_path,
            output_path=detector_responses_path,
            light_field_geometry_path=job["light_field_geometry_path"],
            merlict_plenoscope_propagator_path=job[
                "merlict_plenoscope_propagator_path"
            ],
            merlict_plenoscope_propagator_config_path=job[
                "merlict_plenoscope_propagator_config_path"
            ],
            random_seed=job["run_id"],
            photon_origins=True,
            stdout_path=op.join(tmp_dir, "merlict.stdout"),
            stderr_path=op.join(tmp_dir, "merlict.stderr"),
        )
        rnw.copy(
            op.join(tmp_dir, "merlict.stdout"),
            op.join(job["log_dir"], _run_id_str(job) + "_merlict.stdout"),
        )
        rnw.copy(
            op.join(tmp_dir, "merlict.stderr"),
            op.join(job["log_dir"], _run_id_str(job) + "_merlict.stderr"),
        )
        assert merlict_rc == 0

    return detector_responses_path


def _run_loose_trigger(
    job,
    tabrec,
    detector_responses_path,
    light_field_geometry,
    trigger_geometry,
    tmp_dir,
):
    # loop over sensor responses
    # --------------------------
    merlict_run = pl.Run(detector_responses_path)
    table_past_trigger = []
    tmp_past_trigger_dir = op.join(tmp_dir, "past_trigger")
    os.makedirs(tmp_past_trigger_dir, exist_ok=True)
    RAW_SKIP = int(job["raw_sensor_response"]["skip_num_events"])
    assert RAW_SKIP > 0

    for event in merlict_run:
        # id
        # --
        cevth = event.simulation_truth.event.corsika_event_header.raw
        run_id = int(cevth[cpw.I.EVTH.RUN_NUMBER])
        event_id = int(cevth[cpw.I.EVTH.EVENT_NUMBER])
        ide = {spt.IDX: unique.make_uid(run_id=run_id, event_id=event_id)}

        # export instrument's time relative to CORSIKA's time
        # ---------------------------------------------------
        ttabs = ide.copy()
        ttabs[
            "start_time_of_exposure_s"
        ] = event.simulation_truth.photon_propagator.nsb_exposure_start_time()
        tabrec["instrument"].append(ttabs)

        # apply loose trigger
        # -------------------
        (
            trigger_responses,
            max_response_in_focus_vs_timeslices,
        ) = pl.trigger.estimate.first_stage(
            raw_sensor_response=event.raw_sensor_response,
            light_field_geometry=light_field_geometry,
            trigger_geometry=trigger_geometry,
            integration_time_slices=(
                job["sum_trigger"]["integration_time_slices"]
            ),
        )

        trg_resp_path = op.join(event._path, "refocus_sum_trigger.json")
        with open(trg_resp_path, "wt") as f:
            f.write(json_utils.dumps(trigger_responses, indent=4))

        trg_maxr_path = op.join(
            event._path, "refocus_sum_trigger.focii_x_time_slices.uint32"
        )
        with open(trg_maxr_path, "wb") as f:
            f.write(max_response_in_focus_vs_timeslices.tobytes())

        # export trigger-truth
        # --------------------
        trgtru = ide.copy()
        trgtru["num_cherenkov_pe"] = int(
            event.simulation_truth.detector.number_air_shower_pulses()
        )
        trgtru["response_pe"] = int(
            np.max([focus["response_pe"] for focus in trigger_responses])
        )
        for o in range(len(trigger_responses)):
            trgtru["focus_{:02d}_response_pe".format(o)] = int(
                trigger_responses[o]["response_pe"]
            )
        tabrec["trigger"].append(trgtru)

        # passing loose trigger
        # ---------------------
        if trgtru["response_pe"] >= job["sum_trigger"]["threshold_pe"]:
            ptp = ide.copy()
            ptp["tmp_path"] = event._path
            ptp["unique_id_str"] = unique.UID_FOTMAT_STR.format(ptp[spt.IDX])
            table_past_trigger.append(ptp)

            ptrg = ide.copy()
            tabrec["pasttrigger"].append(ptrg)

            # export past loose trigger
            # -------------------------
            if ide[spt.IDX] % RAW_SKIP == 0:
                pl.tools.acp_format.compress_event_in_place(ptp["tmp_path"])
                final_tarname = ptp["unique_id_str"] + ".tar"
                plenoscope_event_dir_to_tar(
                    event_dir=ptp["tmp_path"],
                    output_tar_path=op.join(
                        tmp_past_trigger_dir, final_tarname
                    ),
                )
                rnw.copy(
                    src=op.join(tmp_past_trigger_dir, final_tarname),
                    dst=op.join(job["past_trigger_dir"], final_tarname),
                )

    return tabrec, table_past_trigger, tmp_past_trigger_dir


def _classify_cherenkov_photons(
    job,
    tabrec,
    tmp_dir,
    table_past_trigger,
    light_field_geometry,
    trigger_geometry,
):
    roi_cfg = job["cherenkov_classification"]["region_of_interest"]
    dbscan_cfg = job["cherenkov_classification"]

    with pl.photon_stream.loph.LopfTarWriter(
        path=os.path.join(tmp_dir, "reconstructed_cherenkov.tar"),
        uid_num_digits=unique.UID_NUM_DIGITS,
    ) as cer_phs_run:
        for ptp in table_past_trigger:
            event = pl.Event(
                path=ptp["tmp_path"], light_field_geometry=light_field_geometry
            )
            trigger_responses = pl.trigger.io.read_trigger_response_from_path(
                path=os.path.join(event._path, "refocus_sum_trigger.json")
            )
            roi = pl.trigger.region_of_interest.from_trigger_response(
                trigger_response=trigger_responses,
                trigger_geometry=trigger_geometry,
                time_slice_duration=event.raw_sensor_response[
                    "time_slice_duration"
                ],
            )
            photons = pl.classify.RawPhotons.from_event(event)
            (
                cherenkov_photons,
                roi_settings,
            ) = pl.classify.cherenkov_photons_in_roi_in_image(
                roi=roi,
                photons=photons,
                roi_time_offset_start=roi_cfg["time_offset_start_s"],
                roi_time_offset_stop=roi_cfg["time_offset_stop_s"],
                roi_cx_cy_radius=np.deg2rad(roi_cfg["direction_radius_deg"]),
                roi_object_distance_offsets=roi_cfg[
                    "object_distance_offsets_m"
                ],
                dbscan_epsilon_cx_cy_radius=np.deg2rad(
                    dbscan_cfg["neighborhood_radius_deg"]
                ),
                dbscan_min_number_photons=dbscan_cfg["min_num_photons"],
                dbscan_deg_over_s=dbscan_cfg[
                    "direction_to_time_mixing_deg_per_s"
                ],
            )
            pl.classify.write_dense_photon_ids_to_event(
                event_path=op.abspath(event._path),
                photon_ids=cherenkov_photons.photon_ids,
                settings=roi_settings,
            )
            crcl = pl.classify.benchmark(
                pulse_origins=event.simulation_truth.detector.pulse_origins,
                photon_ids_cherenkov=cherenkov_photons.photon_ids,
            )
            crcl[spt.IDX] = ptp[spt.IDX]
            tabrec["cherenkovclassification"].append(crcl)

            # export reconstructed Cherenkov photons
            # --------------------------------------
            cer_phs = pl.photon_stream.loph.raw_sensor_response_to_photon_stream_in_loph_repr(
                raw_sensor_response=event.raw_sensor_response,
                cherenkov_photon_ids=cherenkov_photons.photon_ids,
            )
            cer_phs_run.add(uid=ptp[spt.IDX], phs=cer_phs)

    rnw.copy(
        src=op.join(tmp_dir, "reconstructed_cherenkov.tar"),
        dst=op.join(
            job["past_trigger_reconstructed_cherenkov_dir"],
            _run_id_str(job=job) + "_reconstructed_cherenkov.tar",
        ),
    )
    return tabrec


def _extract_features(
    tabrec,
    light_field_geometry,
    table_past_trigger,
    prng,
):
    light_field_geometry_addon = pl.features.make_light_field_geometry_addon(
        light_field_geometry=light_field_geometry
    )

    for pt in table_past_trigger:
        event = pl.Event(
            path=pt["tmp_path"], light_field_geometry=light_field_geometry
        )
        try:
            lfft = pl.features.extract_features(
                cherenkov_photons=event.cherenkov_photons,
                light_field_geometry=light_field_geometry,
                light_field_geometry_addon=light_field_geometry_addon,
                prng=prng,
            )
            lfft[spt.IDX] = pt[spt.IDX]
            tabrec["features"].append(lfft)
        except Exception as excep:
            print("idx:", pt[spt.IDX], excep)

    return tabrec


def _estimate_primary_trajectory(job, tmp_dir, light_field_geometry, tabrec):
    FUZZY_CONFIG = (
        gamrec.trajectory.v2020nov12fuzzy0.config.compile_user_config(
            user_config=job["reconstruction"]["trajectory"]["fuzzy_method"]
        )
    )
    MODEL_FIT_CONFIG = (
        gamrec.trajectory.v2020dec04iron0b.config.compile_user_config(
            user_config=job["reconstruction"]["trajectory"]["core_axis_fit"]
        )
    )

    _feature_table = spt.table_of_records_to_sparse_numeric_table(
        table_records={"features": tabrec["features"]},
        structure={"features": table.STRUCTURE["features"]},
    )
    shower_maximum_object_distance = spt.get_column_as_dict_by_index(
        table=_feature_table,
        level_key="features",
        column_key="image_smallest_ellipse_object_distance",
    )

    run = pl.photon_stream.loph.LopfTarReader(
        op.join(tmp_dir, "reconstructed_cherenkov.tar")
    )
    for event in run:
        uid, loph_record = event

        if uid in shower_maximum_object_distance:
            estimate, debug = gamrec.trajectory.v2020dec04iron0b.estimate(
                loph_record=loph_record,
                light_field_geometry=light_field_geometry,
                shower_maximum_object_distance=shower_maximum_object_distance[
                    uid
                ],
                fuzzy_config=FUZZY_CONFIG,
                model_fit_config=MODEL_FIT_CONFIG,
            )

            if gamrec.trajectory.v2020dec04iron0b.is_valid_estimate(
                estimate=estimate
            ):
                rec = {}
                rec[spt.IDX] = uid

                rec["cx_rad"] = estimate["primary_particle_cx"]
                rec["cy_rad"] = estimate["primary_particle_cy"]
                rec["x_m"] = estimate["primary_particle_x"]
                rec["y_m"] = estimate["primary_particle_y"]

                rec["fuzzy_cx_rad"] = debug["fuzzy_result"]["reco_cx"]
                rec["fuzzy_cy_rad"] = debug["fuzzy_result"]["reco_cy"]
                rec["fuzzy_main_axis_support_cx_rad"] = debug["fuzzy_result"][
                    "main_axis_support_cx"
                ]
                rec["fuzzy_main_axis_support_cy_rad"] = debug["fuzzy_result"][
                    "main_axis_support_cy"
                ]
                rec["fuzzy_main_axis_support_uncertainty_rad"] = debug[
                    "fuzzy_result"
                ]["main_axis_support_uncertainty"]
                rec["fuzzy_main_axis_azimuth_rad"] = debug["fuzzy_result"][
                    "main_axis_azimuth"
                ]
                rec["fuzzy_main_axis_azimuth_uncertainty_rad"] = debug[
                    "fuzzy_result"
                ]["main_axis_azimuth_uncertainty"]

                tabrec["reconstructed_trajectory"].append(rec)

    return tabrec


def _assert_resources_exist(job):
    assert op.exists(job["corsika_primary_path"])
    assert op.exists(job["merlict_plenoscope_propagator_path"])
    assert op.exists(job["merlict_plenoscope_propagator_config_path"])
    assert op.exists(job["plenoscope_scenery_path"])
    assert op.exists(job["light_field_geometry_path"])
    assert op.exists(job["trigger_geometry_path"])


def _make_output_dirs(job):
    os.makedirs(job["log_dir"], exist_ok=True)
    os.makedirs(job["past_trigger_dir"], exist_ok=True)
    os.makedirs(job["past_trigger_reconstructed_cherenkov_dir"], exist_ok=True)
    os.makedirs(job["feature_dir"], exist_ok=True)
    os.makedirs(job["particles_dir"], exist_ok=True)


def _export_event_table(job, tmp_dir, tabrec):
    event_table = spt.table_of_records_to_sparse_numeric_table(
        table_records=tabrec, structure=table.STRUCTURE
    )
    spt.write(
        path=op.join(tmp_dir, "event_table.tar"),
        table=event_table,
        structure=table.STRUCTURE,
    )
    rnw.copy(
        src=op.join(tmp_dir, "event_table.tar"),
        dst=op.join(job["feature_dir"], _run_id_str(job) + "_event_table.tar"),
    )


def _export_grid_region_of_interest_if_passed_loose_trigger(
    job, tabrec, tmp_dir
):
    pasttrigger_set = set(
        [record[spt.IDX] for record in tabrec["pasttrigger"]]
    )

    ipath = op.join(tmp_dir, "grid_roi.tar")
    opath = op.join(tmp_dir, "grid_roi_pasttrigger.tar")

    with tarfile.open(ipath, "r") as itar, tarfile.open(opath, "w") as otar:
        for tarinfo in itar:
            idx = int(tarinfo.name[0 : unique.UID_NUM_DIGITS])
            if idx in pasttrigger_set:
                bimg = itar.extractfile(tarinfo).read()
                filename = unique.UID_FOTMAT_STR.format(idx) + ".f4.gz"
                utils.tar_append(
                    tarout=otar,
                    file_name=filename,
                    file_bytes=bimg,
                )
    rnw.copy(
        src=opath,
        dst=op.join(
            job["feature_dir"], _run_id_str(job) + "_grid_roi_pasttrigger.tar"
        ),
    )


def _init_table_records():
    tabrec = {}
    for level_key in table.STRUCTURE:
        tabrec[level_key] = []
    return tabrec


def _export_job_to_log_dir(job):
    job_path = op.join(job["log_dir"], _run_id_str(job) + "_job.json")
    with open(job_path + ".tmp", "wt") as f:
        f.write(json_utils.dumps(job, indent=4))
    rnw.move(job_path + ".tmp", job_path)


def run_job(job):
    _assert_resources_exist(job=job)
    _make_output_dirs(job=job)
    _export_job_to_log_dir(job=job)

    log_path = op.join(job["log_dir"], _run_id_str(job) + "_runtime.jsonl")
    logger = jlogging.LoggerFile(path=log_path + ".tmp")
    logger.info("starting run")

    logger.info("init prng")
    prng = np.random.Generator(np.random.MT19937(seed=job["run_id"]))

    with jlogging.TimeDelta(logger, "draw_primary"):
        corsika_primary_steering = atmospheric_cherenkov_response.particles.draw_corsika_primary_steering(
            run_id=job["run_id"],
            site=job["site"],
            particle=job["particle"],
            site_particle_deflection=job["site_particle_deflection"],
            num_events=job["num_air_showers"],
            prng=prng,
        )

    if job["tmp_dir"] is None:
        tmp_dir = tempfile.mkdtemp(prefix="plenoscope_irf_")
    else:
        tmp_dir = op.join(job["tmp_dir"], _run_id_str(job))
        os.makedirs(tmp_dir, exist_ok=True)
    logger.info("make tmp_dir: {:s}".format(tmp_dir))

    tabrec = _init_table_records()

    with jlogging.TimeDelta(logger, "corsika_and_grid"):
        (
            cherenkov_pools_path,
            tabrec,
        ) = _run_corsika_and_grid_and_output_to_tmp_dir(
            job=job,
            prng=prng,
            tmp_dir=tmp_dir,
            corsika_primary_steering=corsika_primary_steering,
            tabrec=tabrec,
        )

    with jlogging.TimeDelta(logger, "particlepool"):
        tabrec = _populate_particlepool(job, tabrec)

    with jlogging.TimeDelta(logger, "merlict"):
        detector_responses_path = _run_merlict(
            job=job,
            cherenkov_pools_path=cherenkov_pools_path,
            tmp_dir=tmp_dir,
        )

    if not job["keep_tmp"]:
        os.remove(cherenkov_pools_path)

    with jlogging.TimeDelta(logger, "read_geometry"):
        light_field_geometry = pl.LightFieldGeometry(
            path=job["light_field_geometry_path"]
        )
        trigger_geometry = pl.trigger.geometry.read(
            path=job["trigger_geometry_path"]
        )

    with jlogging.TimeDelta(logger, "pass_loose_trigger"):
        tabrec, table_past_trigger, tmp_past_trigger_dir = _run_loose_trigger(
            job=job,
            tabrec=tabrec,
            detector_responses_path=detector_responses_path,
            light_field_geometry=light_field_geometry,
            trigger_geometry=trigger_geometry,
            tmp_dir=tmp_dir,
        )

    with jlogging.TimeDelta(logger, "export grid region-of-interest"):
        _export_grid_region_of_interest_if_passed_loose_trigger(
            job=job,
            tabrec=tabrec,
            tmp_dir=tmp_dir,
        )

    with jlogging.TimeDelta(logger, "classify_cherenkov"):
        tabrec = _classify_cherenkov_photons(
            job=job,
            tabrec=tabrec,
            tmp_dir=tmp_dir,
            table_past_trigger=table_past_trigger,
            light_field_geometry=light_field_geometry,
            trigger_geometry=trigger_geometry,
        )

    with jlogging.TimeDelta(logger, "extract_features"):
        tabrec = _extract_features(
            tabrec=tabrec,
            light_field_geometry=light_field_geometry,
            table_past_trigger=table_past_trigger,
            prng=prng,
        )

    with jlogging.TimeDelta(logger, "estimate_primary_trajectory"):
        tabrec = _estimate_primary_trajectory(
            job=job,
            tmp_dir=tmp_dir,
            light_field_geometry=light_field_geometry,
            tabrec=tabrec,
        )

    with jlogging.TimeDelta(logger, "export_event_table"):
        _export_event_table(job=job, tmp_dir=tmp_dir, tabrec=tabrec)

    if not job["keep_tmp"]:
        shutil.rmtree(tmp_dir)
    logger.info("ending run")
    rnw.move(log_path + ".tmp", log_path)
