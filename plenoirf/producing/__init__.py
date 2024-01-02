import os
from os import path as op
from os.path import join as opj
import tempfile
import copy
import numpy as np
import tarfile
import gzip

import magnetic_deflection
import json_utils
import json_line_logger
import merlict_development_kit_python
import spherical_coordinates
import atmospheric_cherenkov_response as acr
import rename_after_writing as rnw
import sparse_numeric_table as spt
import corsika_primary as cpw
import homogeneous_transformation

from .. import outer_telescope_array
from .. import bookkeeping
from .. import configurating
from .. import ground_grid
from .. import event_table
from .. import tar_append
from .. import debugging
from .. import constants

from . import sum_trigger
from . import transform_cherenkov_bunches
from . import draw_primary_and_pointing
from . import draw_pointing_range


def make_example_job(
    plenoirf_dir,
    run_id=1337,
    site_key="chile",
    particle_key="electron",
    instrument_key="diag9_default_default",
    num_events=128,
    max_num_events_in_merlict_run=12,
):
    job = {}
    job["run_id"] = run_id
    job["plenoirf_dir"] = plenoirf_dir
    job["site_key"] = site_key
    job["particle_key"] = particle_key
    job["instrument_key"] = instrument_key
    job["num_events"] = num_events
    job["max_num_events_in_merlict_run"] = max_num_events_in_merlict_run
    return job


def run_job(job):
    with tempfile.TemporaryDirectory(suffix="-plenoirf") as tmp_dir:
        return run_job_in_dir(job=job, tmp_dir=tmp_dir)


def run_job_in_dir(job, tmp_dir):
    job = compile_job(job=job, tmp_dir=tmp_dir)

    os.makedirs(job["paths"]["stage_dir"], exist_ok=True)

    logger = json_line_logger.LoggerFile(path=job["paths"]["logger_tmp"])
    logger.info("starting")

    logger.debug("making tmp_dir: {:s}".format(job["paths"]["tmp_dir"]))
    os.makedirs(job["paths"]["tmp_dir"], exist_ok=True)

    logger.debug("initializing prng with seed: {:d}".format(job["run_id"]))
    prng = np.random.Generator(np.random.PCG64(seed=job["run_id"]))

    run = {}

    run["event_ids_for_debug"] = debugging.draw_event_ids_for_debug_output(
        num_events_in_run=job["num_events"],
        min_num_events=job["config"]["debug_output"]["run"]["min_num_events"],
        fraction_of_events=job["config"]["debug_output"]["run"][
            "fraction_of_events"
        ],
        prng=prng,
    )
    logger.debug(
        "event-ids for debugging: {:s}.".format(
            str(run["event_ids_for_debug"].tolist())
        )
    )

    with json_line_logger.TimeDelta(logger, "draw_pointing_range"):
        job, run = draw_pointing_range.run_job(job=job, run=run, prng=prng)


    with json_line_logger.TimeDelta(logger, "draw_primaries_and_pointings"):
        job, run = draw_primaries_and_pointings.run_job(
            job=job,
            run=run,
            prng=prng,
            logger=logger,
        )

    tabrec = event_table.structure.init_table_dynamicsizerecarray()

    with json_line_logger.TimeDelta(logger, "corsika_and_grid"):
        job, run, tabrec = _corsika_and_grid(
            job=job,
            run=run,
            tabrec=tabrec,
            prng=prng,
            logger=logger,
        )

    with rnw.open(opj(job["paths"]["tmp_dir"], "job.json"), "wt") as f:
        f.write(json_utils.dumps(job, indent=4))

    logger.info("ending")
    rnw.move(job["paths"]["logger_tmp"], job["paths"]["logger"])
    return job, run, tabrec


def read_light_field_camera_config(plenoirf_dir, instrument_key):
    return merlict_development_kit_python.plenoscope_propagator.read_plenoscope_geometry(
        merlict_scenery_path=opj(
            plenoirf_dir,
            "plenoptics",
            "instruments",
            instrument_key,
            "light_field_geometry",
            "input",
            "scenery",
            "scenery.json",
        )
    )


def compile_job(job, tmp_dir):
    """
    Adds all kind of static information to the job dict.
    Items in the job dict are not meant to change!
    It is meant to be read only.
    """
    assert job["max_num_events_in_merlict_run"] > 0

    job["run_id_str"] = bookkeeping.uid.make_run_id_str(run_id=job["run_id"])
    job["paths"] = compile_paths(job=job, tmp_dir=tmp_dir)
    job["config"] = configurating.read(
        plenoirf_dir=job["paths"]["plenoirf_dir"]
    )

    _allskycfg = json_utils.tree.read(
        opj(job["paths"]["magnetic_deflection_allsky"], "config")
    )
    job["site"] = copy.deepcopy(_allskycfg["site"])
    job["particle"] = copy.deepcopy(_allskycfg["particle"])

    job["light_field_camera_config"] = read_light_field_camera_config(
        plenoirf_dir=job["paths"]["plenoirf_dir"],
        instrument_key=job["instrument_key"],
    )
    job["instrument"] = {}
    job["instrument"]["field_of_view_half_angle_rad"] = 0.5 * (
        np.deg2rad(job["light_field_camera_config"]["max_FoV_diameter_deg"])
    )
    job["instrument"]["local_speed_of_light_m_per_s"] = (
        constants.speed_of_light_in_vacuum_m_per_s()
        / job["site"]["atmosphere_refractive_index_at_observation_level"]
    )
    return job


def compile_paths(job, tmp_dir):
    paths = {}
    paths["plenoirf_dir"] = job["plenoirf_dir"]

    paths["stage_dir"] = opj(
        job["plenoirf_dir"],
        "response",
        job["instrument_key"],
        job["site_key"],
        job["particle_key"],
        "stage",
    )

    # logger
    # ------
    paths["logger"] = opj(paths["stage_dir"], job["run_id_str"] + "_log.jsonl")
    paths["logger_tmp"] = paths["logger"] + ".tmp"

    # input
    # -----
    paths["magnetic_deflection_allsky"] = opj(
        job["plenoirf_dir"],
        "magnetic_deflection",
        job["site_key"],
        job["particle_key"],
    )
    paths["light_field_calibration"] = opj(
        job["plenoirf_dir"],
        "plenoptics",
        "instruments",
        job["instrument_key"],
        "light_field_geometry",
    )

    # temporary
    # ---------
    paths["tmp_dir"] = tmp_dir
    paths["tmp"] = {}

    paths["tmp"]["cherenkov_pools"] = opj(tmp_dir, "cherenkov_pools.tar")
    paths["tmp"]["cherenkov_pools_block_fmt"] = opj(
        tmp_dir, "cherenkov_pools_{block:06d}.tar"
    )
    paths["tmp"]["particle_pools_dat"] = opj(tmp_dir, "particle_pools.dat")
    paths["tmp"]["particle_pools_tar"] = opj(tmp_dir, "particle_pools.tar.gz")

    paths["tmp"]["ground_grid_intensity"] = opj(tmp_dir, "ground_grid.tar")
    paths["tmp"]["ground_grid_intensity_roi"] = opj(
        tmp_dir, "ground_grid_roi.tar"
    )

    paths["tmp"]["corsika_stdout"] = opj(tmp_dir, "corsika.stdout")
    paths["tmp"]["corsika_stderr"] = opj(tmp_dir, "corsika.stderr")
    paths["tmp"]["merlict_stdout"] = opj(tmp_dir, "merlict.stdout")
    paths["tmp"]["merlict_stderr"] = opj(tmp_dir, "merlict.stderr")

    # debug cache
    # -----------
    paths["cache"] = {}
    paths["cache"]["primary"] = opj(
        tmp_dir, job["run_id_str"] + "_corsika_primary_steering.tar"
    )

    # debug output
    # ------------
    paths["debug"] = {}
    paths["debug"]["draw_primary_and_pointing"] = opj(
        paths["stage_dir"],
        job["run_id_str"] + "_debug_" + "draw_primary_and_pointing" + ".tar",
    )

    return paths


def _corsika_and_grid(
    job,
    run,
    tabrec,
    prng,
    logger,
):
    if not op.exists(job["paths"]["tmp"]["cherenkov_pools"]):
        job, run, tabrec = __corsika_and_grid(
            job=job,
            run=run,
            tabrec=tabrec,
            prng=prng,
            logger=logger,
        )

    cpw.particles.dat_to_tape(
        dat_path=job["paths"]["tmp"]["particle_pools_dat"],
        tape_path=job["paths"]["tmp"]["particle_pools_tar"],
    )

    run["cherenkov_pools"] = event_tape_block_splitter(
        inpath=job["paths"]["tmp"]["cherenkov_pools"],
        outpath_block_fmt=job["paths"]["tmp"]["cherenkov_pools_block_fmt"],
        num_events=job["max_num_events_in_merlict_run"],
    )
    return job, run, tabrec


def __corsika_and_grid(
    job,
    run,
    tabrec,
    prng,
    logger,
):
    with cpw.cherenkov.CherenkovEventTapeWriter(
        path=job["paths"]["tmp"]["cherenkov_pools"]
    ) as evttar, tarfile.open(
        job["paths"]["tmp"]["ground_grid_intensity"], "w"
    ) as imgtar, tarfile.open(
        job["paths"]["tmp"]["ground_grid_intensity_roi"], "w"
    ) as imgroitar:
        with cpw.CorsikaPrimary(
            corsika_path=job["config"]["executables"]["corsika_primary_path"],
            steering_dict=run["corsika_primary_steering"],
            stdout_path=job["paths"]["tmp"]["corsika_stdout"],
            stderr_path=job["paths"]["tmp"]["corsika_stderr"],
            particle_output_path=job["paths"]["tmp"]["particle_pools_dat"],
        ) as corsika_run:
            evttar.write_runh(runh=corsika_run.runh)

            for event_idx, corsika_event in enumerate(corsika_run):
                corsika_evth, cherenkov_reader = corsika_event

                cherenkov_bunches = read_all_cherenkov_bunches(
                    cherenkov_reader=cherenkov_reader
                )

                uid = nail_down_event_identity(
                    corsika_evth=corsika_evth,
                    event_idx=event_idx,
                    corsika_primary_steering=run["corsika_primary_steering"],
                )

                primary_rec = make_primary_record(
                    uid=uid,
                    corsika_evth=corsika_evth,
                    corsika_primary_steering=run["corsika_primary_steering"],
                    primary_directions=run["primary_directions"],
                )
                tabrec["primary"].append_record(primary_rec)

                pointing_rec = make_pointing_record(
                    uid=uid, pointings=run["pointings"]
                )
                tabrec["pointing"].append_record(pointing_rec)
                _ = pointing_rec.pop("idx")
                pointing = pointing_rec

                cherenkovsize_rec = make_cherenkovsize_record(
                    uid=uid,
                    cherenkov_bunches=cherenkov_bunches,
                )
                tabrec["cherenkovsize"].append_record(cherenkovsize_rec)

                cherenkov_pool_median_x_m = 0.0
                cherenkov_pool_median_y_m = 0.0
                if cherenkovsize_rec["num_bunches"] > 0:
                    cherenkovpool_rec = make_cherenkovpool_record(
                        uid=uid,
                        cherenkov_bunches=cherenkov_bunches,
                    )
                    tabrec["cherenkovpool"].append_record(cherenkovpool_rec)
                    cherenkov_pool_median_x_m = cherenkovpool_rec["x_median_m"]
                    cherenkov_pool_median_y_m = cherenkovpool_rec["y_median_m"]

                groundgrid_config = ground_grid.make_ground_grid_config(
                    bin_width_m=job["config"]["ground_grid"]["geometry"][
                        "bin_width_m"
                    ],
                    num_bins_each_axis=job["config"]["ground_grid"][
                        "geometry"
                    ]["num_bins_each_axis"],
                    cherenkov_pool_median_x_m=cherenkov_pool_median_x_m,
                    cherenkov_pool_median_y_m=cherenkov_pool_median_y_m,
                    prng=prng,
                )

                groundgrid = ground_grid.GroundGrid(
                    bin_width_m=groundgrid_config["bin_width_m"],
                    num_bins_each_axis=groundgrid_config["num_bins_each_axis"],
                    center_x_m=groundgrid_config["center_x_m"],
                    center_y_m=groundgrid_config["center_y_m"],
                )

                fov_mask = mask_cherenkov_bunches_in_instruments_field_of_view(
                    cherenkov_bunches=cherenkov_bunches,
                    pointing=pointing,
                    field_of_view_half_angle_rad=job["instrument"][
                        "field_of_view_half_angle_rad"
                    ],
                )
                cherenkov_bunches_in_fov = cherenkov_bunches[fov_mask]
                del cherenkov_bunches

                groundgrid_result, groundgrid_debug = ground_grid.assign(
                    groundgrid=groundgrid,
                    cherenkov_bunches=cherenkov_bunches_in_fov,
                    threshold_num_photons=job["config"]["ground_grid"][
                        "threshold_num_photons"
                    ],
                    prng=prng,
                )

                groundgrid_rec = make_groundgrid_record(
                    uid=uid,
                    groundgrid_config=groundgrid_config,
                    groundgrid_result=groundgrid_result,
                    groundgrid=groundgrid,
                )
                tabrec["groundgrid"].append_record(groundgrid_rec)

                if groundgrid_result["choice"]:
                    cherenkov_bunches_in_choice = cherenkov_bunches_in_fov[
                        groundgrid_result["choice"]["cherenkov_bunches_idxs"]
                    ]
                    del cherenkov_bunches_in_fov

                    cherenkov_bunches_in_instrument = transform_cherenkov_bunches.from_obervation_level_to_instrument(
                        cherenkov_bunches=cherenkov_bunches_in_choice,
                        instrument_pointing=pointing,
                        instrument_pointing_model=job["config"]["pointing"][
                            "model"
                        ],
                        instrument_x_m=groundgrid_result["choice"]["core_x_m"],
                        instrument_y_m=groundgrid_result["choice"]["core_y_m"],
                        speed_of_ligth_m_per_s=job["instrument"][
                            "local_speed_of_light_m_per_s"
                        ],
                    )
                    del cherenkov_bunches_in_choice

                    core_rec = make_core_record(
                        uid=uid,
                        groundgrid_result_choice=groundgrid_result["choice"],
                    )
                    tabrec["core"].append_record(core_rec)

                    EventTape_append_event(
                        evttar=evttar,
                        corsika_evth=corsika_evth,
                        cherenkov_bunches=cherenkov_bunches_in_instrument,
                        core_x_m=groundgrid_result["choice"]["core_x_m"],
                        core_y_m=groundgrid_result["choice"]["core_y_m"],
                    )

                    ImgRoiTar_append(
                        imgroitar=imgroitar,
                        uid=uid,
                        groundgrid_result=groundgrid_result,
                        groundgrid_debug=groundgrid_debug,
                    )

                    cherenkovsizepart_rec = make_cherenkovsize_record(
                        uid=uid,
                        cherenkov_bunches=cherenkov_bunches_in_instrument,
                    )
                    tabrec["cherenkovsizepart"].append_record(
                        cherenkovsizepart_rec
                    )

                    if cherenkovsizepart_rec["num_bunches"] > 0:
                        cherenkovpoolpart_rec = make_cherenkovpool_record(
                            uid=uid,
                            cherenkov_bunches=cherenkov_bunches_in_instrument,
                        )
                        tabrec["cherenkovpoolpart"].append_record(
                            cherenkovpoolpart_rec
                        )

                with open(
                    opj(
                        job["paths"]["tmp_dir"],
                        uid["uid_str"] + "_ground_grid.json",
                    ),
                    "wt",
                ) as f:
                    f.write(json_utils.dumps(groundgrid_result))

                print(uid)

    return job, run, tabrec


def nail_down_event_identity(
    corsika_evth, corsika_primary_steering, event_idx
):
    run_id = int(corsika_evth[cpw.I.EVTH.RUN_NUMBER])
    assert run_id == corsika_primary_steering["run"]["run_id"]
    event_id = event_idx + 1
    assert event_id == corsika_evth[cpw.I.EVTH.EVENT_NUMBER]
    uid = bookkeeping.uid.make_uid(run_id=run_id, event_id=event_id)
    uid_str = bookkeeping.uid.make_uid_str(run_id=run_id, event_id=event_id)

    out = {
        "record": {spt.IDX: uid},
        "uid": uid,
        "uid_str": uid_str,
        "run_id": run_id,
        "event_id": event_id,
        "event_idx": event_idx,
    }
    return out


def make_primary_record(
    uid, corsika_evth, corsika_primary_steering, primary_directions
):
    primary = corsika_primary_steering["primaries"][uid["event_idx"]]
    primary_direction = primary_directions[uid["event_idx"]]

    rec = uid["record"].copy()
    rec["particle_id"] = primary["particle_id"]
    rec["energy_GeV"] = primary["energy_GeV"]
    rec["azimuth_rad"] = primary["azimuth_rad"]
    rec["zenith_rad"] = primary["zenith_rad"]
    rec["depth_g_per_cm2"] = primary["depth_g_per_cm2"]

    rec["momentum_x_GeV_per_c"] = corsika_evth[
        cpw.I.EVTH.PX_MOMENTUM_GEV_PER_C
    ]
    rec["momentum_y_GeV_per_c"] = corsika_evth[
        cpw.I.EVTH.PY_MOMENTUM_GEV_PER_C
    ]
    rec["momentum_z_GeV_per_c"] = (
        -1.0 * corsika_evth[cpw.I.EVTH.PZ_MOMENTUM_GEV_PER_C]
    )

    rec["starting_height_asl_m"] = (
        cpw.CM2M * corsika_evth[cpw.I.EVTH.STARTING_HEIGHT_CM]
    )
    obs_lvl_intersection = acr.utils.ray_plane_x_y_intersection(
        support=[0, 0, rec["starting_height_asl_m"]],
        direction=[
            rec["momentum_x_GeV_per_c"],
            rec["momentum_y_GeV_per_c"],
            rec["momentum_z_GeV_per_c"],
        ],
        plane_z=corsika_primary_steering["run"]["observation_level_asl_m"],
    )

    rec["starting_x_m"] = -1.0 * obs_lvl_intersection[0]
    rec["starting_y_m"] = -1.0 * obs_lvl_intersection[1]

    rec["first_interaction_height_asl_m"] = (
        -1.0 * cpw.CM2M * corsika_evth[cpw.I.EVTH.Z_FIRST_INTERACTION_CM]
    )

    rec["solid_angle_thrown_sr"] = primary_direction["solid_angle_thrown_sr"]
    rec["inner_atmopsheric_magnetic_cutoff"] = primary_direction["cutoff"]

    return rec


def make_pointing_record(uid, pointings):
    rec = uid["record"].copy()
    for key in ["azimuth_rad", "zenith_rad"]:
        rec[key] = pointings[uid["event_idx"]][key]
    return rec


def make_cherenkovsize_record(uid, cherenkov_bunches):
    rec = uid["record"].copy()
    rec["num_bunches"] = cherenkov_bunches.shape[0]
    rec["num_photons"] = np.sum(cherenkov_bunches[:, cpw.I.BUNCH.BUNCH_SIZE_1])
    return rec


def make_cherenkovpool_record(uid, cherenkov_bunches):
    rec = uid["record"].copy()
    cb = cherenkov_bunches
    assert cb.shape[0] > 0
    rec["maximum_asl_m"] = cpw.CM2M * np.median(
        cb[:, cpw.I.BUNCH.EMISSOION_ALTITUDE_ASL_CM]
    )
    rec["wavelength_median_nm"] = np.abs(
        np.median(cb[:, cpw.I.BUNCH.WAVELENGTH_NM])
    )
    rec["cx_median_rad"] = np.median(cb[:, cpw.I.BUNCH.CX_RAD])
    rec["cy_median_rad"] = np.median(cb[:, cpw.I.BUNCH.CY_RAD])
    rec["x_median_m"] = cpw.CM2M * np.median(cb[:, cpw.I.BUNCH.X_CM])
    rec["y_median_m"] = cpw.CM2M * np.median(cb[:, cpw.I.BUNCH.Y_CM])
    rec["bunch_size_median"] = np.median(cb[:, cpw.I.BUNCH.BUNCH_SIZE_1])
    return rec


def make_groundgrid_record(
    uid, groundgrid_config, groundgrid_result, groundgrid
):
    rec = uid["record"].copy()
    rec.update(groundgrid_config)

    rec["num_bins_thrown"] = groundgrid["num_bins_thrown"]
    rec["num_bins_above_threshold"] = groundgrid_result[
        "num_bins_above_threshold"
    ]
    rec["area_thrown_m2"] = groundgrid["area_thrown_m2"]

    rec["num_photons_overflow"] = groundgrid_result["num_photons_overflow"]

    # compare scatter
    # ---------------
    scathist = groundgrid_result["scatter_histogram"]
    num_bins = len(scathist["bin_counts"])
    assert num_bins == 16
    for rbin in range(num_bins):
        rec["scatter_rbin_{:02d}".format(rbin)] = scathist["bin_counts"][rbin]
    return rec


def make_core_record(uid, groundgrid_result_choice):
    rec = uid["record"].copy()
    for key in event_table.init_core_level_structure():
        rec[key] = groundgrid_result_choice[rec]
    return rec


def read_all_cherenkov_bunches(cherenkov_reader):
    return np.vstack([b for b in cherenkov_reader])


def mask_cherenkov_bunches_in_instruments_field_of_view(
    cherenkov_bunches,
    pointing,
    field_of_view_half_angle_rad,
):
    OVERHEAD = 2.0
    return mask_cherenkov_bunches_in_cone(
        cherenkov_bunches_cx=cherenkov_bunches[:, cpw.I.BUNCH.CX_RAD],
        cherenkov_bunches_cy=cherenkov_bunches[:, cpw.I.BUNCH.CY_RAD],
        cone_azimuth_rad=pointing["azimuth_rad"],
        cone_zenith_rad=pointing["zenith_rad"],
        cone_half_angle_rad=OVERHEAD * field_of_view_half_angle_rad,
    )


def mask_cherenkov_bunches_in_cone(
    cherenkov_bunches_cx,
    cherenkov_bunches_cy,
    cone_half_angle_rad,
    cone_azimuth_rad,
    cone_zenith_rad,
):
    cone_cx, cone_cy = spherical_coordinates.az_zd_to_cx_cy(
        azimuth_rad=cone_azimuth_rad,
        zenith_rad=cone_zenith_rad,
    )
    delta_rad = spherical_coordinates.angle_between_cx_cy(
        cx1=cherenkov_bunches_cx,
        cy1=cherenkov_bunches_cy,
        cx2=cone_cx,
        cy2=cone_cy,
    )
    return delta_rad < cone_half_angle_rad


def EventTape_append_event(
    evttar,
    corsika_evth,
    cherenkov_bunches,
    core_x_m,
    core_y_m,
):
    evth = corsika_evth.copy()
    evth[cpw.I.EVTH.NUM_REUSES_OF_CHERENKOV_EVENT] = 1.0
    evth[cpw.I.EVTH.X_CORE_CM(reuse=1)] = cpw.M2CM * core_x_m
    evth[cpw.I.EVTH.Y_CORE_CM(reuse=1)] = cpw.M2CM * core_y_m
    evttar.write_evth(evth=evth)
    evttar.write_payload(payload=cherenkov_bunches)


def ImgRoiTar_append(imgroitar, uid, groundgrid_result, groundgrid_debug):
    bb = outer_telescope_array.init_binning()
    roi_array = ground_grid.bin_photon_assignment_to_array_roi(
        bin_photon_assignment=groundgrid_debug["bin_photon_assignment"],
        x_bin=groundgrid_result["choice"]["bin_idx_x"],
        y_bin=groundgrid_result["choice"]["bin_idx_y"],
        r_bin=bb["num_bins_radius"],
        dtype=np.float32,
    )
    tar_append.tar_append(
        tarout=imgroitar,
        filename=uid["uid_str"] + ".f4.gz",
        filebytes=ground_grid.io.histogram_to_bytes(roi_array),
    )




def event_tape_block_splitter(inpath, outpath_block_fmt, num_events):
    Writer = cpw.cherenkov.CherenkovEventTapeWriter
    Reader = cpw.cherenkov.CherenkovEventTapeReader
    outpaths = {}

    orun = None
    block = 0
    event_counter = 0
    with Reader(inpath) as irun:
        runh = copy.deepcopy(irun.runh)

        for event in irun:
            evth, cherenkov_reader = event
            cherenkov_bunches = read_all_cherenkov_bunches(cherenkov_reader)

            if event_counter % num_events == 0:
                block += 1
                if orun is not None:
                    orun.close()
                outpaths[block] = opj(outpath_block_fmt.format(block=block))
                orun = Writer(outpaths[block])
                orun.write_runh(runh)

            orun.write_evth(evth=evth)
            orun.write_payload(payload=cherenkov_bunches)
            event_counter += 1

        if orun is not None:
            orun.close()

    return outpaths
