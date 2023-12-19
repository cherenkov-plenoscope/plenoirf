import os
from os import path as op
from os.path import join as opj
import tempfile
import copy
import numpy as np
import tarfile

import magnetic_deflection
import json_utils
import json_line_logger
import merlict_development_kit_python
import atmospheric_cherenkov_response as acr
import rename_after_writing as rnw
import sparse_numeric_table as spt
import corsika_primary as cpw

from .. import bookkeeping
from .. import configurating
from . import random
from . import sum_trigger


def make_jobs(production_dir):
    lock = magnetic_deflection.allsky.production.Production(
        os.path.join(path, "lock")
    )


def make_example_job(
    plenoirf_dir,
    run_id=1337,
    site_key="chile",
    particle_key="electron",
    instrument_key="diag9_default_default",
    num_events=128,
):
    job = {}
    job["run_id"] = run_id
    job["plenoirf_dir"] = plenoirf_dir
    job["site_key"] = site_key
    job["particle_key"] = particle_key
    job["instrument_key"] = instrument_key
    job["num_events"] = num_events
    job["debug_probability"] = 1e-2
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
    logger.debug("drawing run's pointing-range")
    run["pointing_range"] = make_pointing_range_for_run(
        config=job["config"], prng=prng
    )

    with json_line_logger.TimeDelta(logger, "draw_primary_and_pointing"):
        job, run = _draw_primaries_and_pointings(
            job=job,
            run=run,
            prng=prng,
            logger=logger,
        )

    tabrec = acr.production.table.init_table_dynamicsizerecarray()

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


def _draw_primaries_and_pointings(job, run, prng, logger):
    _allsky = magnetic_deflection.allsky.AllSky(
        job["paths"]["magnetic_deflection_allsky"]
    )
    job["site"] = copy.deepcopy(_allsky.config["site"])
    job["particle"] = copy.deepcopy(_allsky.config["particle"])

    if not op.exists(job["paths"]["cache"]["primary"]):
        drw = random.draw_primaries_and_pointings(
            prng=prng,
            run_id=job["run_id"],
            site_particle_magnetic_deflection=_allsky,
            pointing_range=run["pointing_range"],
            field_of_view_half_angle_rad=(
                0.5
                * np.deg2rad(
                    job["light_field_camera_config"]["max_FoV_diameter_deg"]
                )
            ),
            num_events=job["num_events"],
        )
        with rnw.open(job["paths"]["cache"]["primary"] + ".json", "wt") as f:
            f.write(json_utils.dumps(drw, indent=4))
        cpw.steering.write_steerings(
            path=job["paths"]["cache"]["primary"],
            runs={job["run_id"]: drw["corsika_primary_steering"]},
        )
        with rnw.open(job["paths"]["cache"]["primary"] + ".prng", "wt") as f:
            f.write(json_utils.dumps(prng.bit_generator.state, indent=4))
    else:
        with rnw.open(job["paths"]["cache"]["primary"] + ".json", "rt") as f:
            drw = json_utils.loads(f.read())
        _rrr = cpw.steering.read_steerings(
            path=job["paths"]["cache"]["primary"],
        )
        drw["corsika_primary_steering"] = _rrr[job["run_id"]]

        with rnw.open(job["paths"]["cache"]["primary"] + ".prng", "rt") as f:
            prng.bit_generator.state = json_utils.loads(f.read())

    run.update(drw)
    return job, run


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
    job["run_id_str"] = bookkeeping.uid.make_run_id_str(run_id=job["run_id"])

    job["paths"] = compile_paths(job=job, tmp_dir=tmp_dir)

    job["light_field_camera_config"] = read_light_field_camera_config(
        plenoirf_dir=job["paths"]["plenoirf_dir"],
        instrument_key=job["instrument_key"],
    )

    job["config"] = configurating.read(
        plenoirf_dir=job["paths"]["plenoirf_dir"]
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
    paths["tmp"]["particle_pools_dat"] = opj(tmp_dir, "particle_pools.dat")
    paths["tmp"]["particle_pools_tar"] = opj(tmp_dir, "particle_pools.tar.gz")

    paths["tmp"]["grid_histogram"] = opj(tmp_dir, "grid.tar")
    paths["tmp"]["grid_roi_histogram"] = opj(tmp_dir, "grid_roi.tar")

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
    return paths


def make_pointing_range_for_run(config, prng):
    """
    Draws the range in solid angle to point in for this particular run.
    We limit this run's range to not be the full sky to avoid queyring the
    entire magnetic_deflection's AllSky solid angle.
    """
    total_range = acr.pointing_range.PointingRange_from_cone(
        azimuth_rad=0.0,
        zenith_rad=0.0,
        half_angel_rad=config["pointing"]["range"]["max_zenith_distance_rad"],
    )
    ptg = acr.pointing_range.draw_pointing(
        pointing_range=total_range, prng=prng
    )
    return acr.pointing_range.PointingRange_from_cone(
        azimuth_rad=ptg["azimuth_rad"],
        zenith_rad=ptg["zenith_rad"],
        half_angel_rad=config["pointing"]["range"]["run_half_angle_rad"],
    )


def _corsika_and_grid(
    job,
    run,
    tabrec,
    prng,
    logger,
):
    with cpw.cherenkov.CherenkovEventTapeWriter(
        path=job["paths"]["tmp"]["cherenkov_pools"]
    ) as evttar, tarfile.open(
        job["paths"]["tmp"]["grid_histogram"], "w"
    ) as imgtar, tarfile.open(
        job["paths"]["tmp"]["grid_roi_histogram"], "w"
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

                print(uid)

                tabrec["primary"].append_record(
                    make_primary_record(
                        uid=uid,
                        corsika_evth=corsika_evth,
                        corsika_primary_steering=run[
                            "corsika_primary_steering"
                        ],
                        primary_directions=run["primary_directions"],
                    )
                )

                tabrec["pointing"].append_record(
                    make_pointing_record(uid=uid, pointings=run["pointings"])
                )

                tabrec["cherenkovsize"].append_record(
                    make_cherenkovsize_record(
                        uid=uid,
                        cherenkov_bunches=cherenkov_bunches,
                    )
                )

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


def read_all_cherenkov_bunches(cherenkov_reader):
    return np.vstack([b for b in cherenkov_reader])
