import numpy as np
import corsika_primary
import magnetic_deflection
import atmospheric_cherenkov_response as acr
import os
from os import path as op
from os.path import join as opj
import json_utils
import rename_after_writing as rnw
import tarfile
import gzip

from .. import bookkeeping
from .. import tar_append
from . import job_io


def draw_primaries_and_pointings(
    prng,
    run_id,
    site_particle_magnetic_deflection,
    pointing_range,
    field_of_view_half_angle_rad,
    num_events,
    event_uids_for_debugging=[],
):
    """
    Draw the random distribution of particles to induce showers and emitt
    Cherenkov-light which is relevant for our instrument.

    Parameters
    ----------
    prng : numpy.random.Generator
        Pseudo random number generator.
    run_id : int
        The run-number/run-id of the corsika-run. Must be > 0.
    site_particle_magnetic_deflection : magnetic_deflection.allsky.Random()
        Describes from what direction the given particle must be thrown in
        order to see its Cherenkov-light. Must match 'particle' and 'site'.
    pointing_range : dict
        Instrument's range to draw pointings from.
    field_of_view_half_angle_rad : float
        Instrument's field-of-view
    num_events : int
        The number of events in the run.
    event_uids_for_debugging : list, array, or set of ints
        Event uids of which full debug output will be returned.

    Returns
    -------
    steering_dict : dict
        To be given to CORSIKA-primary.
        Describes explicitly how each particle shall be thrown in CORSIKA.
    """

    # assertion checks
    # ----------------
    assert run_id > 0
    assert num_events > 0
    assert field_of_view_half_angle_rad > 0.0

    i8 = np.int64
    f8 = np.float64

    site = site_particle_magnetic_deflection.config["site"]
    acr.sites.assert_valid(site)
    particle = site_particle_magnetic_deflection.config["particle"]
    acr.particles.assert_valid(particle)

    # energies
    # --------
    start_energy_GeV = acr.particles.compile_energy(
        particle["population"]["energy"]["start_GeV"]
    )
    stop_energy_GeV = acr.particles.compile_energy(
        particle["population"]["energy"]["stop_GeV"]
    )

    # primary directions
    # ------------------
    rnd = magnetic_deflection.allsky.random.Random(
        allsky_deflection=site_particle_magnetic_deflection
    )
    _shower_spread_half_angle_rad = np.deg2rad(
        particle["population"]["direction"]["scatter_cone_half_angle_deg"]
    )

    run = {
        "run_id": i8(run_id),
        "event_id_of_first_event": i8(1),
        "observation_level_asl_m": f8(site["observation_level_asl_m"]),
        "earth_magnetic_field_x_muT": f8(site["earth_magnetic_field_x_muT"]),
        "earth_magnetic_field_z_muT": f8(site["earth_magnetic_field_z_muT"]),
        "atmosphere_id": i8(site["corsika_atmosphere_id"]),
        "energy_range": {
            "start_GeV": f8(start_energy_GeV),
            "stop_GeV": f8(stop_energy_GeV),
        },
        "random_seed": corsika_primary.random.seed.make_simple_seed(run_id),
    }

    # loop
    energies_GeV = {}
    pointings = {}
    primary_directions = {}
    debug = {}
    primaries = []

    for event_id in np.arange(1, num_events + 1):
        event_uid_str = bookkeeping.uid.make_uid_str(
            run_id=run_id, event_id=event_id
        )

        # energies
        # --------
        energies_GeV[
            event_uid_str
        ] = corsika_primary.random.distributions.draw_power_law(
            prng=prng,
            lower_limit=start_energy_GeV,
            upper_limit=stop_energy_GeV,
            power_slope=particle["population"]["energy"]["power_law_slope"],
            num_samples=1,
        )[
            0
        ]

        # instrument pointings
        # --------------------
        pointings[event_uid_str] = acr.pointing_range.draw_pointing(
            pointing_range=pointing_range,
            prng=prng,
        )

        res, dbg = rnd.draw_particle_direction(
            prng=prng,
            method="grid",
            azimuth_rad=pointings[event_uid_str]["azimuth_rad"],
            zenith_rad=pointings[event_uid_str]["zenith_rad"],
            half_angle_rad=field_of_view_half_angle_rad,
            energy_GeV=energies_GeV[event_uid_str],
            shower_spread_half_angle_rad=_shower_spread_half_angle_rad,
            min_num_cherenkov_photons=1e3,
        )
        primary_directions[event_uid_str] = res

        if int(event_uid_str) in event_uids_for_debugging:
            debug[event_uid_str] = {"result": res, "debug": dbg}

        prm = {}
        prm["particle_id"] = f8(particle["corsika_particle_id"])
        prm["energy_GeV"] = f8(energies_GeV[event_uid_str])
        prm["azimuth_rad"] = f8(
            primary_directions[event_uid_str]["particle_azimuth_rad"]
        )
        prm["zenith_rad"] = f8(
            primary_directions[event_uid_str]["particle_zenith_rad"]
        )
        prm["depth_g_per_cm2"] = f8(0.0)
        primaries.append(prm)

    out = {}
    out["corsika_primary_steering"] = {"run": run, "primaries": primaries}
    out["pointings"] = pointings
    out["primary_directions"] = {}

    for event_uid_str in primary_directions:
        y = {}
        for key in ["cutoff", "solid_angle_thrown_sr"]:
            y[key] = primary_directions[event_uid_str][key]
        out["primary_directions"][event_uid_str] = y
    return out, debug


def run_job(job, logger):
    cache_path = os.path.join(
        job["paths"]["tmp_dir"],
        "draw_primaries_and_pointings",
        "__job_cache__",
    )

    if os.path.exists(cache_path) and job["cache"]:
        logger.info("draw_primaries_and_pointings, read cache")
        return job_io.read(path=cache_path)
    else:
        logger.info("draw_primaries_and_pointings, open AllSky")
        allsky = magnetic_deflection.allsky.AllSky(
            job["paths"]["magnetic_deflection_allsky"]
        )

        logger.info("draw_primaries_and_pointings, draw primaries")

        drw, debug = draw_primaries_and_pointings(
            prng=job["prng"],
            run_id=job["run_id"],
            site_particle_magnetic_deflection=allsky,
            pointing_range=job["run"]["pointing_range"],
            field_of_view_half_angle_rad=job["instrument"][
                "field_of_view_half_angle_rad"
            ],
            num_events=job["num_events"],
            event_uids_for_debugging=job["run"]["event_uids_for_debugging"],
        )

        job["run"].update(drw)

        logger.info("draw_primaries_and_pointings, export debug info")

        write_draw_primaries_and_pointings_debug(
            path=job["paths"]["debug"]["draw_primary_and_pointing"],
            debug=debug,
        )

        if job["cache"]:
            logger.info("draw_primaries_and_pointings, write cache")
            job_io.write(path=cache_path, job=job)

    return job


def write_draw_primaries_and_pointings_debug(
    path,
    debug,
):
    event_uid_strs_for_debugging = sorted(list(debug.keys()))
    with tarfile.open(path, "w") as tarout:
        for event_uid_str in event_uid_strs_for_debugging:
            dbg_text = json_utils.dumps(debug[event_uid_str], indent=None)
            dbg_bytes = dbg_text.encode()
            dbg_gz_bytes = gzip.compress(dbg_bytes)
            tar_append.tar_append(
                tarout=tarout,
                filename=event_uid_str
                + "_magnetic_deflection_allsky_query.json.gz",
                filebytes=dbg_gz_bytes,
            )
