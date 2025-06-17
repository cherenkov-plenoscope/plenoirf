import numpy as np
import binning_utils
import solid_angle_utils
import spherical_coordinates
import corsika_primary
import magnetic_deflection
import atmospheric_cherenkov_response as acr
import os
from os.path import join as opj
import json_utils
import rename_after_writing as rnw
import zipfile
import gzip
import pickle
import json_line_logger

from .. import bookkeeping
from .. import configuration
from .. import utils


def draw_primaries_and_pointings(
    prng,
    run_id,
    site_particle_magnetic_deflection_skymap,
    pointing_range,
    energy_distribution,
    scatter_solid_angle_vs_energy,
    field_of_view_half_angle_rad,
    num_events,
    event_uids_for_debugging=[],
    logger=None,
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
    site_particle_magnetic_deflection_skymap : magnetic_deflection.skymap.SkyMap
        Describes from what direction the given particle must be thrown in
        order to see its Cherenkov-light. Must match 'particle' and 'site'.
    pointing_range : dict
        Instrument's range to draw pointings from.
    energy_distribution : dict
        How to populate the energetic spectrum of the particle type.
    scatter_solid_angle_vs_energy : dict
        How large is the solid angle to scatter in for a given energy.
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
    mag_skymap = site_particle_magnetic_deflection_skymap
    logger = json_line_logger.LoggerStdout_if_logger_is_None(logger=logger)

    # assertion checks
    # ----------------
    assert run_id > 0
    assert num_events > 0
    assert field_of_view_half_angle_rad > 0.0

    i8 = np.int64
    f8 = np.float64

    site = mag_skymap.config["site"]
    acr.sites.assert_valid(site)
    particle = mag_skymap.config["particle"]
    acr.particles.assert_valid(particle)

    # primary directions
    # ------------------
    run = {
        "run_id": i8(run_id),
        "event_id_of_first_event": i8(1),
        "observation_level_asl_m": f8(site["observation_level_asl_m"]),
        "earth_magnetic_field_x_muT": f8(site["earth_magnetic_field_x_muT"]),
        "earth_magnetic_field_z_muT": f8(site["earth_magnetic_field_z_muT"]),
        "atmosphere_id": i8(site["corsika_atmosphere_id"]),
        "energy_range": {
            "start_GeV": f8(energy_distribution["energy_start_GeV"]),
            "stop_GeV": f8(energy_distribution["energy_stop_GeV"]),
        },
        "random_seed": corsika_primary.random.seed.make_simple_seed(run_id),
    }

    # loop
    energies_GeV = {}
    instrument_pointings = {}
    primary_directions = {}
    debug = {}
    primaries = []

    for event_id in np.arange(1, num_events + 1):
        event_uid_str = bookkeeping.uid.make_uid_str(
            run_id=run_id, event_id=event_id
        )

        if utils.is_10th_part_in_current_decade(i=event_id):
            logger.info(__name__ + ": uid={:s}".format(event_uid_str))

        # energies
        # --------
        energies_GeV[event_uid_str] = (
            corsika_primary.random.distributions.draw_power_law(
                prng=prng,
                lower_limit=energy_distribution["energy_start_GeV"],
                upper_limit=energy_distribution["energy_stop_GeV"],
                power_slope=energy_distribution["power_slope"],
                num_samples=1,
            )[0]
        )

        # instrument pointings
        # --------------------
        instrument_pointings[event_uid_str] = acr.pointing_range.draw_pointing(
            pointing_range=pointing_range,
            prng=prng,
        )

        scatter_solid_angle_sr = acr.particles.interpolate_scatter_solid_angle(
            energy_GeV=energies_GeV[event_uid_str],
            scatter_energy_GeV=scatter_solid_angle_vs_energy["energy_GeV"],
            scatter_solid_angle_sr=scatter_solid_angle_vs_energy[
                "solid_angle_sr"
            ],
        )

        if energies_GeV[event_uid_str] <= mag_skymap.binning["energy"]["stop"]:
            res, dbg = mag_skymap.draw(
                azimuth_rad=instrument_pointings[event_uid_str]["azimuth_rad"],
                zenith_rad=instrument_pointings[event_uid_str]["zenith_rad"],
                half_angle_rad=field_of_view_half_angle_rad,
                energy_start_GeV=energies_GeV[event_uid_str] * 0.99,
                energy_stop_GeV=energies_GeV[event_uid_str] * 1.01,
                threshold_cherenkov_density_per_sr=1e3,
                solid_angle_sr=scatter_solid_angle_sr,
                prng=prng,
            )
        else:
            scatter_cone_half_angle_rad = solid_angle_utils.cone.half_angle(
                solid_angle_sr=scatter_solid_angle_sr
            )

            (
                _az,
                _zd,
            ) = corsika_primary.random.distributions.draw_azimuth_zenith_in_viewcone(
                prng=prng,
                azimuth_rad=instrument_pointings[event_uid_str]["azimuth_rad"],
                zenith_rad=instrument_pointings[event_uid_str]["zenith_rad"],
                min_scatter_opening_angle_rad=0.0,
                max_scatter_opening_angle_rad=scatter_cone_half_angle_rad,
                max_zenith_rad=corsika_primary.MAX_ZENITH_DISTANCE_RAD,
                max_iterations=1000 * 1000,
            )
            res = {
                "cutoff": False,
                "particle_azimuth_rad": _az,
                "particle_zenith_rad": _zd,
                "solid_angle_thrown_sr": solid_angle_utils.cone.intersection_of_two_cones(
                    half_angle_one_rad=corsika_primary.MAX_ZENITH_DISTANCE_RAD,
                    half_angle_two_rad=scatter_cone_half_angle_rad,
                    angle_between_cones=instrument_pointings[event_uid_str][
                        "zenith_rad"
                    ],
                ),
            }
            dbg = {"method": "viewcone", "sky_draw_quantile": float("nan")}

        primary_directions[event_uid_str] = res
        for dbgkey in ["method", "sky_draw_quantile"]:
            primary_directions[event_uid_str][dbgkey] = dbg[dbgkey]

        if int(event_uid_str) in event_uids_for_debugging:
            debug[event_uid_str] = {"result": res, "debug": dbg}

        prm = {}
        prm["particle_id"] = f8(particle["corsika"]["particle_id"])
        prm["energy_GeV"] = f8(energies_GeV[event_uid_str])
        prm["phi_rad"] = f8(
            spherical_coordinates.corsika.az_to_phi(
                primary_directions[event_uid_str]["particle_azimuth_rad"]
            )
        )
        prm["theta_rad"] = f8(
            spherical_coordinates.corsika.zd_to_theta(
                primary_directions[event_uid_str]["particle_zenith_rad"]
            )
        )
        prm["depth_g_per_cm2"] = f8(0.0)
        primaries.append(prm)

    out = {}
    out["corsika_primary_steering"] = {"run": run, "primaries": primaries}
    out["instrument_pointings"] = instrument_pointings
    out["primary_directions"] = {}

    for event_uid_str in primary_directions:
        y = {}
        for key in [
            "cutoff",
            "solid_angle_thrown_sr",
            "method",
            "sky_draw_quantile",
        ]:
            y[key] = primary_directions[event_uid_str][key]
        out["primary_directions"][event_uid_str] = y
    return out, debug


def run(env, part, seed):
    name = __name__.split(".")[-1]
    module_work_dir = opj(env["work_dir"], part, name)

    if os.path.exists(module_work_dir):
        return

    os.makedirs(module_work_dir)
    logger = json_line_logger.LoggerFile(opj(module_work_dir, "log.jsonl"))
    logger.info(__name__)
    logger.info(f"seed: {seed:d}")

    prng = np.random.Generator(np.random.PCG64(seed))

    logger.info("open SkyMap")
    skymap = magnetic_deflection.skymap.SkyMap(
        work_dir=opj(
            env["plenoirf_dir"],
            "magnetic_deflection",
            env["site_key"],
            env["particle_key"],
        )
    )

    event_uids_for_debugging = json_utils.read(
        path=opj(
            env["work_dir"],
            "prm2cer",
            "draw_event_uids_for_debugging",
            "event_uids_for_debugging.json",
        )
    )

    pointing_range = acr.pointing_range.PointingRange_from_cone(
        azimuth_rad=0.0,
        zenith_rad=0.0,
        half_angel_rad=env["config"]["pointing"]["range"][
            "max_zenith_distance_rad"
        ],
    )

    logger.info("drawing rimaries_and_pointings")
    out, debug = draw_primaries_and_pointings(
        prng=prng,
        run_id=env["run_id"],
        site_particle_magnetic_deflection_skymap=skymap,
        pointing_range=pointing_range,
        energy_distribution=compile_energy_distribution(env=env),
        scatter_solid_angle_vs_energy=env["config"][
            "particles_scatter_solid_angle"
        ][env["particle_key"]],
        field_of_view_half_angle_rad=env["instrument"][
            "field_of_view_half_angle_rad"
        ],
        num_events=env["num_events"],
        event_uids_for_debugging=event_uids_for_debugging,
        logger=logger,
    )

    logger.info("exporting results.")
    with rnw.open(opj(module_work_dir, "result.pkl"), "wb") as fout:
        fout.write(pickle.dumps(out))

    logger.info("exporting debug.")
    write_debug(path=opj(module_work_dir, "debug.zip"), debug=debug)

    logger.info("done.")
    json_line_logger.shutdown(logger=logger)

    # tidy up and compress
    utils.gzip_file(opj(module_work_dir, "log.jsonl"))
    utils.gzip_file(opj(module_work_dir, "result.pkl"))
    # opj(module_work_dir, "debug.zip") is already compressed internally


def write_debug(path, debug):
    event_uid_strs_for_debugging = sorted(list(debug.keys()))
    with rnw.open(path, "wb") as fff:
        with zipfile.ZipFile(fff, "w") as zout:
            for event_uid_str in event_uid_strs_for_debugging:
                dbg_text = json_utils.dumps(debug[event_uid_str], indent=None)
                dbg_bytes = dbg_text.encode()
                dbg_gz_bytes = gzip.compress(dbg_bytes)
                name = (
                    event_uid_str + "_magnetic_deflection_skymap_query.json.gz"
                )
                with zout.open(name, "w") as fout:
                    fout.write(dbg_gz_bytes)


def compile_energy_distribution(env):
    ene = env["config"]["particles_simulated_energy_distribution"][
        env["particle_key"]
    ]
    out = {}

    out["energy_start_GeV"] = binning_utils.power10.lower_bin_edge(
        **ene["energy_start_GeV_power10"]
    )
    out["energy_stop_GeV"] = binning_utils.power10.lower_bin_edge(
        **ene["energy_stop_GeV_power10"]
    )
    assert ene["spectrum"]["type"] == "power_law"
    out["power_slope"] = ene["spectrum"]["power_slope"]

    configuration.asseret_energy_start_GeV_is_valid(
        particle=acr.particles.init(env["particle_key"]),
        energy_start_GeV=out["energy_start_GeV"],
    )
    return out
