import atmospheric_cherenkov_response as acr
import pickle
import os
import rename_after_writing as rnw

from .. import seeding


def draw_pointing_range(max_zenith_distance_rad, run_half_angle_rad, prng):
    """
    Draws the range in solid angle from a an individual run can draw
    pointings from.
    We limit the run's pointing range to not be the full sky to avoid
    queyring the entire magnetic_deflection's AllSky solid angle.

    Parameters
    ----------
    max_zenith_distance_rad : float
        The maximum distance to zenith to place a run's pointing-range-cone in.
    run_half_angle_rad : float
        The half angle of a run's pointing-range-cone.

    Returns
    -------
    pointing_range : dict
        See atmospheric_cherenkov_response.pointing_range.
    """
    total_range = acr.pointing_range.PointingRange_from_cone(
        azimuth_rad=0.0,
        zenith_rad=0.0,
        half_angel_rad=max_zenith_distance_rad,
    )
    ptg = acr.pointing_range.draw_pointing(
        pointing_range=total_range, prng=prng
    )
    return acr.pointing_range.PointingRange_from_cone(
        azimuth_rad=ptg["azimuth_rad"],
        zenith_rad=ptg["zenith_rad"],
        half_angel_rad=run_half_angle_rad,
    )


def run(env, logger):
    opj = os.path.join

    prng = seeding.init_numpy_random_Generator_PCG64_from_path_and_name(
        path=opj(env["work_dir"], "named_random_seeds.json"),
        name="draw_pointing_range",
    )

    pointing_range = draw_pointing_range(
        max_zenith_distance_rad=env["config"]["pointing"]["range"][
            "max_zenith_distance_rad"
        ],
        run_half_angle_rad=env["config"]["pointing"]["range"][
            "run_half_angle_rad"
        ],
        prng=prng,
    )
    logger.info("draw_pointing_range")

    with rnw.open(opj(env["work_dir"], "pointing_range.pkl"), "wb") as fout:
        fout.write(pickle.dumps(pointing_range))
