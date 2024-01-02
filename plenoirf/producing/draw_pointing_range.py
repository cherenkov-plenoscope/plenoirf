import atmospheric_cherenkov_response as acr


def make_pointing_range_for_run(pointing_range, prng):
    """
    Draws the range in solid angle to point in for this particular run.
    We limit this run's range to not be the full sky to avoid queyring the
    entire magnetic_deflection's AllSky solid angle.
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


def run_job(job, run, prng, logger):
    logger.debug("drawing run's pointing-range")
    run["pointing_range"] = make_pointing_range_for_run(
        max_zenith_distance_rad=config["pointing"]["range"]["max_zenith_distance_rad"],
        run_half_angle_rad=config["pointing"]["range"]["run_half_angle_rad"],
        prng=prng
    )
    return job, run
