import magnetic_deflection
import atmospheric_cherenkov_response
import plenoirf
import numpy as np


def test_dummy():
    NUM_EVENTS = 128
    RUN_ID = 1337

    pointing_range = (
        atmospheric_cherenkov_response.pointing_range.PointingRange_from_cone()
    )

    skymap_deflection = magnetic_deflection.skymap.testing.SkyMapDummy()
    assert skymap_deflection.config["particle"]["key"] == "electron"
    assert skymap_deflection.config["site"]["key"] == "chile"

    prng = np.random.Generator(np.random.PCG64(RUN_ID))

    (
        events,
        debug,
    ) = plenoirf.production.draw_primaries_and_pointings.draw_primaries_and_pointings(
        prng=prng,
        run_id=RUN_ID,
        num_events=NUM_EVENTS,
        field_of_view_half_angle_rad=np.deg2rad(6.5),
        pointing_range=pointing_range,
        energy_range={"energy_start_GeV": 1.0, "energy_stop_GeV": 10.0},
        site_particle_magnetic_deflection_skymap=skymap_deflection,
        skymap_query_mode="cone",
    )

    cps = events["corsika_primary_steering"]
    assert len(cps["primaries"]) == NUM_EVENTS

    pdirs = events["primary_directions"]
    assert len(pdirs) == NUM_EVENTS
