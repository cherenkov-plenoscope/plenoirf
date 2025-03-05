import magnetic_deflection
import atmospheric_cherenkov_response
import plenoirf
import solid_angle_utils
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

    energy_distribution = {
        "energy_start_GeV": 1.0,
        "energy_stop_GeV": 10.0,
        "power_slope": -1.5,
    }

    scatter_solid_angle_vs_energy = {
        "energy_GeV": [1.0, 10.0],
        "solid_angle_sr": [
            solid_angle_utils.squaredeg2sr(4.0),
            solid_angle_utils.squaredeg2sr(16.0),
        ],
    }

    (
        events,
        debug,
    ) = plenoirf.production.draw_primaries_and_pointings.draw_primaries_and_pointings(
        prng=prng,
        run_id=RUN_ID,
        num_events=NUM_EVENTS,
        field_of_view_half_angle_rad=np.deg2rad(6.5),
        pointing_range=pointing_range,
        energy_distribution=energy_distribution,
        site_particle_magnetic_deflection_skymap=skymap_deflection,
        scatter_solid_angle_vs_energy=scatter_solid_angle_vs_energy,
    )

    cps = events["corsika_primary_steering"]
    assert len(cps["primaries"]) == NUM_EVENTS

    pdirs = events["primary_directions"]
    assert len(pdirs) == NUM_EVENTS
