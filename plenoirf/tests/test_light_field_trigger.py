import plenoirf
import numpy as np


def dummy_trigger_table_init(num_foci=3):
    tt = {}
    for i in range(num_foci):
        tt["focus_{:02d}_response_pe".format(i)] = []
    return tt


def dummy_trigger_table_append(trigger_table, response):
    for i, r in enumerate(response):
        trigger_table["focus_{:02d}_response_pe".format(i)].append(r)
    return trigger_table


def dummy_trigger_table_arrayfy(trigger_table):
    out = {}
    for key in trigger_table:
        out[key] = np.array(trigger_table[key])
    return out


def test_trigger_modus():
    tt = dummy_trigger_table_init(num_foci=3)
    tt = dummy_trigger_table_append(tt, [120, 100, 100])
    tt = dummy_trigger_table_append(tt, [120, 130, 140])
    tt = dummy_trigger_table_append(tt, [120, 100, 100])
    tt = dummy_trigger_table_append(tt, [900, 950, 999])
    tt = dummy_trigger_table_append(tt, [900, 850, 800])
    tt = dummy_trigger_table_arrayfy(tt)

    threshold = 101
    modus = {
        "accepting_focus": 0,
        "rejecting_focus": 2,
        "accepting": {
            "threshold_accepting_over_rejecting": [1, 1, 0.5],
            "response_pe": [1e1, 1e2, 1e3],
        },
    }

    mask = plenoirf.light_field_trigger.make_mask(
        trigger_table=tt,
        threshold=threshold,
        modus=modus,
    )

    assert mask[0]
    assert not mask[1]
    assert mask[2]
    assert mask[3]
    assert mask[4]


def test_height_to_depth():

    HEIGHT_M = 100.0
    for zenith_rad in np.deg2rad(np.linspace(0, 45, 100)):
        depth_m = plenoirf.light_field_trigger.height_to_depth(
            height_m=HEIGHT_M, zenith_rad=zenith_rad
        )
        height_m = plenoirf.light_field_trigger.depth_to_height(
            depth_m=depth_m, zenith_rad=zenith_rad
        )
        assert depth_m >= HEIGHT_M
        np.testing.assert_almost_equal(HEIGHT_M, height_m)
        assert depth_m <= np.sqrt(2) * HEIGHT_M


def test_bin_assignment():
    # bins          0     1    2
    bin_edges = [100, 200, 300, 400]
    cases = [
        (99, -1),
        (100, 0),
        (199, 0),
        (200, 1),
        (299, 1),
        (300, 2),
        (399, 2),
        (400, 3),
        (499, 3),
    ]

    for x, b in cases:
        c = plenoirf.light_field_trigger.assign_to_bin(
            x=x, bin_edges=bin_edges
        )
        assert c == b


def test_assign_focus():

    # bins            0     1    2
    bin_edges_m = [100, 200, 300, 400]

    cases = [
        {"a_m": 99, "af": -1, "r_m": 100, "rf": 0},
        {"a_m": 100, "af": 0, "r_m": 199, "rf": 0},
        {"a_m": 199, "af": 0, "r_m": 200, "rf": 1},
        {"a_m": 200, "af": 1, "r_m": 299, "rf": 1},
        {"a_m": 299, "af": 1, "r_m": 300, "rf": 2},
        {"a_m": 300, "af": 2, "r_m": 399, "rf": 2},
        {"a_m": 399, "af": 2, "r_m": 400, "rf": 3},
        {"a_m": 400, "af": 3, "r_m": 1e3, "rf": 3},
    ]

    for case in cases:
        af, rf = (
            plenoirf.light_field_trigger.assign_accepting_and_rejecting_focus_based_on_pointing_zenith(
                pointing_zenith_rad=0.0,
                accepting_height_above_observation_level_m=case["a_m"],
                rejecting_height_above_observation_level_m=case["r_m"],
                trigger_foci_bin_edges_m=bin_edges_m,
            )
        )

        assert af == case["af"]
        assert rf == case["rf"]
