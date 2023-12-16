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

    mask = plenoirf.analysis.light_field_trigger_modi.make_mask(
        trigger_table=tt,
        threshold=threshold,
        modus=modus,
    )

    assert mask[0]
    assert not mask[1]
    assert mask[2]
    assert mask[3]
    assert mask[4]
