import numpy as np


def draw_event_ids_for_debug_output(
    num_events_in_run,
    min_num_events,
    fraction_of_events,
    prng,
):
    assert num_events_in_run >= 1
    assert 0 <= min_num_events <= num_events_in_run
    assert 0.0 <= fraction_of_events <= 1.0

    num_frac = int(np.round(num_events_in_run * fraction_of_events))
    num = max([num_frac, min_num_events])

    EVENT_IDS_START_AT_ONE = 1

    if num == 0:
        out = []
    else:
        out = EVENT_IDS_START_AT_ONE + prng.choice(
            num_events_in_run,
            replace=False,
            size=num,
        )
    out = sorted(out)
    return np.array(out, dtype=np.int64)
