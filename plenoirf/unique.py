"""
The (U)nique (ID)entity of a cosmic particle entering earth's atmosphere.
Airshowers have UIDs. Airshowers may lead to a detection by the
instrument which in turn will create a record.
So record-IDs are an instrument-specific measure,
while UIDs are a simulation specific measure to keep track of all
thrown particles when estimating the instrument's response-function.

The UID is related to CORSIKAs scheme of production RUNs and EVENTs within
the runs.
"""
RUN_ID_NUM_DIGITS = 6
EVENT_ID_NUM_DIGITS = 6
UID_NUM_DIGITS = RUN_ID_NUM_DIGITS + EVENT_ID_NUM_DIGITS

RUN_ID_UPPER = 10**RUN_ID_NUM_DIGITS
EVENT_ID_UPPER = 10**EVENT_ID_NUM_DIGITS

UID_FOTMAT_STR = "{:0" + str(UID_NUM_DIGITS) + "d}"

RUN_ID_FORMAT_STR = "{:0" + str(RUN_ID_NUM_DIGITS) + "d}"
EVENT_ID_FORMAT_STR = "{:0" + str(EVENT_ID_NUM_DIGITS) + "d}"


def make_uid(run_id, event_id):
    assert 0 <= run_id < RUN_ID_UPPER
    assert 0 <= event_id < EVENT_ID_UPPER
    return RUN_ID_UPPER * run_id + event_id


def split_uid(uid):
    run_id = uid // RUN_ID_UPPER
    event_id = uid % RUN_ID_UPPER
    return run_id, event_id


def make_uid_str(run_id, event_id):
    uid = make_uid(run_id, event_id)
    return UID_FOTMAT_STR.format(uid)


def split_uid_str(s):
    uid = int(s)
    return split_uid(uid)
