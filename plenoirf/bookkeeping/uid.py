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


def make_run_id_str(run_id):
    assert 0 <= run_id < RUN_ID_UPPER
    return RUN_ID_FORMAT_STR.format(run_id)


def make_event_id_str(event_id):
    assert 0 <= event_id < EVENT_ID_UPPER
    return EVENT_ID_FORMAT_STR.format(event_id)


def make_uid(run_id, event_id):
    assert 0 <= run_id < RUN_ID_UPPER
    assert 0 <= event_id < EVENT_ID_UPPER
    return RUN_ID_UPPER * run_id + event_id


def split_uid(uid):
    run_id = uid // RUN_ID_UPPER
    event_id = uid % RUN_ID_UPPER
    return run_id, event_id


def make_uid_str(run_id=None, event_id=None, uid=None):
    if uid is None:
        assert run_id is not None and event_id is not None
        uid = make_uid(run_id, event_id)
    else:
        assert run_id is None and event_id is None
    return UID_FOTMAT_STR.format(uid)


def split_uid_str(s):
    uid = int(s)
    return split_uid(uid)


def make_uid_path(run_id, event_id):
    run_id_str = make_run_id_str(run_id=run_id)
    event_id_str = make_event_id_str(event_id=event_id)
    unix_sep = "/"
    return "{:s}{:s}{:s}".format(run_id_str, unix_sep, event_id_str)
