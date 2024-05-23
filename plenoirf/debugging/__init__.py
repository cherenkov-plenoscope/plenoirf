import numpy as np
import tempfile
import pickle
import subprocess


def draw_event_ids_for_debugging(
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


def estimate_memory_size_in_bytes_of_anything(anything):
    suffix = "-{:s}.estimate_memory_size_in_bytes_of_anything".format(__name__)
    size = 0
    with tempfile.TemporaryFile(mode="wb", suffix=suffix) as ftmp:
        start = ftmp.tell()
        pickle.dump(anything, ftmp)
        stop = ftmp.tell()
        size = stop - start
    return size


def estimate_disk_usage_in_bytes(path):
    out = {}
    try:
        proc = subprocess.Popen(
            ["du", "--bytes", path], stdout=subprocess.PIPE
        )
        proc.wait()
        raw_stdout = proc.communicate()[0]
        ascii_stdout = raw_stdout.decode()
        for line in str.splitlines(ascii_stdout):
            size_str, path_str = line.split("\t")
            out[path_str] = int(size_str)
    except:
        out[path] = -1
    return out
