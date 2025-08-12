import copy
import datetime
import gzip
import json_utils
import io


def default_filename():
    return "hotfix.log.jsonl.gz"


def loads_loglist_from_run_zipfile(zin, run_id, part):
    path = f"{run_id:06d}/{part:s}/{default_filename():s}"
    loglist = []
    for fileitem in zin.filelist:
        if path == fileitem.filename:
            with zin.open(fileitem, "r") as fin:
                payload_gz = fin.read()
            payload = gzip.decompress(payload_gz)
            loglist = _bytes_to_loglist(b=payload)
        else:
            assert default_filename() not in fileitem.filename, (
                f"Expected file '{path:s}' "
                f"but found '{fileitem.filename:s}' instead."
            )
    return loglist


def dumps_loglist_to_run_zipfile(zout, run_id, part, loglist):
    path = f"{run_id:06d}/{part:s}/{default_filename():s}"
    with zout.open(path, "w") as fout:
        payload = _loglist_to_bytes(loglist=loglist)
        payload_gz = gzip.compress(payload)
        fout.write(payload_gz)


def _bytes_to_loglist(b):
    sbuff = io.StringIO()
    sbuff.write(bytes.decode(b))
    sbuff.seek(0)
    loglist = []
    with json_utils.lines.Reader(sbuff) as jl:
        for item in jl:
            loglist.append(item)
    return loglist


def _loglist_to_bytes(loglist):
    sbuff = io.StringIO()
    with json_utils.lines.Writer(sbuff) as jl:
        for item in loglist:
            jl.write(item)
    sbuff.seek(0)
    return str.encode(sbuff.read())
