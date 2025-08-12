import copy
import datetime
import gzip
import json_utils


def default_filename():
    return "hotfix.json.gz"


def loads_loglist_from_run_zipfile(zin):
    loglist = []
    for fileitem in zin.filelist:
        if default_filename() in fileitem.filename:
            with zin.open(fileitem, "r") as fin:
                loglist = json_utils.loads(gzip.decompress(fin.read()))
    return loglist


def dumps_loglist_to_run_zipfile(zout, loglist):
    with zout.open(default_filename(), "w") as fout:
        fout.write(gzip.compress(json_utils.dumps(loglist).encode()))
