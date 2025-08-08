import copy
import datetime


def default_hotfix_logfilename():
    return "hotfix.txt.gz"


def append_to_hotfix_logfile(logfile_text, msg):
    out = copy.deepcopy(logfile_text)
    line = f"{datetime.datetime.now().isoformat():s}, {msg:s}\n"
    if "\n" in out and not out.endswith("\n"):
        out += "\n"
    out += line
    return out
