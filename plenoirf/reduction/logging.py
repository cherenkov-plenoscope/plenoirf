import json_line_logger as _jll
from json_line_logger import TimeDelta


def stdout_logger_if_logger_is_None(logger):
    if logger is None:
        return _jll.LoggerStdout(fmt=_jll.SMP)
    else:
        return logger
