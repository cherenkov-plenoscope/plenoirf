import warnings
import os
import rename_after_writing as rnw
import json_utils


def _default():
    return {"name": None, "start": 1, "stop": 999_999}


def _path(filename=".plenoirf.production-run-id-range.json"):
    return os.path.join(os.path.expanduser("~"), filename)


def _read():
    if not os.path.exists(_path()):
        _write(config=_default())

    with open(_path(), "rt") as f:
        config = json_utils.loads(f.read())
    return config


def _write(config):
    _assert_valid(config)
    with rnw.open(_path(), "wt") as f:
        f.write(json_utils.dumps(config, indent=4))


def _assert_valid(config):
    assert config["start"] > 0
    assert config["stop"] >= config["start"]


def read_from_configfile():
    config = _read()
    assert config["name"] is not None, f"Please set 'name' in '{_path():s}'."
    _assert_valid(config)
    return config
