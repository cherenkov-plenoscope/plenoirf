import json_utils
import numpy as np
import dynamicsizerecarray
import copy
import os
import corsika_primary
import rename_after_writing as rnw
from .. import configuration


def write(path, job):
    out = copy.deepcopy(job)
    os.makedirs(path, exist_ok=True)

    if "prng" in out:
        with rnw.open(os.path.join(path, "prng.json"), "wt") as f:
            f.write(json_utils.dumps(todict_prng(prng=out.pop("prng"))))

    if "event_table" in out:
        event_table = out.pop("event_table")
        for key in event_table:
            event_table[key] = todict_recarray(event_table[key])
        with rnw.open(os.path.join(path, "event_table.json"), "wt") as f:
            f.write(json_utils.dumps(event_table))

    if "corsika_primary_steering" in out["run"]:
        corsika_primary_steering = out["run"].pop("corsika_primary_steering")
        corsika_primary.steering.write_steerings(
            path=os.path.join(path, "corsika_primary_steering.tar"),
            runs={out["run_id"]: corsika_primary_steering},
        )

    if "config" in out:
        _ = out["config"]

    with rnw.open(os.path.join(path, "job.json"), "wt") as f:
        f.write(json_utils.dumps(out, indent=4))


def read(path):
    out = {}
    with open(os.path.join(path, "job.json"), "rt") as f:
        out = json_utils.loads(f.read())

    if os.path.exists(os.path.join(path, "prng.json")):
        with open(os.path.join(path, "prng.json"), "rt") as f:
            out["prng"] = fromdict_prng(json_utils.loads(f.read()))

    if os.path.exists(os.path.join(path, "event_table.json")):
        out["event_table"] = {}
        with open(os.path.join(path, "event_table.json"), "rt") as f:
            event_table = json_utils.loads(f.read())
        for key in event_table:
            out["event_table"][key] = fromdict_recarray(event_table[key])

    if "run" not in out:
        out["run"] = {}

    if os.path.exists(os.path.join(path, "corsika_primary_steering.tar")):
        corsika_primary_steerings = corsika_primary.steering.read_steerings(
            path=os.path.join(path, "corsika_primary_steering.tar"),
        )
        corsika_primary_steering = corsika_primary_steerings[out["run_id"]]
        out["run"]["corsika_primary_steering"] = corsika_primary_steering

    out["config"] = configuration.read(plenoirf_dir=out["plenoirf_dir"])
    return out


def todict_prng(prng):
    return prng.bit_generator.state


def fromdict_prng(s):
    prng = np.random.Generator(np.random.PCG64(seed=0))
    prng.bit_generator.state = s
    return prng


def todict_recarray(a):
    out = {}
    out["__class__"] = a.__class__.__module__ + "." + a.__class__.__name__

    if isinstance(a, dynamicsizerecarray.DynamicSizeRecarray):
        b = a.to_recarray()
    else:
        b = a

    out["__dtype__"] = {}
    out["__fields__"] = {}
    for name in b.dtype.names:
        out["__dtype__"][name] = b.dtype[name].str
        out["__fields__"][name] = np.array(b[name])
    return out


def fromdict_recarray(obj):
    dtype = []
    for name in obj["__dtype__"]:
        tup = (name, obj["__dtype__"][name])
        dtype.append(tup)

    shape = []
    for name in obj["__fields__"]:
        shape.append(len(obj["__fields__"][name]))
    shape = np.array(shape)
    assert np.all(shape == shape[0])
    shape = shape[0]

    a = np.core.records.recarray(shape=shape, dtype=dtype)
    for name in obj["__dtype__"]:
        a[name] = obj["__fields__"][name]

    if obj["__class__"] == "numpy.recarray":
        return a
    elif obj["__class__"] == "dynamicsizerecarray.DynamicSizeRecarray":
        return dynamicsizerecarray.DynamicSizeRecarray(recarray=a)
    else:
        raise ValueError(
            "__class__ must be either "
            "'numpy.recarray' or "
            "'dynamicsizerecarray.DynamicSizeRecarray'."
        )
