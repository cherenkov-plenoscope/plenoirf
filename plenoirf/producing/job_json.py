import json_utils
import numpy as np
import dynamicsizerecarray
import copy


def dumps(job):
    out = copy.deepcopy(job)
    out["prng"] = todict_prng(prng=out["prng"])

    for key in out["event_table"]:
        out["event_table"][key] = todict_recarray(out["event_table"][key])

    return json_utils.dumps(out)


def loads(s):
    job = json_utils.loads(s)
    job["prng"] = fromdict_prng(prng=job["prng"])

    for key in job["event_table"]:
        job["event_table"][key] = fromdict_recarray(job["event_table"][key])

    return job


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
