import numpy as np
import os
import rename_after_writing


def write_config(work_dir, zenith_bin_edges, energy_bin_edges):
    config_dir = os.path.join(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    write_numpy(
        path=os.path.join(config_dir, "zenith_bin_edges.numpy"),
        x=zenith_bin_edges,
    )
    write_numpy(
        path=os.path.join(config_dir, "energy_bin_edges.numpy"),
        x=energy_bin_edges,
    )


def read_config_if_None(work_dir, config=None):
    if config is None:
        config_dir = os.path.join(work_dir, "config")
        out = {}
        out["energy_bin"] = read_binning(
            os.path.join(config_dir, "energy_bin_edges.numpy")
        )
        out["zenith_bin"] = read_binning(
            os.path.join(config_dir, "zenith_bin_edges.numpy")
        )
        return out
    else:
        return config


def write_numpy(path, x):
    with rename_after_writing.open(path, "wb") as f:
        np.save(f, x)


def read_numpy(path):
    with open(path, "rb") as f:
        x = np.load(f)
    return x


def read_binning(path):
    out = {}
    out["edges"] = read_numpy(path)
    out["num"] = len(out["edges"]) - 1
    return out
