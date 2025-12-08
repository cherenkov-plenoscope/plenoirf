import numpy as np
import os
import rename_after_writing


def write_config(work_dir, zenith_bin_edges, energy_bin_edges):
    config_dir = os.path.join(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    with rename_after_writing.open(
        os.path.join(config_dir, "zenith_bin_edges.numpy"), "wb"
    ) as f:
        np.save(f, zenith_bin_edges)

    with rename_after_writing.open(
        os.path.join(config_dir, "energy_bin_edges.numpy"), "wb"
    ) as f:
        np.save(f, energy_bin_edges)


def read_config_if_None(work_dir, config):
    if config is None:
        config_dir = os.path.join(work_dir, "config")
        out = {}
        out["energy_bin"] = {}
        out["zenith_bin"] = {}
        with open(
            os.path.join(config_dir, "energy_bin_edges.numpy"), "rb"
        ) as f:
            out["energy_bin"]["edges"] = np.load(f)
        with open(
            os.path.join(config_dir, "zenith_bin_edges.numpy"), "rb"
        ) as f:
            out["zenith_bin"]["edges"] = np.load(f)

        for key in out:
            out[key]["num"] = len(out[key]["edges"]) - 1
        return out
    else:
        return config
