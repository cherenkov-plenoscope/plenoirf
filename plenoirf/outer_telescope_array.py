import numpy as np
import skimage


def init_binning():
    b = {}
    b["num_bins_on_edge"] = 25
    b["num_bins_radius"] = b["num_bins_on_edge"] // 2
    b["center_bin"] = b["num_bins_radius"]
    return b


def init_telescope_positions_in_annulus(outer_radius, inner_radius):
    ccouter, rrouter = skimage.draw.disk(center=(0, 0), radius=outer_radius)
    ccinner, rrinner = skimage.draw.disk(center=(0, 0), radius=inner_radius)

    pos = set()
    for i in range(len(ccouter)):
        pos.add((ccouter[i], rrouter[i]))
    for i in range(len(ccinner)):
        pos.remove((ccinner[i], rrinner[i]))
    pos = list(pos)
    out = [[p[0], p[1]] for p in pos]
    return out


def init_mask_from_telescope_positions(positions):
    bb = init_binning()
    mask = np.zeros(
        shape=(bb["num_bins_on_edge"], bb["num_bins_on_edge"]), dtype=bool
    )
    for pos in positions:
        mask[pos[0] + bb["center_bin"], pos[1] + bb["center_bin"]] = True
    return mask


def make_example_config():
    cfg = {
        "mirror_diameter_m": 11.5,
        "positions": init_telescope_positions_in_annulus(
            outer_radius=2.5,
            inner_radius=0.5,
        ),
    }
    cfg["mask"] = init_mask_from_telescope_positions(
        positions=cfg["positions"]
    )
    return cfg
