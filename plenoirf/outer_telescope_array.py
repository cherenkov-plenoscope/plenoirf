import numpy as np
import skimage

NUM_BINS_ON_EDGE = 25
NUM_BINS_RADIUS = NUM_BINS_ON_EDGE // 2
CENTER_BIN = NUM_BINS_RADIUS


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
    mask = np.zeros(shape=(NUM_BINS_ON_EDGE, NUM_BINS_ON_EDGE), dtype=bool)
    for pos in positions:
        mask[pos[0] + CENTER_BIN, pos[1] + CENTER_BIN] = True
    return mask


EXAMPLE_CONFIGURATION = {
    "mirror_diameter_m": 11.5,
    "positions": init_telescope_positions_in_annulus(
        outer_radius=2.5,
        inner_radius=0.5,
    ),
}
EXAMPLE_CONFIGURATION["mask"] = init_mask_from_telescope_positions(
    positions=EXAMPLE_CONFIGURATION["positions"]
)
