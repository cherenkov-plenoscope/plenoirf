import numpy as np
from .. import utils


def reweight(
    initial_energies,
    initial_rates,
    target_energies,
    target_rates,
    event_energies,
):
    assert len(initial_energies) == len(initial_rates)
    assert len(target_energies) == len(target_rates)

    relative_target_rate = target_rates / np.sum(target_rates)
    relative_initial_rate = initial_rates / np.sum(initial_rates)
    relative_initial_aban = np.interp(
        x=event_energies,
        xp=initial_energies,
        fp=relative_initial_rate,
    )
    relative_target_aban = np.interp(
        x=event_energies,
        xp=target_energies,
        fp=relative_target_rate,
    )
    weights = utils._divide_silent(
        numerator=relative_target_aban,
        denominator=relative_initial_aban,
        default=0.0,
    )

    return weights / np.max(weights)
