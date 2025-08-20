import numpy as np


def arange(num_events, bootstrip, num_bootstrips):
    assert 0 <= bootstrip < num_bootstrips
    assert 0 <= num_events
    assert num_bootstrips > 0
    return np.where(
        0 == np.mod(np.arange(num_events) - bootstrip, num_bootstrips)
    )


def draw(samples, bootstrip, num_bootstrips):
    samples = np.asarray(samples)
    arr = arange(
        num_events=samples.shape[0],
        bootstrip=bootstrip,
        num_bootstrips=num_bootstrips,
    )
    return samples[arr]
