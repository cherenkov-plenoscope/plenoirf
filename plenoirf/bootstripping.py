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


def train_test_split(x, bootstrip, num_bootstrips):
    x = np.asarray(x)
    x_test = draw(
        samples=x, bootstrip=bootstrip, num_bootstrips=num_bootstrips
    )
    _x_train = set.difference(set(x), set(x_test))
    x_train = np.array(list(_x_train), dtype=x.dtype)
    return x_train, x_test
