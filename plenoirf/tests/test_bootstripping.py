from plenoirf import bootstripping
import numpy as np
import pytest


def test_draw_simple():
    boot = bootstripping.draw(
        samples=np.arange(100), bootstrip=0, num_bootstrips=1
    )
    assert len(boot) == 100


def test_draw_empty():
    boot = bootstripping.draw(samples=[], bootstrip=0, num_bootstrips=1)
    assert len(boot) == 0


def test_bad_bootstrip():
    with pytest.raises(AssertionError) as aerr:
        boot = bootstripping.draw(samples=[], bootstrip=-1, num_bootstrips=1)

    for n in range(10):
        with pytest.raises(AssertionError) as aerr:
            boot = bootstripping.draw(
                samples=[], bootstrip=n, num_bootstrips=n
            )


def test_bad_num_bootstrip():
    with pytest.raises(AssertionError) as aerr:
        boot = bootstripping.draw(
            samples=[1, 2, 3], bootstrip=0, num_bootstrips=-1
        )


def test_splitting():
    NUM = 13
    SIZE = 10_000
    res = {}
    for i in range(NUM):
        res[i] = bootstripping.draw(
            samples=np.arange(SIZE), bootstrip=i, num_bootstrips=NUM
        )

    for i in range(NUM):
        si = set(res[i])
        assert 0.9 * (SIZE / NUM) <= len(si) < 1.1 * (SIZE / NUM)
        for j in range(NUM):
            if i != j:
                sj = set(res[j])
                inter = set.intersection(si, sj)
                assert len(inter) == 0
                assert abs(len(si) - len(sj)) < (SIZE / NUM) * 0.1
