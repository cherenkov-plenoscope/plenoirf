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


def test_train_test_split():
    x = np.arange(1_000)

    for num_bootstrips in range(2, 10):
        for bootstrip in range(num_bootstrips):
            x_train, x_test = bootstripping.train_test_split(
                x=x,
                bootstrip=bootstrip,
                num_bootstrips=num_bootstrips,
            )

            assert len(x_train) > 0
            assert len(x_test) > 0
            assert len(x_train) + len(x_test) == len(x)

            intersection = set.intersection(set(x_train), set(x_test))
            assert len(intersection) == 0

            union = set.union(set(x_train), set(x_test))
            assert len(union) == len(x)

            expected_train_test_ratio = num_bootstrips - 1
            assert (
                0.95 * expected_train_test_ratio
                <= len(x_train) / len(x_test)
                < expected_train_test_ratio * 1.05
            )
