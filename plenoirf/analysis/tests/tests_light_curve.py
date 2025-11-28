from plenoirf.analysis.light_curve import estimate_max_rate_vs_observation_time
import numpy as np
import pytest


def test_max_rate_uniform():
    t = np.arange(100)

    max_rate, obs_time = estimate_max_rate_vs_observation_time(
        t=t,
        min_num_in_window=5,
    )

    assert 1.3 > np.max(max_rate) > 1.1
    assert 4 < min(obs_time) < 5


def test_max_rate_no_samples():
    with pytest.raises(AssertionError):
        _ = estimate_max_rate_vs_observation_time(
            t=[],
        )


def test_max_rate_too_few_samples():
    with pytest.raises(AssertionError):
        _ = estimate_max_rate_vs_observation_time(
            t=[0],
        )

    with pytest.raises(AssertionError):
        _ = estimate_max_rate_vs_observation_time(
            t=[0, 1],
        )


def test_max_rate_not_degenerated():
    with pytest.raises(AssertionError):
        _ = estimate_max_rate_vs_observation_time(
            t=np.ones(50),
        )


def test_max_rate_find_peak():
    t_start = 0
    t_stop = 1

    t = []
    # background with rate 1k
    t += np.linspace(0.0, 1.0, 1_000).tolist()
    # short burst with additional rate 1k
    t += np.linspace(0.0, 0.1, 1_00).tolist()

    max_rate, obs_time = estimate_max_rate_vs_observation_time(
        t=t,
        min_num_in_window=5,
    )

    assert len(max_rate) > 0

    assert 2_200 > np.max(max_rate) > 2_000
    assert 1_100 > np.min(max_rate) > 900

    for i in range(len(max_rate)):
        Rmax = max_rate[i]
        Tobs = obs_time[i]

        if 2_200 > Rmax > 2_000:
            assert Tobs < 0.15

        if 1_100 > Rmax > 900:
            assert Tobs > 0.5
