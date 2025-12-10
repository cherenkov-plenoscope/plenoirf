from plenoirf.binning import energy
import numpy as np


def test_energy_binning():
    bM = energy.init_from_decades(
        start_decade=-1,
        start_bin=2,
        stop_decade=3,
        stop_bin=3,
        num_bins_per_decade=5,
    )

    for N in [2, 3, 4, 5, 6, 7]:
        bN = energy.init_from_decades(
            start_decade=-1,
            start_bin=2 * N,
            stop_decade=3,
            stop_bin=(3 * N) - N + 1,
            num_bins_per_decade=5 * N,
        )

        print(N)
        assert bM["start"] == bN["start"]
        assert bM["stop"] == bN["stop"]
        np.testing.assert_array_equal(bM["limits"], bN["limits"])

    # multi     1       2       3       4       5
    # b/10      5       10      15      20      25
    # stop      3       5       7       9       11


def test_from_config():
    analysis_config = {
        "energy_binning": {
            "start": {"decade": -1, "bin": 2},
            "stop": {"decade": 3, "bin": 2},
            "num_bins_per_decade": 5,
            "fine": {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "12": 12},
        }
    }

    b1 = energy.init_from_analysis_config(
        analysis_config=analysis_config, key="1"
    )

    for key in analysis_config["energy_binning"]["fine"]:
        bK = energy.init_from_analysis_config(
            analysis_config=analysis_config, key=key
        )
        assert b1["start"] == bK["start"]
        assert b1["stop"] == bK["stop"]
