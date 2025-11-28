import numpy as np


def estimate_max_rate_vs_observation_time(
    t,
    min_num_in_window=5,
    window_duration_s=None,
    window_centers_s=None,
):
    """
    Parameters
    ----------
    t : array like
        Moments in time. E.g. arrival times of gamma rays in a light curve.
    """
    assert min_num_in_window > 0
    assert len(t) > 3
    t_order = np.sort(t)
    dt_order = np.diff(t_order)
    assert np.any(
        dt_order > 0
    ), "At least two points must have different times."
    min_dt_not_zero = np.min(dt_order[dt_order > 0])

    sqnum = int(np.ceil(np.sqrt(len(t))))

    if window_duration_s is None:
        min_window_duration_s = float(min_dt_not_zero)
        max_window_duration_s = max(t) - min(t)

        num_window_widths = 2 * sqnum
        window_durations_s = np.geomspace(
            min_window_duration_s,
            max_window_duration_s,
            num_window_widths,
        )
    num_window_widths = len(window_durations_s)
    assert np.all(window_durations_s > 0.0)

    if window_centers_s is None:
        num_window_centers = 2 * sqnum
        window_centers_s = np.linspace(
            min(t),
            max(t),
            num_window_centers,
        )
    num_window_centers = len(window_centers_s)

    max_rate_per_s = []
    valid_window_durations_s = []
    for ttt in range(num_window_widths):
        window_duration_s = window_durations_s[ttt]
        window_radiu_s = 0.5 * window_duration_s

        rates_in_windows_per_s = []
        for ccc in range(num_window_centers):
            window_center_s = window_centers_s[ccc]
            t_delta_s = np.abs(t - window_center_s)
            num_in_window = np.sum(t_delta_s <= window_radiu_s)
            if num_in_window >= min_num_in_window:
                rate_in_window_per_s = num_in_window / window_duration_s
                rates_in_windows_per_s.append(rate_in_window_per_s)

        if len(rates_in_windows_per_s) > 0:
            max_rate_per_s.append(np.max(rates_in_windows_per_s))
            valid_window_durations_s.append(window_duration_s)

    return np.array(max_rate_per_s), np.array(valid_window_durations_s)
