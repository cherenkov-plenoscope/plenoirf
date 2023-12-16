import numpy as np


def estimate_energy_resolution(
    true_energy, reco_energy, containment_fraction=0.68
):
    """
    Returns the energy-resolution according to the Cherenkov-Telescope-Array

    Parameters
    ----------
    true_energy : array-of-floats
            The true energy of the events.
    reco_energy : array-of-floats
            The reconstructed energy of the events in the same order as
            true_energy.
    containment_fraction: float
            The containment-fraction of events in the delta_E / E distribution
    """

    assert len(true_energy) == len(reco_energy)
    assert containment_fraction >= 0.0
    assert containment_fraction <= 1.0

    num_events = len(true_energy)
    if num_events > 0:
        delta_E_relunc = 1.0 / np.sqrt(num_events)
        delta_energy = np.abs(reco_energy - true_energy) / true_energy
        delta_energy_sorted = np.sort(delta_energy)
        delta_E = delta_energy_sorted[int(containment_fraction * num_events)]
    else:
        delta_E_relunc = float("nan")
        delta_E = float("nan")
    return delta_E, delta_E_relunc


def estimate_energy_resolution_vs_reco_energy(
    true_energy, reco_energy, reco_energy_bin_edges, containment_fraction=0.68
):
    """
    Returns the energy-resolution and its rel. uncertainty
    Adopted from the Cherenkov-Telescope-Array.

    Parameters
    ----------
    true_energy : array-of-floats
            The true energy of the events.
    reco_energy : array-of-floats
            The reconstructed energy of the events in the same order as
            true_energy.
    reco_energy_bin_edges : array-of-floats
            The bin-edges in the reconstructed energy to estimate the
            resolution in.
    containment_fraction: float
            The containment-fraction of events in the delta_E / E distribution

    Original comment from CTA
    -------------------------
    The energy resolution Delta E / E is obtained from the
    distribution of (ER – ET) / ET, where R and T refer to
    the reconstructed and true energies of gamma-ray events
    recorded by CTA. Delta E/E is the half-width of the interval
    around 0 which contains 68% of the distribution.
    The plot shows the energy resolution as a function of
    reconstructed energy  (the result depends only weakly on
    the assumed gamma-ray spectrum; for the results here
    we use dNgamma/dE ~E-2.62). The full energy migration
    matrix is provided, in each of the the IRF files, in
    two versions: one filled with all gamma events surviving
    the gamma/hadron separation cuts, suitable for cases in
    which there is no a priori knowledge of the true
    direction of incoming gamma rays (e.g. for the observation
    of diffuse sources), and another one filled after
    applying a cut on the angle between the true and the
    reconstructed gamma-ray direction
    (for observations of point-like objects) -
    the angular cut is the same used for the calculation of
    the point source sensitivity

    """
    assert len(true_energy) == len(reco_energy)
    assert len(reco_energy_bin_edges) >= 2
    assert np.all(np.gradient(reco_energy_bin_edges) > 0)
    assert containment_fraction >= 0.0
    assert containment_fraction <= 1.0

    delta_energy = []
    delta_energy_relunc = []
    for ebin in range(len(reco_energy_bin_edges) - 1):
        reco_energy_start = reco_energy_bin_edges[ebin]
        reco_energy_stop = reco_energy_bin_edges[ebin + 1]
        reco_energy_mask = np.logical_and(
            reco_energy >= reco_energy_start, reco_energy < reco_energy_stop
        )
        bin_true_energy = true_energy[reco_energy_mask]
        bin_reco_energy = reco_energy[reco_energy_mask]
        delta_E, delta_E_relunc = estimate_energy_resolution(
            true_energy=bin_true_energy,
            reco_energy=bin_reco_energy,
            containment_fraction=containment_fraction,
        )
        delta_energy.append(delta_E)
        delta_energy_relunc.append(delta_E_relunc)
    delta_energy = np.array(delta_energy)
    delta_energy_relunc = np.array(delta_energy_relunc)
    return delta_energy, delta_energy_relunc
