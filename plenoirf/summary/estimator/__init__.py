from .zenith_energy_altitude_binning import Bins


def default_config():
    c = {}
    c["final_uids"] = ["0055_passing_trigger", "0056_passing_basic_quality"]
    c["signal"] = ["gamma"]
    c["background"] = ["proton", "helium"]
    return c


class Estimator:
    def __init__(self, script_resources, bins, config=None):
        self.res = script_resources
        self.bins = bins
        self.config = default_config()

    @property
    def final_uids(self):
        pass
