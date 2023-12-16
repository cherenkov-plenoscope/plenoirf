import atmospheric_cherenkov_response as acr
import magnetic_deflection as mdfl
import json_utils
import rename_after_writing as rnw


def make_config():
    c = {}
    c["sites"] = ["namibia", "chile"]
    c["particles"] = ["gamma", "electron", "proton", "helium"]
    c["magnetic_deflection"] = {"cherenkov_population_target": 2*1000*1000}
    return c



def init(work_dir):
    os.makedirs(work_dir, exist_ok=True)

    c = {}
    c["sites"] = {}

    config_dir = os.path.join(work_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    config_sites_dir = os.path.join(config_dir, "sites")
    os.makedirs(config_sites_dir, exist_ok=True)




