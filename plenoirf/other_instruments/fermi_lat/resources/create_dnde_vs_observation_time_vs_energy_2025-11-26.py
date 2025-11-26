"""
This script uses fermipy to create the performance figures for
Cherenkov plenoscope project.

THIS SCRIPT RUNS IN A DEDICATED PYTHON ENVIRONMENT NAMED 'fermipy'.
NONE OF OUR TOOLS CAN BE IMPORTED HERE. THANK YOU FERMI-LAT!

installing fermipy
------------------

1st) Get fermipy

'''bash
mkdir fermi
cd fermi
git clone https://github.com/fermiPy/fermipy.git
'''

2nd) Install micromamba

'''bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
'''

3rd) Make miniconda environment

'''bash
cd fermipy/
miniconda create --name fermipy -f environment.yml clhep=2.4.4.1
'''

4th) Activate microconda and install fermipy

'''bash
micromamba activate fermipy
pip install -e fermipy/
'''


"""

import numpy as np
import fermipy
import subprocess
import astropy.io.fits
import datetime
import json


NUM_EBINS_PER_DECADE = 5  # Same as for CTA and the Cherenkov plenoscope
START_DECADE = 1
STOP_DECADE = 5

all_energy_edges_MeV = np.geomspace(
    10**START_DECADE,
    10**STOP_DECADE,
    NUM_EBINS_PER_DECADE * (STOP_DECADE - START_DECADE) + 1,
)

energy_edges_MeV = all_energy_edges_MeV[3:]

emin_MeV = energy_edges_MeV[0]
emax_MeV = energy_edges_MeV[-1]
NUM_ENERGY_BINS = len(energy_edges_MeV) - 1

ONE_YEAR_S = 365 * 24 * 3_600
ONE_MINUTE_S = 60

NUM_OBSERVATION_TIMES = 32

observation_times_s = np.geomspace(
    ONE_MINUTE_S,
    ONE_YEAR_S,
    NUM_OBSERVATION_TIMES,
)

FERMIPY_FLUX_SENSITIVITY_COMMAND = "fermipy-flux-sensitivity"

options = {
    "--glon": {
        "value": 120.0,
        "help": (
            "Galactic longitude in deg at which the sensitivity will be "
            "evaluated. Also sets the center of the sensitivity map for "
            "the `wcs` map type."
        ),
    },
    "--glat": {
        "value": 60.0,
        "help": (
            "Galactic latitude in deg at which the sensitivity will be "
            "evaluated. Also sets the center of the sensitivity map for "
            "the `wcs` map type."
        ),
    },
    "--emin": {"value": emin_MeV, "help": "Minimum energy in MeV."},
    "--emax": {"value": emax_MeV, "help": "Maximum energy in MeV."},
    "--nbin": {
        "value": NUM_ENERGY_BINS,
        "help": "Number of energy bins for differential flux calculation.",
    },
    "--ltcube": {
        "value": None,
        "help": (
            "Set the path to the livetime cube. If no livetime cube is "
            "provided the calculation will use an idealized observation "
            "profile for a uniform all-sky survey with no Earth obscuration "
            "or deadtime."
        ),
    },
    "--galdiff": {
        "value": "data/gll_iem_v06.fits",
        "help": (
            "Set the path to the galactic diffuse model used for "
            "fitting. This can be used to assess the impact of IEM systematics "
            "from fitting with the wrong model.  If none then the same model "
            "will be used for data and fit."
        ),
    },
    "--event_class": {
        "value": "P8R2_SOURCE_V6",
        "help": "Set the IRF name (e.g. P8R2_SOURCE_V6).",
    },
    "--min_counts": {
        "value": 10.0,
        "help": "Set the minimum number of counts.",
    },
    "--ts_thresh": {
        "value": 25.0,
        "help": "Set the test statistic (TS) detection threshold.",
    },
}


def call_fermipy_flux_sensitivity(options, observation_time_s, output_path):
    observation_time_years = observation_time_s / (365 * 24 * 3_600)
    call = [FERMIPY_FLUX_SENSITIVITY_COMMAND]
    for option in options:
        value = options[option]["value"]
        if value:
            call += [option, str(value)]

    call += ["--obs_time_yr", f"{observation_time_years:.7e}"]
    call += ["--output", output_path]
    return subprocess.call(call)


def read_table_1(path):
    with astropy.io.fits.open(path) as fin:
        table = fin[1]
        head = table.columns
        raw = table.data

    return head, raw


DNDE_UNIT = "cm-2 MeV-1 ph s-1"

result = {
    "energy_bin_edges": {"value": None, "unit": "MeV"},
    "observation_times": {
        "value": observation_times_s.tolist(),
        "unit": "s",
    },
    "dnde": {
        "value": np.nan
        * np.ones(shape=(NUM_OBSERVATION_TIMES, NUM_ENERGY_BINS)),
        "unit": DNDE_UNIT,
        "axes": ["observation_time", "energy"],
    },
    "provenance": {
        "creation_time": datetime.datetime.now().isoformat(),
        "author": "Sebastian A. Mueller",
        "fermipy": fermipy.__version__,
        "command": FERMIPY_FLUX_SENSITIVITY_COMMAND,
        "options": options,
    },
}

e_mins = np.nan * np.ones(shape=(NUM_OBSERVATION_TIMES, NUM_ENERGY_BINS))
e_maxs = np.nan * np.ones(shape=(NUM_OBSERVATION_TIMES, NUM_ENERGY_BINS))

for iobs in range(NUM_OBSERVATION_TIMES):
    ipath = f"fermi_{iobs:06d}.fits"

    rc = call_fermipy_flux_sensitivity(
        options=options,
        observation_time_s=result["observation_times"]["value"][iobs],
        output_path=ipath,
    )

    table_head, table_content = read_table_1(path=ipath)
    assert table_head["dnde"].unit == DNDE_UNIT
    result["dnde"]["value"][iobs] = table_content["dnde"]
    e_mins[iobs] = table_content["e_min"]
    e_maxs[iobs] = table_content["e_max"]


for i in range(NUM_OBSERVATION_TIMES - 1):
    np.testing.assert_array_almost_equal(e_mins[i], e_mins[i + 1])
    np.testing.assert_array_almost_equal(e_maxs[i], e_maxs[i + 1])

result["energy_bin_edges"]["value"] = e_mins[0].tolist() + [e_maxs[0][-1]]

result["dnde"]["value"] = result["dnde"]["value"].tolist()

with open("dnde_vs_obs_time.json", mode="wt") as fout:
    fout.write(json.dumps(result, indent=4))
