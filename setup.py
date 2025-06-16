import setuptools
import os

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(os.path.join("plenoirf", "version.py")) as f:
    txt = f.read()
    last_line = txt.splitlines()[-1]
    version_string = last_line.split()[-1]
    version = version_string.strip("\"'")


setuptools.setup(
    name="plenoirf_cherenkov-plenoscope-project",
    version=version,
    description="Estimate Portal's instrument response function",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/cherenkov-plenoscope/plenoirf",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    packages=[
        "plenoirf",
        "plenoirf.bookkeeping",
        "plenoirf.configuration",
        "plenoirf.event_table",
        "plenoirf.ground_grid",
        "plenoirf.analysis",
        "plenoirf.other_instruments",
        "plenoirf.reconstruction",
        "plenoirf.production",
        "plenoirf.production.simulate_instrument_and_reconstruct_cherenkov",
        "plenoirf.summary",
        "plenoirf.features",
        "plenoirf.seeding",
        "plenoirf.logging",
    ],
    install_requires=[
        "xmltodict",
        "sympy>=1.11.1",
        "scikit-image",
        "un_bound_histogram>=0.0.1",
        "sparse_numeric_table_sebastian-achim-mueller>=1.0.6",
        "cosmic_fluxes_cherenkov-plenoscope-project>=0.0.3",
        "corsika_primary>=2.3.3",
        "atmospheric_cherenkov_response_cherenkov-plenoscope-project>=0.1.3",
        "json_line_logger>=0.0.3",
        "propagate_uncertainties_sebastian-achim-mueller>=0.2.3",
        "shapely",
        "binning_utils_sebastian-achim-mueller>=0.0.20",
        "json_numpy_sebastian-achim-mueller",
        "confusion_matrix_sebastian-achim-mueller>=0.0.4",
        "flux_sensitivity_sebastian-achim-mueller>=0.0.1",
        "rename_after_writing>=0.0.6",
        "gitpython>=3.1.40",
        "plenoptics_cherenkov-plenoscope-project>=0.0.8",
        "gamma_ray_reconstruction_cherenkov-plenoscope-project>=0.0.5",
        "magnetic_deflection_cherenkov-plenoscope-project>=0.14.10",
        "sebastians_matplotlib_addons>=0.0.17",
        "pylatex>=1.4.2",
    ],
    package_data={"plenoirf": [os.path.join("summary", "scripts", "*")]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
