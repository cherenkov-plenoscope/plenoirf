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
    name="plenoirf",
    version=version,
    description="Explore magnetic deflection of cosmic-rays below 10GeV.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/cherenkov-plenoscope/plenoirf",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    packages=[
        "plenoirf",
        "plenoirf.analysis",
        "plenoirf.other_instruments",
        "plenoirf.reconstruction",
        "plenoirf.production",
        "plenoirf.summary",
        "plenoirf.features",
    ],
    install_requires=[
        "cosmic_fluxes",
        "corsika_primary",
        "atmospheric_cherenkov_response_cherenkov-plenoscope-project",
        "json_line_logger>=0.0.3",
        "propagate_uncertainties_sebastian-achim-mueller>=0.2.3",
        "shapely",
        "binning_utils_sebastian-achim-mueller",
        "json_numpy_sebastian-achim-mueller",
        "confusion_matrix_sebastian-achim-mueller>=0.0.4",
        "flux_sensitivity_sebastian-achim-mueller>=0.0.1",
        "rename_after_writing",
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
