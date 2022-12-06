# DeepIceDrain [[poster]](https://github.com/weiji14/nzasc2021)

Mapping and monitoring deep subglacial water activity
in Antarctica using remote sensing and machine learning.

[![Zenodo Digital Object Identifier](https://zenodo.org/badge/DOI/10.5281/zenodo.4071235.svg)](https://doi.org/10.5281/zenodo.4071235)
![GitHub top language](https://img.shields.io/github/languages/top/weiji14/deepicedrain.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Test DeepIceDrain package](https://github.com/weiji14/deepicedrain/actions/workflows/python-app.yml/badge.svg)](https://github.com/weiji14/deepicedrain/actions/workflows/python-app.yml)
[![Dependabot Status](https://api.dependabot.com/badges/status?host=github&repo=weiji14/deepicedrain)](https://dependabot.com)
![License](https://img.shields.io/github/license/weiji14/deepicedrain)

| Ice Surface Elevation trends over Antactica | Active Subglacial Lake fill-drain event |
|---|---|
| ![ICESat-2 ATL11 rate of height change over time in Antarctica 2019-03-29 to 2021-07-15](https://user-images.githubusercontent.com/23487320/138962441-1288f99c-1bfd-4405-a516-5deee5bb4492.png) | ![dsm_whillans_ix_cycles_3-9.gif](https://user-images.githubusercontent.com/23487320/124219379-5ed7ce00-db50-11eb-95d0-f1f660d4d688.gif) |

![DeepIceDrain Pipeline Part 1 Exploratory Data Analysis](https://yuml.me/diagram/scruffy;dir:LR/class/[Land-Ice-Elevation|atl06_play.ipynb]->[Convert|atl06_to_atl11.ipynb],[Convert]->[Land-Ice-Height-time-series|atl11_play.ipynb])
![DeepIceDrain Pipeline Part 2 Subglacial Lake Analysis](https://yuml.me/diagram/scruffy;dir:LR/class/[Height-Change-over-Time-(dhdt)|atlxi_dhdt.ipynb],[Height-Change-over-Time-(dhdt)]->[Subglacial-Lake-Finder|atlxi_lake.ipynb],[Subglacial-Lake-Finder]->[Crossover-Analysis|atlxi_xover.ipynb])

| Along track view of an ATL11 Ground Track | Elevation time-series at Crossover Points |
|---|---|
| ![alongtrack_whillans_ix_1080_pt3](https://user-images.githubusercontent.com/23487320/124219416-744cf800-db50-11eb-83a1-45e1e1159ba6.png) | ![crossover_anomaly_whillans_ix_2019-03-29_2020-12-24](https://user-images.githubusercontent.com/23487320/124219432-7a42d900-db50-11eb-92b4-c83728b8dc1c.png) |



# Getting started

## Quickstart

Launch in [Binder](https://mybinder.readthedocs.io) (Interactive jupyter lab environment in the cloud).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/weiji14/deepicedrain/main)

Alternative [Pangeo BinderHub](https://pangeo-binder.readthedocs.io) link.
Requires a GitHub account and you'll have to install your own computing environment,
but it runs on AWS uswest2 which allows for
[cloud access to ICESat-2](https://nsidc.org/data/user-resources/data-announcements/data-set-updates-new-earthdata-cloud-access-option-icesat-2-and-icesat-data-sets)!

[![Pangeo BinderHub](https://aws-uswest2-binder.pangeo.io/badge_logo.svg)](https://hub.aws-uswest2-binder.pangeo.io/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fweiji14%2Fdeepicedrain&urlpath=lab%2Ftree%2Fdeepicedrain%2F&branch=main)


## Usage

Once you've properly installed the [`deepicedrain` package](deepicedrain)
(see installation instructions further below), you'll have access to a
[wide range of tools](https://github.com/weiji14/deepicedrain/tree/main/deepicedrain)
for downloading and performing quick calculations on ICESat-2 datasets.
The example below shows how to calculate ice surface elevation change
on a sample ATL11 dataset between ICESat's Cycle 3 and Cycle 4.

    import deepicedrain
    import xarray as xr

    # Loads a sample ATL11 file from the intake catalog into xarray
    atl11_dataset: xr.Dataset = deepicedrain.catalog.test_data.atl11_test_case.read()

    # Calculate elevation change in metres from ICESat-2 Cycle 3 to Cycle 4
    delta_height: xr.DataArray = deepicedrain.calculate_delta(
          dataset=atl11_dataset, oldcyclenum=3, newcyclenum=4, variable="h_corr"
    )

    # Quick plot of delta_height along the ICESat-2 track
    delta_height.plot()

![ATL11 delta_height along ref_pt track](https://user-images.githubusercontent.com/23487320/83319030-bf7e4280-a28e-11ea-9bed-331e35dbc266.png)



## Installation

### Basic

To just try out the scripts, download the `environment.yml` file from the repository and run the commands below:

    cd deepicedrain
    mamba env create --name deepicedrain --file environment.yml
    pip install git+https://github.com/weiji14/deepicedrain.git

### Intermediate

To help out with development, start by cloning this [repo-url](/../../)

    git clone <repo-url>

Then I recommend [using mamba](https://mamba.readthedocs.io/en/latest/installation.html)
to install the non-python binaries.
A virtual environment will also be created with Python and
[poetry](https://github.com/python-poetry/poetry) installed.

    cd deepicedrain
    mamba env create --file environment.yml

Activate the virtual environment first.

    mamba activate deepicedrain

Then install the python libraries listed in the `pyproject.toml`/`poetry.lock` file.

    poetry install

Finally, double-check that the libraries have been installed.

    poetry show

### Advanced

This is for those who want full reproducibility of the virtual environment,
and more computing power by using Graphical Processing Units (GPU).

Making an explicit [conda-lock](https://github.com/conda-incubator/conda-lock) file
(only needed if creating a new virtual environment/refreshing an existing one).

    mamba env create -f environment.yml
    mamba list --explicit > environment-linux-64.lock

Creating/Installing a virtual environment from a conda lock file.
See also https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments.

    mamba create --name deepicedrain --file environment-linux-64.lock
    mamba install --name deepicedrain --file environment-linux-64.lock

If you have a [CUDA](https://en.wikipedia.org/wiki/CUDA)-capable GPU,
you can also install the optional "cuda" packages to accelerate some calculations.

    poetry install --extras cuda


## Running jupyter lab

    mamba activate deepicedrain
    python -m ipykernel install --user --name deepicedrain  # to install virtual env properly
    jupyter kernelspec list --json                          # see if kernel is installed
    jupyter lab &


## Related Projects

This work would not be possible without inspiration
from the following cool open source projects!
Go check them out if you have time.

- [ATL11](https://github.com/suzanne64/ATL11)
- [ICESAT-2 HackWeek](https://github.com/ICESAT-2HackWeek)
- [icepyx](https://github.com/icesat2py/icepyx)


## Citing

The work in this repository has not been peer-reviewed, but if you do want to
cite it for some reason, use the following BibLaTeX code from this conference
proceedings ([poster presentation](https://github.com/weiji14/nzasc2021)):

    @inproceedings{LeongSpatiotemporalvariabilityactive2021,
      title = {{Spatiotemporal Variability of Active Subglacial Lakes in Antarctica from 2018-2020 Using ICESat-2 Laser Altimetry}},
      author = {Leong, W. J. and Horgan, H. J.},
      date = {2021-02-10},
      publisher = {{Unpublished}},
      location = {{Christchurch, New Zealand}},
      doi = {10.13140/RG.2.2.27952.07680},
      eventtitle = {{New Zealand Antarctic Science Conference}}},
      langid = {english}
    }

Python code for the DeepIceDrain package here on Github is also mirrored on Zenodo at https://doi.org/10.5281/zenodo.4071235.
