# DeepIceDrain

Mapping and monitoring deep subglacial water activity
in Antarctica using remote sensing and machine learning.

![GitHub top language](https://img.shields.io/github/languages/top/weiji14/deepicedrain.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![Test DeepIceDrain package](https://github.com/weiji14/deepicedrain/workflows/Test%20DeepIceDrain%20package/badge.svg)
[![Dependabot Status](https://api.dependabot.com/badges/status?host=github&repo=weiji14/deepicedrain)](https://dependabot.com)
![License](https://img.shields.io/github/license/weiji14/deepicedrain)

| Ice Surface Elevation trends over Antactica | Active Subglacial Lake filling event |
|---|---|
| ![ICESat-2 ATL11 rate of height change over time in Antarctica 2018-10-14 to 2020-07-16](https://user-images.githubusercontent.com/23487320/94127906-9755c780-fead-11ea-9a3d-622042e27541.png) | ![dsm_whillans_ix_cycles_3-8.gif](https://user-images.githubusercontent.com/23487320/97156701-f1fb7f80-17db-11eb-880c-87df2961e1c3.gif) |

![DeepIceDrain Pipeline](https://yuml.me/diagram/scruffy;dir:LR/class/[Land-Ice-Elevation|atl06_play.ipynb]->[Convert|atl06_to_atl11.ipynb],[Convert]->[Ice-Sheet-H(t)-Series|atl11_play.ipynb],[Ice-Sheet-H(t)-Series]->[Height-Change-over-Time-(dhdt)|atlxi_dhdt.ipynb],[Height-Change-over-Time-(dhdt)]->[Subglacial-Lake-Finder|atlxi_lake.ipynb])

| Along track view of an ATL11 Ground Track | Elevation time-series at Crossover Points |
|---|---|
| ![alongtrack_whillans_ix_1080_pt3](https://user-images.githubusercontent.com/23487320/97514092-ca323480-19f2-11eb-8d5c-c287b42eceda.png) | ![crossover_many_normalized_whillans_ix_2018-10-14_2020-07-16](https://user-images.githubusercontent.com/23487320/97514801-7e808a80-19f4-11eb-82b9-be45fccf2dc9.png) |



# Getting started

## Quickstart

Launch in [Pangeo Binder](https://pangeo-binder.readthedocs.io) (Interactive jupyter lab environment in the cloud).

[![Binder](https://binder.pangeo.io/badge_logo.svg)](https://binder.pangeo.io/v2/gh/weiji14/deepicedrain/master)

## Usage

Once you've properly installed the [`deepicedrain` package](deepicedrain)
(see installation instructions further below), you'll have access to a
[wide range of tools](https://github.com/weiji14/deepicedrain/tree/master/deepicedrain)
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
    conda env create --name deepicedrain --file environment.yml
    pip install git+https://github.com/weiji14/deepicedrain.git

### Intermediate

To help out with development, start by cloning this [repo-url](/../../)

    git clone <repo-url>

Then I recommend [using conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install the non-python binaries.
The conda virtual environment will also be created with Python and [poetry](https://github.com/python-poetry/poetry) installed.

    cd deepicedrain
    conda env create -f environment.yml

Activate the conda environment first.

    conda activate deepicedrain

Then install the python libraries listed in the `pyproject.toml`/`poetry.lock` file.

    poetry install

Finally, double-check that the libraries have been installed.

    poetry show

(Optional) Install jupyterlab extensions for interactive [bokeh](https://bokeh.org) visualizations.

    jupyter labextension install @pyviz/jupyterlab_pyviz
    jupyter labextension install dask-labextension

    jupyter labextension list  # ensure that extensions are installed


### Advanced

This is for those who want full reproducibility of the conda environment,
and more computing power by using Graphical Processing Units (GPU).

Making an explicit conda-lock file
(only needed if creating a new conda environment/refreshing an existing one).

    conda env create -f environment.yml
    conda list --explicit > environment-linux-64.lock

Creating/Installing a virtual environment from a conda lock file.
See also https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments.

    conda create --name deepicedrain --file environment-linux-64.lock
    conda install --name deepicedrain --file environment-linux-64.lock

If you have a [CUDA](https://en.wikipedia.org/wiki/CUDA)-capable GPU,
you can also install the optional "cuda" packages to accelerate some calculations.

    poetry install --extras cuda


## Running jupyter lab

    conda activate deepicedrain
    python -m ipykernel install --user --name deepicedrain  # to install conda env properly
    jupyter kernelspec list --json                          # see if kernel is installed
    jupyter lab &


## Related Projects

This work would not be possible without inspiration
from the following cool open source projects!
Go check them out if you have time.

- [ATL11](https://github.com/suzanne64/ATL11)
- [ICESAT-2 HackWeek](https://github.com/ICESAT-2HackWeek)
- [icepyx](https://github.com/icesat2py/icepyx)
