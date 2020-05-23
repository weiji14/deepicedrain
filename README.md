# DeepIceDrain

Mapping Antarctic subglacial water using a deep neural network.

![GitHub top language](https://img.shields.io/github/languages/top/weiji14/deepicedrain.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Github Actions Status](https://github.com/weiji14/deepicedrain/workflows/Build%20DeepIceDrain/badge.svg)](https://github.com/weiji14/deepicedrain/actions)
[![Dependabot Status](https://api.dependabot.com/badges/status?host=github&repo=weiji14/deepicedrain)](https://dependabot.com)

# Getting started

## Quickstart

Launch in [Pangeo Binder](https://pangeo-binder.readthedocs.io) (Interactive jupyter lab environment in the cloud).

[![Binder](https://binder.pangeo.io/badge_logo.svg)](https://binder.pangeo.io/v2/gh/weiji14/deepicedrain/master)

## Installation

Start by cloning this [repo-url](/../../)

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

    jupyter labextension list  # ensure that extension is installed

## Running jupyter lab

    conda activate deepicedrain
    python -m ipykernel install --user --name deepicedrain  # to install conda env properly
    jupyter kernelspec list --json                          # see if kernel is installed
    jupyter lab &
