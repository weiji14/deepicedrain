# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: deepicedrain
#     language: python
#     name: deepicedrain
# ---

# %% [markdown]
# # **ATLAS/ICESat-2 Land Ice Height [ATL06](https://nsidc.org/data/atl06/) Exploratory Data Analysis**
#
# [Yet another](https://xkcd.com/927) take on playing with ICESat-2's Land Ice Height ATL06 data,
# specfically with a focus on analyzing ice elevation changes over Antarctica.
# Specifically, this jupyter notebook will cover:
#
# - Downloading datasets from the web via [intake](https://intake.readthedocs.io)
# - Performing [Exploratory Data Analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis)
#   using the [PyData](https://pydata.org) stack (e.g. [xarray](http://xarray.pydata.org), [dask](https://dask.org))
# - Plotting figures using [Hvplot](https://hvplot.holoviz.org) and [PyGMT](https://www.pygmt.org)
#
# This is in contrast with the [icepyx](https://github.com/icesat2py/icepyx) package
# and 'official' 2019/2020 [ICESat-2 Hackweek tutorials](https://github.com/ICESAT-2HackWeek/ICESat2_hackweek_tutorials) (which are also awesome!)
# that tends to use a slightly different approach (e.g. handcoded download scripts, [h5py](http://www.h5py.org) for data reading, etc).
# The core concept here is to run things in a more intuitive and scalable (parallelizable) manner on a continent scale (rather than just a specific region).

# %%
import glob
import json
import logging
import netrc
import os

import cartopy
import dask
import dask.distributed
import hvplot.dask
import hvplot.pandas
import hvplot.xarray
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import requests
import tqdm
import xarray as xr

import deepicedrain

# %%
# Limit compute to 10 cores for download part using intake
# Can possibly go up to 10 because there are 10 DPs?
# See https://n5eil02u.ecs.nsidc.org/opendap/hyrax/catalog.xml
client = dask.distributed.Client(n_workers=10, threads_per_worker=1)
client

# %% [markdown]
# ## Quick view
#
# Use our [intake catalog](https://intake.readthedocs.io/en/latest/catalog.html) to get some sample ATL06 data
# (while making sure we have our Earthdata credentials set up properly),
# and view it using [xarray](https://xarray.pydata.org) and [hvplot](https://hvplot.pyviz.org).

# %%
# Open the local intake data catalog file containing ICESat-2 stuff
catalog = intake.open_catalog("deepicedrain/atlas_catalog.yaml")
# or if the deepicedrain python package is installed, you can use either of the below:
# catalog = deepicedrain.catalog
# catalog = intake.cat.atlas_cat

# %%
try:
    netrc.netrc()
except FileNotFoundError as error_msg:
    print(
        f"{error_msg}, please follow instructions to create one at "
        "https://nsidc.org/support/faq/what-options-are-available-bulk-downloading-data-https-earthdata-login-enabled "
        'basically using `echo "machine urs.earthdata.nasa.gov login <uid> password <password>" >> ~/.netrc`'
    )
    raise

# data download will depend on having a .netrc file in home folder
dataset: xr.Dataset = catalog.icesat2atl06.to_dask().unify_chunks()
print(dataset)

# %%
# dataset.hvplot.points(
#     x="longitude",
#     y="latitude",
#     c="h_li",
#     cmap="Blues",
#     rasterize=True,
#     hover=True,
#     width=800,
#     height=500,
#     geo=True,
#     coastline=True,
#     crs=cartopy.crs.PlateCarree(),
#     projection=cartopy.crs.Stereographic(central_latitude=-71),
# )
catalog.icesat2atl06.hvplot.quickview()

# %% [markdown]
# ## Data intake
#
# Pulling in all of the raw ATL06 data (HDF5 format) from the NSIDC servers via an intake catalog file.
# Note that this will involve 100s if not 1000s of GBs of data, so make sure there's enough storage!!

# %%
# Download all ICESAT2 ATLAS hdf files from start to end date
dates1 = pd.date_range(start="2018.10.14", end="2018.12.08")  # 1st batch
dates2 = pd.date_range(start="2018.12.10", end="2019.06.26")  # 2nd batch
dates3 = pd.date_range(start="2019.07.26", end="2020.11.11")  # 3rd batch
dates = dates1.append(other=dates2).append(other=dates3)
# dates = pd.date_range(start="2020.09.30", end="2020.11.11")  # custom batch

# %%
# Submit download jobs to Client
futures = []
for date in dates:
    revision = 2 if date in pd.date_range(start="2020.04.22", end="2020.05.04") else 1
    source = catalog.icesat2atlasdownloader(date=date, revision=revision)
    future = client.submit(
        func=source.discover, key=f"download-{date}"
    )  # triggers download of the file(s), or loads from cache
    futures.append(future)

# %%
# Check download progress here, https://stackoverflow.com/a/37901797/6611055
responses = [f.result() for f in tqdm.tqdm(
    iterable=dask.distributed.as_completed(futures=futures), total=len(futures)
)]
# %%
# In case of error, check which downloads are unfinished
# Manually delete those folders and retry
unfinished = []
for foo in futures:
    if foo.status != "finished":
        print(foo)
        unfinished.append(foo)
        if foo.status == "error":
            foo.retry()
            # pass

# %%
try:
    assert len(unfinished) == 0
except AssertionError:
    for task in unfinished:
        print(task)
    raise ValueError(
        f"{len(unfinished)} download tasks are unfinished,"
        " please delete those folders and retry again!"
    )

# %%

# %% [markdown]
# ## Exploratory data analysis on local files
#
# Now that we've downloaded a good chunk of data and cached them locally,
# we can have some fun with visualizing the point clouds!

# %%
root_directory = os.path.dirname(
    catalog.icesat2atl06.storage_options["simplecache"]["cache_storage"]
)

# %%
def get_crossing_dates(
    catalog_entry: intake.catalog.local.LocalCatalogEntry,
    root_directory: str,
    referencegroundtrack: str = "????",
    datetimestr: str = "*",
    cyclenumber: str = "??",
    orbitalsegment: str = "??",
    version: str = "003",
    revision: str = "01",
) -> dict:
    """
    Given a 4-digit reference groundtrack (e.g. 1234),
    we output a dictionary where the
    key is the date in "YYYY.MM.DD" format when an ICESAT2 crossing was made and the
    value is the filepath to the HDF5 data file.
    """

    # Get a glob string that looks like "ATL06_??????????????_XXXX????_002_01.h5"
    globpath: str = catalog_entry.path_as_pattern
    if datetimestr == "*":
        globpath: str = globpath.replace("{datetime:%Y%m%d%H%M%S}", "??????????????")
    globpath: str = globpath.format(
        referencegroundtrack=referencegroundtrack,
        cyclenumber=cyclenumber,
        orbitalsegment=orbitalsegment,
        version=version,
        revision=revision,
    )

    # Get list of filepaths (dates are contained in the filepath)
    globedpaths: list = glob.glob(os.path.join(root_directory, "??????????", globpath))

    # Pick out just the dates in "YYYY.MM.DD" format from the globedpaths
    # crossingdates = [os.path.basename(os.path.dirname(p=p)) for p in globedpaths]
    crossingdates: dict = {
        os.path.basename(os.path.dirname(p=p)): p for p in sorted(globedpaths)
    }

    return crossingdates


# %%
crossing_dates_dict = {}
for rgt in range(1, 1388):  # ReferenceGroundTrack goes from 0001 to 1387
    referencegroundtrack: str = f"{rgt}".zfill(4)
    crossing_dates: dict = dask.delayed(get_crossing_dates)(
        catalog_entry=catalog.icesat2atl06,
        root_directory=root_directory,
        referencegroundtrack=referencegroundtrack,
    )
    crossing_dates_dict[referencegroundtrack] = crossing_dates
crossing_dates_dict = dask.compute(crossing_dates_dict)[0]

# %%
crossing_dates_dict["0349"].keys()


# %% [markdown]
# ![ICESat-2 Laser Beam Pattern](https://ars.els-cdn.com/content/image/1-s2.0-S0034425719303712-gr1.jpg)

# %%
def six_laser_beams(filepaths: list) -> dask.dataframe.DataFrame:
    """
    For all 6 lasers along one reference ground track,
    concatenate all points from all crossing dates into one Dask DataFrame

    E.g. if there are 5 crossing dates and 6 lasers,
    there would be data from 5 x 6 = 30 files being concatenated together.
    """
    lasers: list = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]

    objs: list = [
        xr.open_mfdataset(
            paths=filepaths,
            combine="by_coords",
            engine="h5netcdf",
            group=f"{laser}/land_ice_segments",
            parallel=True,
        ).assign_coords(coords={"laser": laser})
        for laser in lasers
    ]

    try:
        da: xr.Dataset = xr.concat(objs=objs, dim="laser")
        df: dask.dataframe.DataFrame = da.unify_chunks().to_dask_dataframe()
    except ValueError:
        # ValueError: cannot reindex or align along dimension 'delta_time'
        # because the index has duplicate values
        df: dask.dataframe.DataFrame = dask.dataframe.concat(
            [obj.unify_chunks().to_dask_dataframe() for obj in objs]
        )

    return df


# %%
dataset_dict = {}
# ReferenceGroundTrack goes from 0001 to 1387
for referencegroundtrack in list(crossing_dates_dict)[348:349]:
    # print(referencegroundtrack)
    filepaths = list(crossing_dates_dict[referencegroundtrack].values())
    if filepaths:
        dataset_dict[referencegroundtrack] = dask.delayed(obj=six_laser_beams)(
            filepaths=filepaths
        )
        # df = six_laser_beams(filepaths=filepaths)

# %%
df = dataset_dict["0349"].compute()  # loads into a dask dataframe (lazy)

# %%
df

# %%

# %%
# compute every referencegroundtrack, slow... though somewhat parallelized
# dataset_dict = dask.compute(dataset_dict)[0]

# %%
# big dataframe containing data across all 1387 reference ground tracks!
# bdf = dask.dataframe.concat(dfs=list(dataset_dict.values()))

# %%
# %% [raw]
# # https://xarray.pydata.org/en/stable/combining.html#concatenate
# # For all 6 lasers one one date ~~along one reference ground track~~,
# # concatenate all points ~~from one dates~~ into one xr.Dataset
# lasers = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
# da = xr.concat(
#     objs=(
#         catalog.icesat2atl06(laser=laser, referencegroundtrack=referencegroundtrack)
#         .to_dask()
#         for laser in lasers
#     ),
#     dim=pd.Index(data=lasers, name="laser")
# )

# %%

# %% [markdown]
# ## Plot ATL06 points!

# %%
# Convert dask.DataFrame to pd.DataFrame
df: pd.DataFrame = df.compute()

# %%
# Drop points with poor quality
df = df.dropna(subset=["h_li"]).query(expr="atl06_quality_summary == 0").reset_index()

# %%
# Get a small random sample of our data
dfs = df.sample(n=1_000, random_state=42)
dfs.head()

# %%
dfs.hvplot.scatter(
    x="longitude",
    y="latitude",
    by="laser",
    hover_cols=["delta_time", "segment_id"],
    # datashade=True, dynspread=True,
    # width=800, height=500, colorbar=True
)

# %% [markdown]
# ### Transform from EPSG:4326 (lat/lon) to EPSG:3031 (Antarctic Polar Stereographic)

# %%
dfs["x"], dfs["y"] = deepicedrain.lonlat_to_xy(
    longitude=dfs.longitude, latitude=dfs.latitude
)

# %%
dfs.head()

# %%
dfs.hvplot.scatter(
    x="x",
    y="y",
    by="laser",
    hover_cols=["delta_time", "segment_id", "h_li"],
    # datashade=True, dynspread=True,
    # width=800, height=500, colorbar=True
)

# %%
# Plot cross section view
dfs.hvplot.scatter(x="x", y="h_li", by="laser")

# %%

# %% [markdown]
# ## Experimental Work-in-Progress stuff below

# %% [markdown]
# ### Play using XrViz

# %%
import xrviz

# %%
xrviz.example()

# %%
# https://xrviz.readthedocs.io/en/latest/set_initial_parameters.html
initial_params = {
    # Select variable to plot
    "Variables": "h_li",
    # Set coordinates
    "Set Coords": ["longitude", "latitude"],
    # Axes
    "x": "longitude",
    "y": "latitude",
    # "sigma": "animate",
    # Projection
    # "is_geo": True,
    # "basemap": True,
    # "crs": "PlateCarree"
}
dashboard = xrviz.dashboard.Dashboard(data=dataset)  # , initial_params=initial_params)

# %%
dashboard.panel

# %%
dashboard.show()

# %%

# %% [markdown]
# ## OpenAltimetry

# %%
"minx=-154.56678505984297&miny=-88.82881451427136&maxx=-125.17872921546498&maxy=-81.34051361301398&date=2019-05-02&trackId=516"

# %%
# Paste the OpenAltimetry selection parameters here
OA_REFERENCE_URL = "minx=-177.64275595145213&miny=-88.12014866942751&maxx=-128.25920892322736&maxy=-85.52394234080862&date=2019-05-02&trackId=515"
# We populate a list with the photon data using the OpenAltimetry API, no HDF!
OA_URL = (
    "https://openaltimetry.org/data/icesat2/getPhotonData?client=jupyter&"
    + OA_REFERENCE_URL
)
OA_PHOTONS = ["Noise", "Low", "Medium", "High"]
# OA_PLOTTED_BEAMS = [1,2,3,4,5,6] you can select up to 6 beams for each ground track.
# Some beams may not be usable due cloud covering or QC issues.
OA_BEAMS = [3, 4]

# %%
minx, miny, maxx, maxy = [-156, -88, -127, -84]
date = "2019-05-02"  # UTC date?
track = 515  #
beam = 1  # 1 to 6
params = {
    "client": "jupyter",
    "minx": minx,
    "miny": miny,
    "maxx": maxx,
    "maxy": maxy,
    "date": date,
    "trackId": str(track),
    "beam": str(beam),
}

# %%
r = requests.get(
    url="https://openaltimetry.org/data/icesat2/getPhotonData", params=params
)

# %%
# OpenAltimetry Data cleansing
df = pd.io.json.json_normalize(data=r.json()["series"], meta="name", record_path="data")
df.name = df.name.str.split().str.get(0)  # Get e.g. just "Low" instead of "Low [12345]"
df.query(
    expr="name in ('Low', 'Medium', 'High')", inplace=True
)  # filter out Noise and Buffer points

df.rename(columns={0: "latitude", 1: "elevation", 2: "longitude"}, inplace=True)
df = df.reindex(
    columns=["longitude", "latitude", "elevation", "name"]
)  # reorder columns
df.reset_index(inplace=True)
df

# %%
df.hvplot.scatter(x="latitude", y="elevation")

# %%
