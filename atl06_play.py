# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.4.2
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
# - Plotting figures using [Hvplot](https://hvplot.holoviz.org) and [PyGMT](https://www.pygmt.org) (TODO)
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

import dask
import dask.distributed
import hvplot.dask
import hvplot.pandas
import hvplot.xarray
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tqdm
import xarray as xr

# %matplotlib inline

# %%
# Configure intake and set number of compute cores for data download
intake.config.conf["cache_dir"] = "catdir"  # saves data to current folder
intake.config.conf["download_progress"] = False  # disable automatic tqdm progress bars

logging.basicConfig(level=logging.WARNING)

# Limit compute to 8 cores for download part using intake
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
catalog = intake.open_catalog(
    uri="catalog.yaml"
)  # open the local catalog file containing ICESAT2 stuff

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

dataset = (
    catalog.icesat2atl06.to_dask().unify_chunks()
)  # depends on .netrc file in home folder
dataset

# %%
# dataset.hvplot.points(
#    x="longitude", y="latitude", datashade=True, width=800, height=500, hover=True,
#    #geo=True, coastline=True, crs=cartopy.crs.PlateCarree(), #projection=cartopy.crs.Stereographic(central_latitude=-71),
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
dates3 = pd.date_range(start="2019.07.26", end="2020.03.06")  # 3rd batch
dates = dates1.append(other=dates2).append(other=dates3)

# %%
# Submit download jobs to Client
futures = []
for date in dates:
    source = catalog.icesat2atlasdownloader(date=date)
    future = client.submit(
        func=source.discover, key=f"download-{date}",
    )  # triggers download of the file(s), or loads from cache
    futures.append(future)

# %%
# Check download progress here, https://stackoverflow.com/a/37901797/6611055
responses = []
for f in tqdm.tqdm(
    iterable=dask.distributed.as_completed(futures=futures), total=len(futures)
):
    responses.append(f.result())

# %%
# In case of error, check which downloads are unfinished
# Manually delete those folders and retry
unfinished = []
for foo in futures:
    if foo.status != "finished":
        print(foo)
        unfinished.append(foo)
        # foo.retry()

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

# %% [raw]
# with tqdm.tqdm(total=len(dates)) as pbar:
#     for date in dates:
#         source = catalog.icesat2atlasdownloader(date=date)
#         source_urlpath = source.urlpath
#         try:
#             pbar.set_postfix_str(f"Obtaining files from {source_urlpath}")
#             source.discover()  # triggers download of the file(s), or loads from cache
#         except (requests.HTTPError, OSError, KeyError, TypeError) as error:
#             # clear cache and try again
#             print(f"Errored: {error}, trying again")
#             source.cache[0].clear_cache(urlpath=source_urlpath)
#             source.discover()
#         except (ValueError, pd.core.index.InvalidIndexError) as error:
#             print(f"Errored: {error}, ignoring")
#             pass
#         pbar.update(n=1)
#         #finally:
#         #    source.close()
#     #    del source

# %% [raw]
# catalog.icesat2atl06(date="2019.06.24", laser="gt1l").discover()  # ValueError??
# catalog.icesat2atl06(date="2019.02.28", laser="gt2l").discover()  # InvalidIndexError
# catalog.icesat2atl06(date="2019.11.13", laser="gt2l").discover()  # ValueError

# %%

# %% [markdown]
# ## Exploratory data analysis on local files
#
# Now that we've downloaded a good chunk of data and cached them locally,
# we can have some fun with visualizing the point clouds!

# %%
dataset = (
    catalog.icesat2atl06.to_dask()
)  # unfortunately, we have to load this in dask to get the path...
root_directory = os.path.dirname(os.path.dirname(dataset.encoding["source"]))

# %%
def get_crossing_dates(
    catalog_entry: intake.catalog.local.LocalCatalogEntry,
    root_directory: str,
    referencegroundtrack: str = "????",
    datetime="*",
    cyclenumber="??",
    orbitalsegment="??",
    version="003",
    revision="01",
):
    """
    Given a 4-digit reference groundtrack (e.g. 1234),
    we output a dictionary where the
    key is the date in "YYYY.MM.DD" format when an ICESAT2 crossing was made and the
    value is the filepath to the HDF5 data file.
    """

    # Get a glob string that looks like "ATL06_??????????????_XXXX????_002_01.h5"
    globpath = catalog_entry.path_as_pattern
    if datetime == "*":
        globpath = globpath.replace("{datetime:%Y%m%d%H%M%S}", "??????????????")
    globpath = globpath.format(
        referencegroundtrack=referencegroundtrack,
        cyclenumber=cyclenumber,
        orbitalsegment=orbitalsegment,
        version=version,
        revision=revision,
    )

    # Get list of filepaths (dates are contained in the filepath)
    globedpaths = glob.glob(os.path.join(root_directory, "??????????", globpath))

    # Pick out just the dates in "YYYY.MM.DD" format from the globedpaths
    # crossingdates = [os.path.basename(os.path.dirname(p=p)) for p in globedpaths]
    crossingdates = {
        os.path.basename(os.path.dirname(p=p)): p for p in sorted(globedpaths)
    }

    return crossingdates


# %%
crossing_dates_dict = {}
for rgt in range(0, 1388):  # ReferenceGroundTrack goes from 0001 to 1387
    referencegroundtrack = f"{rgt}".zfill(4)
    crossing_dates = dask.delayed(get_crossing_dates)(
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

# %% [raw]
# # For one laser along one reference ground track,
# # concatenate all points from all dates into one xr.Dataset
# da = xr.concat(
#     objs=(
#         catalog.icesat2atl06(date=date, laser="gt1r")
#         .to_dask()
#         .sel(referencegroundtrack=referencegroundtrack)
#         for date in crossing_dates
#     ),
#     dim=pd.Index(data=crossing_dates, name="crossingdates"),
# )

# %%
def six_laser_beams(crossing_dates: list):
    """
    For all 6 lasers along one reference ground track,
    concatenate all points from all crossing dates into one xr.Dataset
    """
    lasers = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]

    objs = [
        xr.open_mfdataset(
            paths=crossing_dates.values(),
            combine="nested",
            engine="h5netcdf",
            concat_dim="delta_time",
            group=f"{laser}/land_ice_segments",
            parallel=True,
        ).assign_coords(coords={"laser": laser})
        for laser in lasers
    ]

    try:
        da = xr.concat(
            objs=objs, dim="laser"
        )  # dim=pd.Index(data=lasers, name="laser")
        df = da.unify_chunks().to_dask_dataframe()
    except ValueError:
        # ValueError: cannot reindex or align along dimension 'delta_time' because the index has duplicate values
        df = dask.dataframe.concat(
            [obj.unify_chunks().to_dask_dataframe() for obj in objs]
        )

    return df


# %%
dataset_dict = {}
# for referencegroundtrack in list(crossing_dates_dict)[349:350]:   # ReferenceGroundTrack goes from 0001 to 1387
for referencegroundtrack in list(crossing_dates_dict)[
    340:350
]:  # ReferenceGroundTrack goes from 0001 to 1387
    # print(referencegroundtrack)
    if len(crossing_dates_dict[referencegroundtrack]) > 0:
        da = dask.delayed(six_laser_beams)(
            crossing_dates=crossing_dates_dict[referencegroundtrack]
        )
        # da = six_laser_beams(crossing_dates=crossing_dates_dict[referencegroundtrack])
        dataset_dict[referencegroundtrack] = da

# %%
df = dataset_dict["0349"].compute()  # loads into a dask dataframe (lazy)

# %%
df

# %%

# %%
dataset_dict = dask.compute(dataset_dict)[
    0
]  # compute every referencegroundtrack, slow... though somewhat parallelized

# %%
bdf = dask.dataframe.concat(dfs=list(dataset_dict.values()))

# %%

# %%
da.sel(crossingdates="2018.10.21").h_li.unify_chunks().drop(
    labels=["longitude", "datetime", "cyclenumber"]
).hvplot(
    kind="scatter",
    x="latitude",
    by="crossingdates",
    datashade=True,
    dynspread=True,
    width=800,
    height=500,
    dynamic=True,
    flip_xaxis=True,
    hover=True,
)

# %%

# %% [raw]
# # https://xarray.pydata.org/en/stable/combining.html#concatenate
# # For all 6 lasers one one date ~~along one reference ground track~~,
# # concatenate all points ~~from one dates~~ into one xr.Dataset
# lasers = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
# da = xr.concat(
#     objs=(
#         catalog.icesat2atl06(laser=laser)
#         .to_dask()
#         #.sel(referencegroundtrack=referencegroundtrack)
#         for laser in lasers
#     ),
#     dim=pd.Index(data=lasers, name="laser")
# )

# %%

# %% [markdown]
# ## Plot them points!

# %%
# convert dask.dataframe to pd.DataFrame
df = df.compute()

# %%
df = df.dropna(subset=["h_li"]).query(expr="atl06_quality_summary == 0").reset_index()

# %%
dfs = df.query(expr="0 <= segment_id - 1443620 < 900")
dfs

# %%
dfs.hvplot.scatter(
    x="longitude",
    y="latitude",
    by="laser",
    hover_cols=["delta_time", "segment_id"],
    # datashade=True, dynspread=True,
    # width=800, height=500, colorbar=True
)

# %%
import pyproj

# %%
transformer = pyproj.Transformer.from_crs(
    crs_from=pyproj.CRS.from_epsg(4326),
    crs_to=pyproj.CRS.from_epsg(3031),
    always_xy=True,
)

# %%
dfs["x"], dfs["y"] = transformer.transform(
    xx=dfs.longitude.values, yy=dfs.latitude.values
)

# %%
dfs

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
dfs.hvplot.scatter(x="x", y="h_li", by="laser")

# %%
dfs.to_pickle(path="icesat2_sample.pkl")

# %%

# %% [markdown]
# ## Old making a DEM grid surface from points

# %%
import scipy


# %%
# https://github.com/ICESAT-2HackWeek/gridding/blob/master/notebook/utils.py#L23
def make_grid(xmin, xmax, ymin, ymax, dx, dy):
    """Construct output grid-coordinates."""

    # Setup grid dimensions
    Nn = int((np.abs(ymax - ymin)) / dy) + 1
    Ne = int((np.abs(xmax - xmin)) / dx) + 1

    # Initiate x/y vectors for grid
    x_i = np.linspace(xmin, xmax, num=Ne)
    y_i = np.linspace(ymin, ymax, num=Nn)

    return np.meshgrid(x_i, y_i)


# %%
xi, yi = make_grid(
    xmin=dfs.x.min(), xmax=dfs.x.max(), ymin=dfs.y.max(), ymax=dfs.y.min(), dx=10, dy=10
)

# %%
ar = scipy.interpolate.griddata(points=(dfs.x, dfs.y), values=dfs.h_li, xi=(xi, yi))

# %%
plt.imshow(ar, extent=(dfs.x.min(), dfs.x.max(), dfs.y.min(), dfs.y.max()))

# %%

# %%
import plotly.express as px

# %%
px.scatter_3d(data_frame=dfs, x="longitude", y="latitude", z="h_li", color="laser")

# %%

# %% [markdown]
# ### Play using XrViz
#
# Install the PyViz JupyterLab extension first using the [extension manager](https://jupyterlab.readthedocs.io/en/stable/user/extensions.html#using-the-extension-manager) or via the command below:
#
# ```bash
# jupyter labextension install @pyviz/jupyterlab_pyviz@v0.8.0 --no-build
# jupyter labextension list  # check to see that extension is installed
# jupyter lab build --debug  # build extension ??? with debug messages printed
# ```
#
# Note: Had to add `network-timeout 600000` to `.yarnrc` file to resolve university network issues.

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
