# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: deepicedrain
#     language: python
#     name: deepicedrain
# ---

# %% [markdown]
# # **ATLAS/ICESat-2 Land Ice Height Changes ATL11 Exploratory Data Analysis**
#
# Adapted from https://github.com/suzanne64/ATL11/blob/master/intro_to_ATL11.ipynb

# %%
import glob

import dask
import dask.array
import datashader
import geopandas as gpd
import holoviews as hv
import hvplot.dask
import hvplot.pandas
import hvplot.xarray
import intake
import matplotlib.cm
import numpy as np
import pandas as pd
import pygmt
import shapely
import xarray as xr

import deepicedrain

# %%
client = dask.distributed.Client(n_workers=32, threads_per_worker=1)
client

# %% [markdown]
# # Load Data from Zarr
#
# Let's start by getting our data and running some preprocessing steps:
# - Load 1387 (reference ground tracks) ATL11/*.zarr files
# - Convert coordinates from longitude/latitude to x/y
# - Convert GPS delta_time to UTC time
# - Mask out low quality height (h_corr) data

# %%
# Xarray open_dataset preprocessor to add fields based on input filename.
# Adapted from the intake.open_netcdf._add_path_to_ds function.
add_path_to_ds = lambda ds: ds.assign_coords(
    coords=intake.source.utils.reverse_format(
        format_string="ATL11.001z123/ATL11_{referencegroundtrack:04d}1x_{}_{}_{}.zarr",
        resolved_string=ds.encoding["source"],
    )
)

# Load dataset from all Zarr stores
# Aligning chunks spatially along cycle_number (i.e. time)
ds: xr.Dataset = xr.open_mfdataset(
    paths="ATL11.001z123/ATL11_*_003_01.zarr",
    chunks="auto",
    engine="zarr",
    combine="nested",
    concat_dim="ref_pt",
    parallel="True",
    preprocess=add_path_to_ds,
    backend_kwargs={"consolidated": True},
)
# ds = ds.unify_chunks().compute()
# TODO use intake, wait for https://github.com/intake/intake-xarray/issues/70
# source = intake.open_ndzarr(url="ATL11.001z123/ATL11_0*.zarr")
# %% [markdown]
# ## Convert geographic lon/lat to x/y
#
# To center our plot on the South Pole,
# we'll reproject the original longitude/latitude coordinates
# to the Antarctic Polar Stereographic (EPSG:3031) projection.

# %%
ds["x"], ds["y"] = deepicedrain.lonlat_to_xy(
    longitude=ds.longitude, latitude=ds.latitude
)


# %%
# Also set x/y as coordinates in xarray.Dataset instead of longitude/latitude
ds: xr.Dataset = ds.set_coords(names=["x", "y"])
ds: xr.Dataset = ds.reset_coords(names=["longitude", "latitude"])


# %% [markdown]
# ## Convert delta_time to utc_time
#
# To get more human-readable datetimes,
# we'll convert the delta_time attribute from the original GPS time format
# (nanoseconds since the beginning of ICESat-2 starting epoch)
# to Coordinated Universal Time (UTC).
# The reference date for the ICESat-2 Epoch is 2018 January 1st according to
# https://github.com/SmithB/pointCollection/blob/master/is2_calendar.py#L11-L15
#
# TODO: Account for [leap seconds](https://en.wikipedia.org/wiki/Leap_second)
# in the future.

# %%
# ds["utc_time"] = ds.delta_time.rename(new_name_or_name_dict="utc_time")
ds["utc_time"] = deepicedrain.deltatime_to_utctime(dataarray=ds.delta_time)


# %% [markdown]
# ## Mask out low quality height data
#
# Good quality data has value 0, not so good is > 0.
# Look at the 'fit_quality' attribute in `ds`
# for more information on what this quality flag means.
#
# We'll mask out values other than 0 with NaN using xarray's
# [where](http://xarray.pydata.org/en/v0.15.1/indexing.html#masking-with-where).

# %%
ds["h_corr"] = ds.h_corr.where(cond=ds.fit_quality == 0)

# %%

# %% [markdown]
# ## Subset to geographic region of interest (optional)
#
# Take a geographical subset and save to a NetCDF/Zarr format for distribution.

# %%
# Antarctic bounding box locations with EPSG:3031 coordinates
regions = gpd.read_file(filename="deepicedrain/deepicedrain_regions.geojson")
regions: gpd.GeoDataFrame = regions.set_index(keys="placename")

# Subset to essential columns
essential_columns: list = [
    "x",
    "y",
    "utc_time",
    "h_corr",
    "longitude",
    "latitude",
    "delta_time",
    "cycle_number",
]

# %%
# Do the actual computation to find data points within region of interest
placename: str = "kamb"  # Select Kamb Ice Stream region
region: deepicedrain.Region = deepicedrain.Region.from_gdf(gdf=regions.loc[placename])
ds_subset: xr.Dataset = region.subset(data=ds)
ds_subset = ds_subset.unify_chunks()
ds_subset = ds_subset.compute()

# %%
# Save to Zarr/NetCDF formats for distribution
ds_subset.to_zarr(
    store=f"ATLXI/ds_subset_{placename}.zarr", mode="w", consolidated=True
)
ds_subset.to_netcdf(path=f"ATLXI/ds_subset_{placename}.nc", engine="h5netcdf")

# %%
# Look at Cycle Number 7 only for plotting
points_subset = hv.Points(
    data=ds_subset.sel(cycle_number=7)[[*essential_columns]],
    label="Cycle_7",
    kdims=["x", "y"],
    vdims=["utc_time", "h_corr", "cycle_number", "referencegroundtrack"],
    datatype=["xarray"],
)
df_subset = points_subset.dframe()

# %%
# Plot our subset of points on an interactive map
df_subset.hvplot.points(
    title=f"Elevation (metres) at Cycle 7",
    x="x",
    y="y",
    c="referencegroundtrack",
    cmap="Set3",
    # rasterize=True,
    hover=True,
    datashade=True,
    dynspread=True,
)


# %% [markdown]
# # Pivot into a pandas/dask dataframe
#
# To make data analysis and plotting easier,
# let's flatten our n-dimensional `xarray.Dataset`
# to a 2-dimensiontal `pandas.DataFrame` table format.
#
# There are currently 8 cycles (as of July 2020),
# and by selecting just one cycle at a time,
# we can see what the height (`h_corr`)
# of the ice is like at that time.

# %% [markdown]
# ## Looking at ICESat-2 Cycle 7

# %%
cycle_number: int = 7
dss = ds.sel(cycle_number=cycle_number)[[*essential_columns]]
print(dss)

# %%
points = hv.Points(
    data=dss,
    label=f"Cycle_{cycle_number}",
    kdims=["x", "y"],
    vdims=["utc_time", "h_corr", "cycle_number"],
    datatype=["xarray"],
)

# %%
df = points.dframe()  # convert to pandas.DataFrame, slow
df = df.dropna()  # drop empty rows
print(len(df))
df.head()

# %% [markdown]
# ### Plot a sample of the points over Antarctica
#
# Let's take a look at an interactive map
# of the ICESat-2 ATL11 height for Cycle 6!
# We'll plot a random sample (n=5 million)
# of the points instead of the whole dataset,
# it should give a good enough picture.

# %%
df.sample(n=5_000_000).hvplot.points(
    title=f"Elevation (metres) at Cycle {cycle_number}",
    x="x",
    y="y",
    c="h_corr",
    cmap="Blues",
    rasterize=True,
    hover=True,
)

# %%

# %% [markdown]
# # Calculate Elevation Change (dh) over ICESAT-2 cycles!!
#
# Let's take a look at the change in elevation over a year (4 ICESat-2 cycles).
# From our loaded dataset (ds), we'll select Cycles 3 and 7,
# and subtract the height (h_corr) between them to get a height difference (dh).

# %%
dh: xr.DataArray = deepicedrain.calculate_delta(
    dataset=ds, oldcyclenum=3, newcyclenum=7, variable="h_corr"
)


# %%
# Persist data in memory
dh = dh.persist()

# %%
delta_h: xr.Dataset = dh.dropna(dim="ref_pt").to_dataset(name="delta_height")
print(delta_h)

# %%
df_dh: pd.DataFrame = delta_h.to_dataframe()
print(df_dh.head())

# %%
# Save or Load delta height data
# df_dh.to_parquet(f"ATLXI/df_dh_{placename}.parquet")
# df_dh: pd.DataFrame = pd.read_parquet(f"ATLXI/df_dh_{placename}.parquet")
# df_dh = df_dh.sample(n=1_000_000)

# %% [markdown]
# ## Plot elevation difference for a region
#
# Using [datashader](https://datashader.org) to make the plotting real fast,
# it actually rasterizes the vector points into a raster grid,
# since our eyes can't see millions of points that well anyway.
# You can choose any region, but we'll focus on the Siple Coast Ice Streams.
# Using [PyGMT](https://pygmt.org), we'll plot the Antarctic grounding line
# as well as the ATL11 height changes overlaid with Subglacial Lake outlines
# from [Smith et al., 2009](https://doi.org/10.3189/002214309789470879).

# %%
# Select region here, see also geodataframe of regions at top
placename: str = "antarctica"
region: deepicedrain.Region = deepicedrain.Region.from_gdf(gdf=regions.loc[placename])

# %%
# Find subglacial lakes (Smith et al., 2009) within region of interest
subglacial_lakes_gdf = gpd.read_file(
    filename=r"Quantarctica3/Glaciology/Subglacial Lakes/SubglacialLakes_Smith.shp"
)
subglacial_lakes_gdf = subglacial_lakes_gdf.loc[
    subglacial_lakes_gdf.within(
        shapely.geometry.Polygon.from_bounds(*region.bounds(style="lbrt"))
    )
]
subglacial_lakes_geom = [g for g in subglacial_lakes_gdf.geometry]
subglacial_lakes = [
    np.dstack(g.exterior.coords.xy).squeeze().astype(np.float32)
    for g in subglacial_lakes_geom
]


# %%
# Datashade our height values (vector points) onto a grid (raster image)
agg_grid: xr.DataArray = region.datashade(df=df_dh, z_dim="delta_height")
print(agg_grid)

# %%
# Plot our map!
scale: int = region.scale
fig = pygmt.Figure()
# fig.grdimage(
#    grid="Quantarctica3/SatelliteImagery/MODIS/MODIS_Mosaic.tif",
#    region=region,
#    projection=f"x{scale}",
#    I="+d",
# )
pygmt.makecpt(cmap="roma", series=[-2, 2])
fig.grdimage(
    grid=agg_grid,
    region=region.bounds(),
    projection=f"x1:{scale}",
    frame=["afg", f'WSne+t"ICESat-2 Ice Surface Change over {region.name}"'],
    Q=True,
)
for subglacial_lake in subglacial_lakes:
    fig.plot(data=subglacial_lake, L=True, pen="thinnest")
fig.colorbar(
    position="JCR+e", frame=["af", 'x+l"Elevation Change from Cycle 3 to 7"', "y+lm"]
)
fig.coast(
    region=region.bounds(),
    projection=f"s0/-90/-71/1:{scale}",
    area_thresh="+ag",
    resolution="i",
    shorelines="0.5p",
    # land="snow4",
    # water="snow3",
    V="q",
)
fig.savefig(f"figures/plot_atl11_dh37_{placename}.png")
fig.show(width=600)


# %%

# %% [markdown]
# #### Non-PyGMT plotting code on PyViz stack
#
# Meant to be a bit more interactive but slightly buggy,
# need to sort out python dependency issues.

# %%
shade_grid = datashader.transfer_functions.shade(
    agg=agg_grid, cmap=matplotlib.cm.RdYlBu, how="linear", span=[-2, 2]
)
spread_grid = datashader.transfer_functions.dynspread(shade_grid)
spread_grid

# %%
df_dh.hvplot.points(
    # title="Elevation Change (metres) from Cycle 5 to 6",
    x="x",
    y="y",
    c="delta_height",
    # cmap="RdYlBu",
    # aggregator=datashader.mean("delta_height"),
    rasterize=True,
    # responsive=True,
    # datashade=True,
    # dynamic=True,
    # dynspread=True,
    hover=True,
    height=400,
    symmetric=True,
    clim=(-20, 20),
)

# %%
points = hv.Points(
    data=df_dh,
    kdims=["x", "y"],
    vdims=["delta_height"],
    # datatype=["xarray"],
)

# %%
hv.operation.datashader.datashade(points)


# %%
