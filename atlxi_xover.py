# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: deepicedrain
#     language: python
#     name: deepicedrain
# ---

# %% [markdown]
# # **ICESat-2 Crossover Track Analysis**
#
# To increase the temporal resolution of
# our ice elevation change analysis
# (i.e. at time periods less than
# the 91 day repeat cycle of ICESat-2),
# we can look at the locations where the
# ICESat-2 tracks intersect and get the
# height values there!
# Uses [pygmt.x2sys_cross](https://www.pygmt.org/v0.2.0/api/generated/pygmt.x2sys_cross.html).
#
# References:
# - Wessel, P. (2010). Tools for analyzing intersecting tracks: The x2sys package.
# Computers & Geosciences, 36(3), 348–354. https://doi.org/10.1016/j.cageo.2009.05.009


# %%
import itertools
import os

import dask
import deepicedrain
import geopandas as gpd
import numpy as np
import pandas as pd
import pint
import pint_pandas
import pygmt
import shapely.geometry
import tqdm
import uncertainties

# %%
ureg = pint.UnitRegistry()
pint_pandas.PintType.ureg = ureg
tag: str = "X2SYS"
os.environ["X2SYS_HOME"] = os.path.abspath(tag)
client = dask.distributed.Client(
    n_workers=8, threads_per_worker=1, env={"X2SYS_HOME": os.environ["X2SYS_HOME"]}
)
client


# %%
min_date, max_date = ("2018-10-14", "2020-09-30")

# %%
# Initialize X2SYS database in the X2SYS/ICESAT2 folder
pygmt.x2sys_init(
    tag="ICESAT2",
    fmtfile=f"{tag}/ICESAT2/xyht",
    suffix="tsv",
    units=["de", "se"],  # distance in metres, speed in metres per second
    gap="d250e",  # distance gap up to 250 metres allowed
    force=True,
    verbose="q",
)

# %% [markdown]
# # Select a subglacial lake to examine

# %%
# Save or load dhdt data from Parquet file
placename: str = "siple_coast"  # "slessor_downstream"  #  "Recovery"  # "Whillans"
df_dhdt: pd.DataFrame = pd.read_parquet(f"ATLXI/df_dhdt_{placename.lower()}.parquet")


# %%
# Choose one Antarctic active subglacial lake polygon with EPSG:3031 coordinates
lake_name: str = "Whillans 12"
lake_catalog = deepicedrain.catalog.subglacial_lakes()
lake_ids: list = (
    pd.json_normalize(lake_catalog.metadata["lakedict"])
    .query("lakename == @lake_name")
    .ids.iloc[0]
)
lake = (
    lake_catalog.read()
    .loc[lake_ids]
    .dissolve(by=np.zeros(shape=len(lake_ids), dtype="int64"), as_index=False)
    .squeeze()
)

region = deepicedrain.Region.from_gdf(gdf=lake, name=lake_name)
draining: bool = lake.inner_dhdt < 0

print(lake)
lake.geometry

# %%
# Subset data to lake of interest
placename: str = region.name.lower().replace(" ", "_")
df_lake: cudf.DataFrame = region.subset(data=df_dhdt)  # bbox subset
gdf_lake = gpd.GeoDataFrame(
    df_lake, geometry=gpd.points_from_xy(x=df_lake.x, y=df_lake.y, crs=3031)
)
df_lake: pd.DataFrame = df_lake.loc[gdf_lake.within(lake.geometry)]  # polygon subset


# %%
# Run crossover analysis on all tracks
track_dict: dict = deepicedrain.split_tracks(df=df_lake)
rgts, tracks = track_dict.keys(), track_dict.values()
# Parallelized paired crossover analysis
futures: list = []
for rgt1, rgt2 in itertools.combinations(rgts, r=2):
    # skip if same referencegroundtrack but different laser pair
    # as they are parallel and won't cross
    if rgt1[:4] == rgt2[:4]:
        continue
    track1 = track_dict[rgt1][["x", "y", "h_corr", "utc_time"]]
    track2 = track_dict[rgt2][["x", "y", "h_corr", "utc_time"]]
    shape1 = shapely.geometry.LineString(coordinates=track1[["x", "y"]].to_numpy())
    shape2 = shapely.geometry.LineString(coordinates=track2[["x", "y"]].to_numpy())
    if not shape1.intersects(shape2):
        continue
    future = client.submit(
        key=f"{rgt1}x{rgt2}",
        func=pygmt.x2sys_cross,
        tracks=[track1, track2],
        tag="ICESAT2",
        # region=[-460000, -400000, -560000, -500000],
        interpolation="l",  # linear interpolation
        coe="e",  # external crossovers
        trackvalues=True,  # Get track 1 height (h_1) and track 2 height (h_2)
        # trackvalues=False,  # Get crossover error (h_X) and mean height value (h_M)
        # outfile="xover_236_562.tsv"
    )
    futures.append(future)


# %%
crossovers: dict = {}
for f in tqdm.tqdm(
    iterable=dask.distributed.as_completed(futures=futures), total=len(futures)
):
    if f.status != "error":  # skip those track pairs which don't intersect
        crossovers[f.key] = f.result().dropna().reset_index(drop=True)

df_cross: pd.DataFrame = pd.concat(objs=crossovers, names=["track1_track2", "id"])
df: pd.DataFrame = df_cross.reset_index(level="track1_track2").reset_index(drop=True)
# Report on how many unique crossover intersections there were
# df.plot.scatter(x="x", y="y")  # quick plot of our crossover points
print(
    f"{len(df.groupby(by=['x', 'y']))} crossover intersection point locations found "
    f"with {len(df)} crossover height-time pairs "
    f"over {len(tracks)} tracks"
)


# %%
# Calculate crossover error
df["h_X"]: pd.Series = df.h_2 - df.h_1  # crossover error (i.e. height difference)
df["t_D"]: pd.Series = df.t_2 - df.t_1  # elapsed time in ns (i.e. time difference)
ns_in_yr: int = 365.25 * 24 * 60 * 60 * 1_000_000_000  # nanoseconds in a year
df["dhdt"]: pd.Series = df.h_X / (df.t_D.astype(np.int64) / ns_in_yr)

# %%
# Get some summary statistics of our crossover errors
sumstats: pd.DataFrame = df[["h_X", "t_D", "dhdt"]].describe()
# Find location with highest absolute crossover error, and most sudden height change
max_h_X: pd.Series = df.iloc[np.nanargmax(df.h_X.abs())]  # highest crossover error
max_dhdt: pd.Series = df.iloc[df.dhdt.argmax()]  # most sudden change in height


# %% [markdown]
# ### 2D Map view of crossover points
#
# Bird's eye view of the crossover points
# overlaid on top of the ICESat-2 tracks.

# %%
# 2D plot of crossover locations
var: str = "h_X"
fig = pygmt.Figure()
# Setup basemap
plotregion = pygmt.info(table=df[["x", "y"]], spacing=1000)
pygmt.makecpt(cmap="batlow", series=[sumstats[var]["25%"], sumstats[var]["75%"]])
# Map frame in metre units
fig.basemap(frame="f", region=plotregion, projection="X8c")
# Plot actual track points in green
for track in tracks:
    tracklabel = f"{track.iloc[0].referencegroundtrack} {track.iloc[0].pairtrack}"
    fig.plot(
        x=track.x,
        y=track.y,
        pen="thinnest,green,.",
        style=f'qN+1:+l"{tracklabel}"+f3p,Helvetica,darkgreen',
    )
# Plot crossover point locations
fig.plot(x=df.x, y=df.y, color=df.h_X, cmap=True, style="c0.1c", pen="thinnest")
# Plot lake boundary in blue
lakex, lakey = lake.geometry.exterior.coords.xy
fig.plot(x=lakex, y=lakey, pen="thin,blue,-.")
# Map frame in kilometre units
fig.basemap(
    frame=[
        f'WSne+t"Crossover points at {region.name}"',
        'xaf+l"Polar Stereographic X (km)"',
        'yaf+l"Polar Stereographic Y (km)"',
    ],
    region=plotregion / 1000,
    projection="X8c",
)
fig.colorbar(position="JMR+e", frame=['x+l"Crossover Error"', "y+lm"])
fig.savefig(f"figures/{placename}/crossover_area_{placename}_{min_date}_{max_date}.png")
fig.show()

# %% [markdown]
# ### Plot Crossover Elevation time-series
#
# Plot elevation change over time at:
#
# 1. One single crossover point location
# 2. Many crossover locations over an area

# %%
# Tidy up dataframe first using pd.wide_to_long
# I.e. convert 't_1', 't_2', 'h_1', 'h_2' columns into just 't' and 'h'.
df_th: pd.DataFrame = deepicedrain.wide_to_long(
    df=df.loc[:, ["track1_track2", "x", "y", "t_1", "t_2", "h_1", "h_2"]],
    stubnames=["t", "h"],
    j="track",
)
df_th: pd.DataFrame = df_th.drop_duplicates(ignore_index=True)
df_th: pd.DataFrame = df_th.sort_values(by="t").reset_index(drop=True)

# %%
# Plot at single location with **maximum** absolute crossover height error (max_h_X)
df_max = df_th.query(expr="x == @max_h_X.x & y == @max_h_X.y")
track1, track2 = df_max.track1_track2.iloc[0].split("x")
print(f"{max_h_X.h_X:.2f} metres height change at {max_h_X.x}, {max_h_X.y}")
plotregion = np.array(
    [df_max.t.min(), df_max.t.max(), *pygmt.info(table=df_max[["h"]], spacing=2.5)[:2]]
)
plotregion += np.array([-pd.Timedelta(2, unit="W"), +pd.Timedelta(2, unit="W"), 0, 0])

fig = pygmt.Figure()
with pygmt.config(
    FONT_ANNOT_PRIMARY="9p", FORMAT_TIME_PRIMARY_MAP="abbreviated", FORMAT_DATE_MAP="o"
):
    fig.basemap(
        projection="X12c/8c",
        region=plotregion,
        frame=[
            f'WSne+t"Max elevation change over time at {region.name}"',
            "pxa1Of1o+lDate",  # primary time axis, 1 mOnth annotation and minor axis
            "sx1Y",  # secondary time axis, 1 Year intervals
            'yaf+l"Elevation at crossover (m)"',
        ],
    )
fig.text(
    text=f"Track {track1} and {track2} crossover",
    position="TC",
    offset="jTC0c/0.2c",
    V="q",
)
# Plot data points
fig.plot(x=df_max.t, y=df_max.h, style="c0.15c", color="darkblue", pen="thin")
# Plot dashed line connecting points
fig.plot(x=df_max.t, y=df_max.h, pen=f"faint,blue,-")
fig.savefig(
    f"figures/{placename}/crossover_point_{placename}_{track1}_{track2}_{min_date}_{max_date}.png"
)
fig.show()

# %%
# Plot all crossover height points over time over the lake area
fig = deepicedrain.plot_crossovers(df=df_th, regionname=region.name)
fig.savefig(f"figures/{placename}/crossover_many_{placename}_{min_date}_{max_date}.png")
fig.show()

# %%
# Calculate height anomaly at crossover point as
# height at t=n minus height at t=0 (first observation date at crossover point)
anomfunc = lambda h: h - h.iloc[0]  # lambda h: h - h.mean()
df_th["h_anom"] = df_th.groupby(by="track1_track2").h.transform(func=anomfunc)
# Calculate ice volume displacement (dvol) in unit metres^3
# and rolling mean height anomaly (h_roll) in unit metres
surface_area: pint.Quantity = lake.geometry.area * ureg.metre ** 2
ice_dvol: pd.Series = deepicedrain.ice_volume_over_time(
    df_elev=df_th.astype(dtype={"h_anom": "pint[metre]"}),
    surface_area=surface_area,
    time_col="t",
    outfile=f"figures/{placename}/ice_dvol_dt_{placename}.txt",
)
df_th["h_roll"]: pd.Series = uncertainties.unumpy.nominal_values(
    arr=ice_dvol.pint.magnitude / surface_area.magnitude
)

# %%
# Plot all crossover height point anomalies over time over the lake area
fig = deepicedrain.plot_crossovers(
    df=df_th,
    regionname=region.name,
    elev_var="h_anom",
    outline_points=f"figures/{placename}/{placename}.gmt",
)
fig.plot(x=df_th.t, y=df_th.h_roll, pen="thick,-")  # plot rolling mean height anomaly
fig.savefig(
    f"figures/{placename}/crossover_anomaly_{placename}_{min_date}_{max_date}.png"
)
fig.show()

# %%

# %% [markdown]
# ## Combined ice volume displacement plot
#
# Showing how subglacial water cascades down a drainage basin!

# %%
fig = pygmt.Figure()
fig.basemap(
    region=f"2019-02-28/2020-09-30/-0.3/0.5",
    frame=["wSnE", "xaf", 'yaf+l"Ice Volume Displacement (km@+3@+)"'],
)
pygmt.makecpt(cmap="davosS", color_model="+c", series=(-2, 4, 0.5))
for i, (_placename, linestyle) in enumerate(
    iterable=zip(
        ["whillans_ix", "subglacial_lake_whillans", "whillans_12", "whillans_7"],
        ["", ".-", "-", "..-"],
    )
):
    fig.plot(
        data=f"figures/{_placename}/ice_dvol_dt_{_placename}.txt",
        cmap=True,
        pen=f"thick,{linestyle}",
        zvalue=i,
        label=_placename,
        columns="0,3",  # time column (0), ice_dvol column (3)
    )
fig.text(
    position="TL",
    offset="j0.2c",
    text="Whillans Ice Stream Central Catchment active subglacial lakes",
)
fig.legend(position="jML+jML+o0.2c", box="+gwhite")
fig.savefig("figures/cascade_whillans_ice_stream.png")
fig.show()

# %%
