# -*- coding: utf-8 -*-
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
# # **ICESat-2 Active Subglacial Lakes in Antarctica**
#
# Finding subglacial lakes that are draining or filling under the ice!
# They can be detected with ICESat-2 data, as significant changes in height
# (> 1 metre) over a relatively short duration (< 1 year), i.e. a high rate of
# elevation change over time (dhdt).
#
# In this notebook, we'll use some neat tools to help us examine the lakes:
# - To find active subglacial lake boundaries,
# use an *unsupervised clustering* technique
# - To see ice surface elevation trends at a higher temporal resolution,
# perform *crossover track error analysis* on intersecting ICESat-2 tracks
#
# To speed up analysis on millions of points,
# we will use state of the art GPU algorithms enabled by RAPIDS AI libraries,
# or parallelize the processing across our HPC's many CPU cores using Dask.

# %%
import os

import numpy as np

import cudf
import cuml
import dask
import dask.array
import deepicedrain
import geopandas as gpd
import hvplot.cudf
import panel as pn
import pygmt
import scipy.spatial
import shapely.geometry
import tqdm
import zarr

# %% [markdown]
# # Data Preparation

# %%
if not os.path.exists("ATLXI/df_dhdt_antarctica.parquet"):
    zarrarray = zarr.open_consolidated(store=f"ATLXI/ds_dhdt_antarctica.zarr", mode="r")
    _ = deepicedrain.zarr_to_parquet(
        zarrarray=zarrarray,
        parquetpath="ATLXI/df_dhdt_antarctica.parquet",
        variables=["x", "y", "dhdt_slope", "referencegroundtrack", "h_corr"],
        dropnacols=["dhdt_slope"],
    )

# %% [markdown]
# ## Load in ICESat-2 data (x, y, dhdt) and do initial trimming

# %%
# Read in raw x, y, dhdt_slope and referencegroundtrack data into the GPU
cudf_raw: cudf.DataFrame = cudf.read_parquet(
    filepath_or_buffer="ATLXI/df_dhdt_antarctica.parquet",
    columns=["x", "y", "dhdt_slope", "referencegroundtrack"],
)
# Filter to points with dhdt that is less than -1 m/yr or more than +1 m/yr
cudf_many = cudf_raw.loc[abs(cudf_raw.dhdt_slope) > 1]
print(f"Trimmed {len(cudf_raw)} -> {len(cudf_many)}")

# %% [markdown]
# ## Label ICESat-2 points according to their drainage basin
#
# Uses Point in Polygon.
# For each point, find out which Antarctic Drainage Basin they are in.
# This will also remove the points on floating (FR) ice shelves and islands (IS),
# so that we keep only points on the grounded (GR) ice regions.


# %%
# Read in Antarctic Drainage Basin Boundaries shapefile into a GeoDataFrame
ice_boundaries: gpd.GeoDataFrame = gpd.read_file(
    filename="Quantarctica3/Glaciology/MEaSUREs Antarctic Boundaries/IceBoundaries_Antarctica_v2.shp"
)
drainage_basins: gpd.GeoDataFrame = ice_boundaries.query(expr="TYPE == 'GR'")


# %%
# Use point in polygon to label points according to the drainage basins they fall in
cudf_many["drainage_basin"]: cudf.Series = deepicedrain.point_in_polygon_gpu(
    points_df=cudf_many, poly_df=drainage_basins
)
X_many = cudf_many.dropna()  # drop points that are not in a drainage basin

# %% [markdown]
# # Find Active Subglacial Lake clusters
#
# Uses Density-based spatial clustering of applications with noise (DBSCAN).

# %%
def find_clusters(X: cudf.core.dataframe.DataFrame) -> cudf.core.series.Series:
    """
    Density-based spatial clustering of applications with noise (DBSCAN)
    See also https://www.naftaliharris.com/blog/visualizing-dbscan-clustering
    """
    # Run DBSCAN using 2500 m distance, and minimum of 250 points
    dbscan = cuml.DBSCAN(eps=2500, min_samples=250)
    dbscan.fit(X=X)

    cluster_labels = dbscan.labels_ + 1  # noise points -1 becomes 0
    cluster_labels = cluster_labels.mask(cond=cluster_labels == 0)  # turn 0 to NaN
    cluster_labels.index = X.index  # let labels have same index as input data

    return cluster_labels


# %%
# Subglacial lake finder
activelakes: dict = {
    "basin_name": [],
    "maxabsdhdt": [],
    "refgtracks": [],
    "geometry": [],
}
# for basin_index in tqdm.tqdm(
#     iterable=drainage_basins[drainage_basins.NAME.str.startswith("Whillans")].index
# ):
for basin_index in tqdm.tqdm(iterable=drainage_basins.index):
    # Initial data cleaning, filter to rows that are in the drainage basin
    basin = drainage_basins.loc[basin_index]
    X = X_many.loc[X_many.drainage_basin == basin.NAME]  # .reset_index(drop=True)
    if len(X) <= 1000:  # don't run on too few points
        continue
    print(f"{len(X)} rows at {basin.NAME}")

    # Run unsupervised clustering separately on draining and filling lakes
    # Draining lake points have negative labels (e.g. -1, -2, 3),
    # Filling lake points have positive labels (e.g. 1, 2, 3),
    # Noise points have NaN labels (i.e. NaN)
    cluster_vars = ["x", "y", "dhdt_slope"]
    draining_lake_labels = -find_clusters(X=X.loc[X.dhdt_slope < 0][cluster_vars])
    filling_lake_labels = find_clusters(X=X.loc[X.dhdt_slope > 0][cluster_vars])
    lake_labels = cudf.concat(objs=[draining_lake_labels, filling_lake_labels])
    lake_labels.name = "cluster_label"

    clusters: cudf.Series = lake_labels.unique()
    print(
        f"{(clusters < 0).sum()} draining and {(clusters > 0).sum()} filling lakes found"
    )

    for cluster_label in clusters.to_array():
        # Store attribute and geometry information of each active lake
        lake_points: cudf.DataFrame = X.loc[lake_labels == cluster_label]

        try:
            assert len(lake_points) > 2
        except AssertionError:
            continue

        multipoint: shapely.geometry.MultiPoint = shapely.geometry.MultiPoint(
            points=lake_points[["x", "y"]].as_matrix()
        )
        convexhull: shapely.geometry.Polygon = multipoint.convex_hull

        maxabsdhdt: float = (
            lake_points.dhdt_slope.max()
            if cluster_label > 0  # positive label = filling
            else lake_points.dhdt_slope.min()  # negative label = draining
        )
        refgtracks: str = "|".join(
            map(str, lake_points.referencegroundtrack.unique().to_pandas())
        )

        activelakes["basin_name"].append(basin.NAME)
        activelakes["maxabsdhdt"].append(maxabsdhdt)
        activelakes["refgtracks"].append(refgtracks)
        activelakes["geometry"].append(convexhull)

if len(activelakes["geometry"]) >= 1:
    gdf = gpd.GeoDataFrame(activelakes, crs="EPSG:3031")
    gdf.to_file(filename="antarctic_subglacial_lakes.geojson", driver="GeoJSON")

print(f"Total of {len(gdf)} subglacial lakes founds")

# %% [markdown]
# ## Visualize lakes

# %%
# Concatenate XY points with labels, and move data from GPU to CPU
X: cudf.DataFrame = cudf.concat(objs=[X, lake_labels], axis="columns")
X_ = X.to_pandas()


# %%
# Plot clusters on a map in colour, noise points/outliers as small dots
fig = pygmt.Figure()
n_clusters_ = len(X_.cluster_label.unique()) - 1  # No. of clusters minus noise (NaN)
sizes = (X_.cluster_label.isna()).map(arg={True: 0.01, False: 0.1})
if n_clusters_:
    pygmt.makecpt(cmap="polar+h0", series=(-1.5, 1.5, 1), reverse=True, D=True)
else:
    pygmt.makecpt(cmap="gray")
fig.plot(
    x=X_.x,
    y=X_.y,
    sizes=sizes,
    style="cc",
    color=X_.cluster_label,
    cmap=True,
    frame=[
        f'WSne+t"Estimated number of lake clusters at {basin.NAME}: {n_clusters_}"',
        'xaf+l"Polar Stereographic X (m)"',
        'yaf+l"Polar Stereographic Y (m)"',
    ],
)
basinx, basiny = basin.geometry.exterior.coords.xy
fig.plot(x=basinx, y=basiny, pen="thinnest,-")
fig.colorbar(frame='af+l"Draining/Filling"', position='JBC+n"Unclassified"')
fig.savefig(fname=f"figures/subglacial_lake_clusters_at_{basin.NAME}.png")
fig.show()

# %% [markdown]
# # Select a lake to examine

# %%
# Save or load dhdt data from Parquet file
placename: str = "Recovery"  # "Whillans"
drainage_basins: gpd.GeoDataFrame = drainage_basins.set_index(keys="NAME")
region: deepicedrain.Region = deepicedrain.Region.from_gdf(
    gdf=drainage_basins.loc[placename], name="Recovery Basin"
)
df_dhdt: cudf.DataFrame = cudf.read_parquet(
    f"ATLXI/df_dhdt_{placename.lower()}.parquet"
)

# %%
# Antarctic subglacial lake polygons with EPSG:3031 coordinates
antarctic_lakes: gpd.GeoDataFrame = gpd.read_file(
    filename="antarctic_subglacial_lakes.geojson"
)

# %%
# Choose one draining/filling lake
draining: bool = True  # False
placename: str = "Slessor"  # "Whillans"
lakes: gpd.GeoDataFrame = antarctic_lakes.query(expr="basin_name == @placename")
lake = lakes.loc[lakes.maxabsdhdt.idxmin() if draining else lakes.maxabsdhdt.idxmax()]
lakedict = {
    76: "Subglacial Lake Conway",  # draining lake
    78: "Whillans IX",  # filling lake
    103: "Slessor 45",  # draining lake
    108: "Slessor 23",  # filling lake
}
region = deepicedrain.Region.from_gdf(gdf=lake, name=lakedict[lake.name])

# %%
# Subset data to lake of interest
placename: str = region.name.lower().replace(" ", "_")
df_lake: cudf.DataFrame = region.subset(data=df_dhdt)


# %% [markdown]
# # Crossover Track Analysis
#
# To increase the temporal resolution of
# our ice elevation change analysis
# (i.e. at time periods less than
# the 91 day repeat cycle of ICESat-2),
# we can look at the locations where the
# ICESat-2 tracks intersect and get the
# height values there!
# Uses [x2sys_cross](https://docs.generic-mapping-tools.org/6.1/supplements/x2sys/x2sys_cross).
#
# References:
# - Wessel, P. (2010). Tools for analyzing intersecting tracks: The x2sys package.
# Computers & Geosciences, 36(3), 348–354. https://doi.org/10.1016/j.cageo.2009.05.009


# %%
# Initialize X2SYS database in the X2SYS/ICESAT2 folder
tag = "X2SYS"
os.environ["X2SYS_HOME"] = os.path.abspath(tag)
os.getcwd()
pygmt.x2sys_init(
    tag="ICESAT2",
    fmtfile=f"{tag}/ICESAT2/xyht",
    suffix="tsv",
    units=["de", "se"],  # distance in metres, speed in metres per second
    gap="d250e",  # distance gap up to 250 metres allowed
    force=True,
    verbose="q",
)

# %%
# Run crossover analysis on all tracks
rgts: list = [135, 327, 388, 577, 1080, 1272]  # Whillans upstream
# rgts: list = [236, 501, 562, 1181]  # Whillans_downstream
tracks = [f"{tag}/track_{i}.tsv" for i in rgts]
assert all(os.path.exists(k) for k in tracks)

# Parallelized paired crossover analysis
futures: list = []
for track1, track2 in itertools.combinations(rgts, r=2):
    future = client.submit(
        key=f"{track1}_{track2}",
        func=pygmt.x2sys_cross,
        tracks=[f"{tag}/track_{track1}.tsv", f"{tag}/track_{track2}.tsv"],
        tag="ICESAT2",
        region=[-460000, -400000, -560000, -500000],
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
ns_in_yr: int = (365.25 * 24 * 60 * 60 * 1_000_000_000)  # nanoseconds in a year
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
region = np.array([df.x.min(), df.x.max(), df.y.min(), df.y.max()])
buffer = np.array([-2000, +2000, -2000, +2000])
pygmt.makecpt(cmap="batlow", series=[sumstats[var]["25%"], sumstats[var]["75%"]])
# Map frame in metre units
fig.basemap(frame="f", region=region + buffer, projection="X8c")
# Plot actual track points
for track in tracks:
    fig.plot(data=track, color="green", style="c0.01c")
# Plot crossover point locations
fig.plot(x=df.x, y=df.y, color=df.h_X, cmap=True, style="c0.1c", pen="thinnest")
# Map frame in kilometre units
fig.basemap(
    frame=[
        "WSne",
        'xaf+l"Polar Stereographic X (km)"',
        'yaf+l"Polar Stereographic Y (km)"',
    ],
    region=(region + buffer) / 1000,
    projection="X8c",
)
fig.colorbar(position="JMR", frame=['x+l"Crossover Error"', "y+lm"])
fig.savefig("figures/crossover_area.png")
fig.show()


# %% [markdown]
# ### 1D plots of height changing over time
#
# Plot height change over time at:
#
# 1. One single crossover point location
# 2. Many crossover locations over an area

# %%
# Tidy up dataframe first using pd.wide_to_long
# I.e. convert 't_1', 't_2', 'h_1', 'h_2' columns into just 't' and 'h'.
df["id"] = df.index
df_th: pd.DataFrame = pd.wide_to_long(
    df=df[["id", "track1_track2", "x", "y", "t_1", "t_2", "h_1", "h_2"]],
    stubnames=["t", "h"],
    i="id",
    j="track",
    sep="_",
)
df_th = df_th.reset_index(level="track").drop_duplicates(ignore_index=True)

# %%
# 1D Plot at location with **maximum** absolute crossover height error (max_h_X)
df_max = df_th.query(expr="x == @max_h_X.x & y == @max_h_X.y").sort_values(by="t")
track1, track2 = df_max.track1_track2.iloc[0].split("_")
print(f"{round(max_h_X.h_X, 2)} metres height change at {max_h_X.x}, {max_h_X.y}")
t_min = (df_max.t.min() - pd.Timedelta(2, unit="W")).isoformat()
t_max = (df_max.t.max() + pd.Timedelta(2, unit="W")).isoformat()
h_min = df_max.h.min() - 0.2
h_max = df_max.h.max() + 0.4

fig = pygmt.Figure()
with pygmt.config(
    FONT_ANNOT_PRIMARY="9p", FORMAT_TIME_PRIMARY_MAP="abbreviated", FORMAT_DATE_MAP="o"
):
    fig.basemap(
        projection="X12c/8c",
        region=[t_min, t_max, h_min, h_max],
        frame=[
            "WSne",
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
fig.savefig(f"figures/crossover_{track1}_{track2}_{min_date}_{max_date}.png")
fig.show()

# %%
# 1D plots of a crossover area, all the height points over time
t_min = (df_th.t.min() - pd.Timedelta(1, unit="W")).isoformat()
t_max = (df_th.t.max() + pd.Timedelta(1, unit="W")).isoformat()
h_min = df_th.h.min() - 0.2
h_max = df_th.h.max() + 0.2

fig = pygmt.Figure()
with pygmt.config(
    FONT_ANNOT_PRIMARY="9p", FORMAT_TIME_PRIMARY_MAP="abbreviated", FORMAT_DATE_MAP="o"
):
    fig.basemap(
        projection="X12c/12c",
        region=[t_min, t_max, h_min, h_max],
        frame=[
            "WSne",
            "pxa1Of1o+lDate",  # primary time axis, 1 mOnth annotation and minor axis
            "sx1Y",  # secondary time axis, 1 Year intervals
            'yaf+l"Elevation at crossover (m)"',
        ],
    )

crossovers = df_th.groupby(by=["x", "y"])
pygmt.makecpt(cmap="categorical", series=[1, len(crossovers) + 1, 1])
for i, ((x_coord, y_coord), indexes) in enumerate(crossovers.indices.items()):
    df_ = df_th.loc[indexes].sort_values(by="t")
    # if df_.h.max() - df_.h.min() > 1.0:  # plot only > 1 metre height change
    track1, track2 = df_.track1_track2.iloc[0].split("_")
    label = f'"Track {track1} {track2}"'
    fig.plot(x=df_.t, y=df_.h, Z=i, style="c0.1c", cmap=True, pen="thin+z", label=label)
    # Plot line connecting points
    fig.plot(
        x=df_.t, y=df_.h, Z=i, pen=f"faint,+z,-", cmap=True
    )  # , label=f'"+g-1l+s0.15c"')
fig.legend(position="JMR+JMR+o0.2c", box="+gwhite+p1p")
fig.savefig(f"figures/crossover_many_{min_date}_{max_date}.png")
fig.show()

# %%
