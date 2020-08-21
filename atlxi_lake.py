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
