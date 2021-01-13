"""
Feature tests to calculate ice volume displacement of Active Subglacial Lakes
in Antactica.
"""
import itertools

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import pint
import pint_pandas
import pygmt
import shapely
import tqdm
import uncertainties
from pytest_bdd import given, scenario, then, when

import deepicedrain

ureg = pint.UnitRegistry()
pint_pandas.PintType.ureg = ureg


@scenario(
    feature_name="features/subglacial_lakes.feature",
    scenario_name="Subglacial Lake Crossover Anomalies",
    example_converters=dict(lake_name=str, location=str),
)
def test_subglacial_lake_anomalies():
    """
    Calaculate and visualize of ice surface height anomalies at crossover
    points for an active subglacial lake.
    """
    pass


@when("ice surface height anomalies are calculated at crossovers within the lake area")
def crossover_height_anomaly(
    df_lake: pd.DataFrame, client: dask.distributed.Client, context
) -> pd.DataFrame:

    # Subset data to lake of interest
    gdf_lake = gpd.GeoDataFrame(
        df_lake, geometry=gpd.points_from_xy(x=df_lake.x, y=df_lake.y, crs=3031)
    )
    df_lake: pd.DataFrame = df_lake.loc[
        gdf_lake.within(context.lake.geometry)
    ]  # polygon subset

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

    crossovers: dict = {}
    for f in tqdm.tqdm(
        iterable=dask.distributed.as_completed(futures=futures), total=len(futures)
    ):
        if f.status != "error":  # skip those track pairs which don't intersect
            crossovers[f.key] = f.result().dropna().reset_index(drop=True)

    df_cross: pd.DataFrame = pd.concat(objs=crossovers, names=["track1_track2", "id"])
    df: pd.DataFrame = df_cross.reset_index(level="track1_track2").reset_index(
        drop=True
    )

    # Calculate crossover error
    df["h_X"]: pd.Series = df.h_2 - df.h_1  # crossover error (i.e. height difference)
    df["t_D"]: pd.Series = df.t_2 - df.t_1  # elapsed time in ns (i.e. time difference)
    ns_in_yr: int = 365.25 * 24 * 60 * 60 * 1_000_000_000  # nanoseconds in a year
    df["dhdt"]: pd.Series = df.h_X / (df.t_D.astype(np.int64) / ns_in_yr)

    # Get some summary statistics of our crossover errors
    sumstats: pd.DataFrame = df[["h_X", "t_D", "dhdt"]].describe()
    # Find location with highest absolute crossover error, and most sudden height change
    max_h_X: pd.Series = df.iloc[np.nanargmax(df.h_X.abs())]  # highest crossover error
    max_dhdt: pd.Series = df.iloc[df.dhdt.argmax()]  # most sudden change in height

    # Tidy up dataframe first using pd.wide_to_long
    # I.e. convert 't_1', 't_2', 'h_1', 'h_2' columns into just 't' and 'h'.
    df_th: pd.DataFrame = deepicedrain.wide_to_long(
        df=df.loc[:, ["track1_track2", "x", "y", "t_1", "t_2", "h_1", "h_2"]],
        stubnames=["t", "h"],
        j="track",
    )
    df_th: pd.DataFrame = df_th.drop_duplicates(ignore_index=True)
    df_th: pd.DataFrame = df_th.sort_values(by="t").reset_index(drop=True)

    # Calculate height anomaly at crossover point as
    # height at t=n minus height at t=0 (first observation date at crossover point)
    anomfunc = lambda h: h - h.iloc[0]  # lambda h: h - h.mean()
    df_th["h_anom"] = df_th.groupby(by="track1_track2").h.transform(func=anomfunc)
    # Calculate ice volume displacement (dvol) in unit metres^3
    # and rolling mean height anomaly (h_roll) in unit metres
    surface_area: pint.Quantity = context.lake.geometry.area * ureg.metre ** 2
    ice_dvol: pd.Series = deepicedrain.ice_volume_over_time(
        df_elev=df_th.astype(dtype={"h_anom": "pint[metre]"}),
        surface_area=surface_area,
        time_col="t",
        outfile=f"figures/{context.placename}/ice_dvol_dt_{context.placename}.txt",
    )
    df_th["h_roll"]: pd.Series = uncertainties.unumpy.nominal_values(
        arr=ice_dvol.pint.magnitude / surface_area.magnitude
    )

    context.df_th = df_th
    return context.df_th


@then("we see a trend of active subglacial lake surfaces changing over time")
def crossover_height_anomaly_trend(context) -> str:
    fig = deepicedrain.plot_crossovers(
        df=context.df_th,
        regionname=context.region.name,
        elev_var="h_anom",
        outline_points=f"figures/{context.placename}/{context.placename}.gmt",
    )
    fig.plot(
        x=context.df_th.t, y=context.df_th.h_roll, pen="thick,-"
    )  # plot rolling mean height anomaly

    min_date, max_date = ("2018-10-14", "2020-09-30")
    fname: str = f"figures/{context.placename}/crossover_anomaly_{context.placename}_{min_date}_{max_date}.png"
    fig.savefig(fname=fname)

    assert context.df_th.h_roll.max() > 0.1

    return fname
