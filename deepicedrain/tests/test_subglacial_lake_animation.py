"""
Feature tests for animating Active Subglacial Lakes in Antactica.
"""

import os
import subprocess

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import pygmt
import tqdm
import xarray as xr
from pytest_bdd import given, scenario, then, when

import deepicedrain


@scenario(
    feature_name="features/subglacial_lakes.feature",
    scenario_name="Subglacial Lake Animation",
    example_converters=dict(
        lake_name=str, lake_id=int, cycles=str, azimuth=float, elevation=float
    ),
)
def test_subglacial_lake_animation():
    """
    Generate an animated time-series visualization for an active subglacial lake
    """
    pass


@given(
    "some altimetry data at <location> spatially subsetted to <lake_name> with <lake_id>",
    target_fixture="df_lake",
)
def lake_altimetry_data(
    location: str, lake_name: str, lake_id: int, context
) -> pd.DataFrame:
    """
    Load up some pre-processed ICESat-2 ATL11 altimetry data from a Parquet
    file and subset it to a specific lake region.
    """
    # TODO use intake_parquet after https://github.com/intake/intake-parquet/issues/18
    with fsspec.open(
        f"simplecache::https://github.com/weiji14/deepicedrain/releases/download/v0.3.1/df_dhdt_{location}.parquet",
        simplecache=dict(cache_storage="ATLXI", same_names=True),
    ) as openfile:
        dataframe: pd.DataFrame = pd.read_parquet(openfile)

    antarctic_lakes: gpd.GeoDataFrame = deepicedrain.catalog.subglacial_lakes.read()

    context.lake_name: str = lake_name
    context.placename: str = context.lake_name.lower().replace(" ", "_")

    context.lake: pd.Series = antarctic_lakes.loc[lake_id]
    context.draining = True if context.lake.inner_dhdt < 0 else False

    context.region = deepicedrain.Region.from_gdf(
        gdf=context.lake, name=context.lake_name
    )
    df_lake: pd.DataFrame = context.region.subset(data=dataframe)

    return df_lake


@when("it is turned into a spatiotemporal cube over ICESat-2 cycles <cycles>")
def create_spatiotemporal_cube(
    df_lake: pd.DataFrame, cycles: str, context
) -> xr.Dataset:
    """
    Generate gridded time-series of ice elevation over lake.
    """
    os.makedirs(name=f"figures/{context.placename}", exist_ok=True)

    start, end = cycles.split("-")  # E.g. "3-8"
    context.cycles = tuple(range(int(start), int(end) + 1))  # ICESat-2 cycles
    context.ds_lake: xr.Dataset = deepicedrain.spatiotemporal_cube(
        table=df_lake,
        placename=context.placename,
        cycles=context.cycles,
        folder=f"figures/{context.placename}",
    )
    context.ds_lake.to_netcdf(
        path=f"figures/{context.placename}/xyht_{context.placename}.nc", mode="w"
    )

    return context.ds_lake


@when("visualized at each cycle using a 3D perspective at <azimuth> and <elevation>")
def visualize_grid_in_3D(
    df_lake: pd.DataFrame, azimuth: float, elevation: float, context
):
    """
    Create 3D plots of gridded ice surface elevation over time.
    """
    # Get 3D grid_region (xmin/xmax/ymin/ymax/zmin/zmax),
    # and calculate normalized z-values as Elevation delta relative to Cycle 3
    z_limits: tuple = (
        float(context.ds_lake.z.min()),
        float(context.ds_lake.z.max()),
    )  # original z limits
    grid_region: tuple = context.region.bounds() + z_limits

    ds_lake_diff: xr.Dataset = (
        context.ds_lake - context.ds_lake.sel(cycle_number=context.cycles[0]).z
    )
    z_diff_limits: tuple = (float(ds_lake_diff.z.min()), float(ds_lake_diff.z.max()))
    diff_grid_region: np.ndarray = np.append(arr=grid_region[:4], values=z_diff_limits)

    print(f"Elevation limits are: {z_limits}")

    # 3D plots of gridded ice surface elevation over time
    for cycle in tqdm.tqdm(iterable=context.cycles):
        time_nsec: pd.Timestamp = df_lake[f"utc_time_{cycle}"].mean()
        time_sec: str = np.datetime_as_string(arr=time_nsec.to_datetime64(), unit="s")

        grid = (
            f"figures/{context.placename}/h_corr_{context.placename}_cycle_{cycle}.nc"
        )
        points = pd.DataFrame(
            data=np.vstack(context.lake.geometry.boundary.coords.xy).T,
            columns=("x", "y"),
        )
        outline_points = pygmt.grdtrack(points=points, grid=grid, newcolname="z")
        outline_points["z"] = outline_points.z.fillna(value=outline_points.z.median())

        fig = deepicedrain.plot_icesurface(
            grid=grid,  # ds_lake.sel(cycle_number=cycle).z
            grid_region=grid_region,
            diff_grid=ds_lake_diff.sel(cycle_number=cycle).z,
            diff_grid_region=diff_grid_region,
            track_points=df_lake[["x", "y", f"h_corr_{cycle}"]].dropna().to_numpy(),
            outline_points=outline_points,
            azimuth=157.5,  # 202.5  # 270
            elevation=45,  # 60
            title=context.region.name,
            subtitle=f"Cycle {cycle} at {time_sec}",
        )
        fig.savefig(
            f"figures/{context.placename}/dsm_{context.placename}_cycle_{cycle}.png"
        )

    return fig


@then("the result is an animation of ice surface elevation changing over time")
def animated_dsm_gif(context) -> str:
    """
    Make an animated GIF of changing ice surface from the PNG files.
    """
    gif_fname: str = os.path.join(
        "figures",
        f"{context.placename}",
        f"dsm_{context.placename}_cycles_{context.cycles[0]}-{context.cycles[-1]}.gif",
    )
    subprocess.call(
        [
            "convert",
            "-delay",
            "120",
            "-loop",
            "0",
            f"figures/{context.placename}/dsm_*.png",
            gif_fname,
        ]
    )
    assert os.path.exists(path=gif_fname)

    return gif_fname
