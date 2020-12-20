"""
This module contains shared fixtures, steps, and hooks.
"""
import os

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pytest_bdd import given

import deepicedrain


@pytest.fixture
def context():
    """
    An empty context class, used for passing arbitrary objects between tests.
    """

    class Context:
        pass

    return Context()


@pytest.fixture(scope="session", name="client")
def fixture_client():
    """
    A dask distributed client to throw compute tasks at!
    """
    import dask

    tag: str = "X2SYS"
    os.environ["X2SYS_HOME"] = os.path.abspath(tag)
    client = dask.distributed.Client(
        n_workers=8, threads_per_worker=1, env={"X2SYS_HOME": os.environ["X2SYS_HOME"]}
    )
    yield client

    client.shutdown()


@given("some altimetry data over <lake_name> at <location>", target_fixture="df_lake")
def lake_altimetry_data(lake_name: str, location: str, context) -> pd.DataFrame:
    """
    Load up some pre-processed ICESat-2 ATL11 altimetry data from a Parquet
    file and subset it to a specific lake region.
    """
    context.lake_name: str = lake_name
    # TODO use intake_parquet after https://github.com/intake/intake-parquet/issues/18
    with fsspec.open(
        f"simplecache::https://github.com/weiji14/deepicedrain/releases/download/v0.3.1/df_dhdt_{location}.parquet",
        simplecache=dict(cache_storage="ATLXI", same_names=True),
    ) as openfile:
        dataframe: pd.DataFrame = pd.read_parquet(openfile)

    # Get lake outline from intake catalog
    lake_catalog = deepicedrain.catalog.subglacial_lakes()
    lake_ids: list = (
        pd.json_normalize(lake_catalog.metadata["lakedict"])
        .query("lakename == @lake_name")
        .ids.iloc[0]
    )
    context.lake: pd.Series = (
        lake_catalog.read()
        .loc[lake_ids]
        .dissolve(by=np.zeros(shape=len(lake_ids), dtype="int64"), as_index=False)
        .squeeze()
    )

    # Contextual metadata on lake's bounding box region, and its drain/fill state
    context.region = deepicedrain.Region.from_gdf(
        gdf=context.lake, name=context.lake_name
    )
    context.draining: bool = context.lake.inner_dhdt < 0

    # Subset data to lake of interest
    context.placename: str = context.lake_name.lower().replace(" ", "_")
    df_lake: cudf.DataFrame = context.region.subset(data=dataframe)  # bbox subset
    gdf_lake = gpd.GeoDataFrame(
        df_lake, geometry=gpd.points_from_xy(x=df_lake.x, y=df_lake.y, crs=3031)
    )
    df_lake: pd.DataFrame = df_lake.loc[
        gdf_lake.within(context.lake.geometry)
    ]  # polygon subset

    # Save lake outline to OGR GMT file format
    os.makedirs(name=f"figures/{context.placename}", exist_ok=True)
    context.outline_points: str = f"figures/{context.placename}/{context.placename}.gmt"
    try:
        os.remove(path=context.outline_points)
    except FileNotFoundError:
        pass
    lake_catalog.read().loc[list(lake_ids)].to_file(
        filename=context.outline_points, driver="OGR_GMT", mode="w"
    )

    return df_lake
