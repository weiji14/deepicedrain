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
    # Data files are version controlled using DVC and stored on
    # https://dagshub.com/weiji14/deepicedrain/src/main/ATLXI
    # They will also be uploaded as assets every release at e.g.
    # https://github.com/weiji14/deepicedrain/releases
    dataframe: pd.DataFrame = pd.read_parquet(path=f"ATLXI/df_dhdt_{location}.parquet")

    # Get lake outline from intake catalog
    lake_catalog = deepicedrain.catalog.subglacial_lakes()
    lake_ids, transect_id = (
        pd.json_normalize(lake_catalog.metadata["lakedict"])
        .query("lakename == @lake_name")[["ids", "transect"]]
        .iloc[0]
    )
    context.transect_id: str = transect_id
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
    # Get all raw xyz points and one transect line dataframe
    track_dict: dict = deepicedrain.split_tracks(df=df_lake)
    context.track_points: pd.DataFrame = (
        pd.concat(track_dict.values())
        .groupby(by=["x", "y"])
        .mean()  # z value is mean h_corr over all cycles
        .reset_index()[["x", "y", "h_corr"]]
    )
    try:
        _rgt, _pt = transect_id.split("_")
        context.df_transect: pd.DataFrame = (
            track_dict[transect_id][["x", "y", "h_corr", "cycle_number"]]
            .groupby(by=["x", "y"])
            .max()  # z value is maximum h_corr over all cycles
            .reset_index()
        )
    except AttributeError:
        pass

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
