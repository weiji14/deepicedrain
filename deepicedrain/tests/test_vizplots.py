"""
Tests that various visualizations can be made to appear!
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import pygmt.helpers.testing

from deepicedrain import (
    catalog,
    deltatime_to_utctime,
    ndarray_to_parquet,
    plot_alongtrack,
    plot_crossovers,
    wide_to_long,
)


@pytest.fixture(scope="module", name="dataframe")
def fixture_dataframe():
    """
    Loads the sample ICESat-2 ATL11 data, and processes it into an suitable
    pandas.DataFrame format.
    """
    dataset: xr.Dataset = catalog.test_data.atl11_test_case.to_dask()
    dataset["utc_time"] = deltatime_to_utctime(dataarray=dataset.delta_time)

    with tempfile.TemporaryDirectory() as tmpdir:
        df: pd.DataFrame = ndarray_to_parquet(
            ndarray=dataset,
            parquetpath=os.path.join(tmpdir, "temp.parquet"),
            variables=["longitude", "latitude", "h_corr", "utc_time"],
            use_deprecated_int96_timestamps=True,
        )
    dataframe: pd.DataFrame = wide_to_long(
        df=df, stubnames=["h_corr", "utc_time"], j="cycle_number"
    )
    dataframe: pd.DataFrame = dataframe.reset_index(drop=True)

    # Mock up a dummy track1_track2 column based on the cycle_number
    dataframe["track1_track2"] = np.where(
        dataframe["cycle_number"] == 1, "0111_pt1x0222_pt2", "0333pt3x0111_pt1"
    )
    return dataframe


@pygmt.helpers.testing.check_figures_equal()
def test_plot_alongtrack(dataframe):
    """
    Tests that a 2D along track plot figure can be produced. Also ensure that
    the default for oldtonew is True (i.e. legend shows Cycle 1 before Cycle 2).
    """
    kwargs = dict(
        df=dataframe,
        rgtpair="788_pt2",
        regionname="Greenland",
        xatc_var="longitude",
        spacing="0.1/5",
    )
    fig_ref = plot_alongtrack(**kwargs)
    fig_test = plot_alongtrack(**kwargs, oldtonew=True)

    return fig_ref, fig_test


@pygmt.helpers.testing.check_figures_equal()
def test_plot_crossovers(dataframe):
    """
    Tests that a crossover elevation plot figure can be produced. Also ensure
    that the default y axis spacing increment is 2.5.
    """

    kwargs = dict(
        df=dataframe, regionname="Greenland", time_var="utc_time", elev_var="h_corr"
    )
    fig_ref = plot_crossovers(**kwargs)
    fig_test = plot_crossovers(**kwargs, spacing=2.5)

    return fig_ref, fig_test
