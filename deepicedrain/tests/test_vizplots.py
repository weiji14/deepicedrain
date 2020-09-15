"""
Tests that various visualizations can be made to appear!
"""
import os
import tempfile

import pandas as pd
import pytest
import xarray as xr
import pygmt.helpers.testing

from deepicedrain import (
    catalog,
    deltatime_to_utctime,
    ndarray_to_parquet,
    plot_alongtrack,
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
    return dataframe


@pygmt.helpers.testing.check_figures_equal()
def test_plot_alongtrack(dataframe):
    """
    Tests that a 2D along track plot figure can be produced. Also make sure that
    the default for oldtonew is True (i.e. legend shows Cycle 1 before Cycle 2).
    """
    kwargs = dict(
        df=dataframe,
        rgtpair="788_pt2",
        regionname="Greenland",
        x_var="longitude",
        spacing="0.1/5",
    )
    fig_ref = plot_alongtrack(**kwargs, oldtonew=True)
    fig_test = plot_alongtrack(**kwargs)

    return fig_ref, fig_test
