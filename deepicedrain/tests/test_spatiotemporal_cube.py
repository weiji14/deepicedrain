"""
Tests creation of spatiotemporal NetCDF data cubes
"""
import os
import tempfile

import pandas as pd
import pytest
import xarray as xr

from deepicedrain import catalog, ndarray_to_parquet, spatiotemporal_cube


@pytest.fixture(scope="module", name="table")
def fixture_table():
    """
    Loads the sample ICESat-2 ATL11 data, and processes it into an suitable
    pandas.DataFrame table format.
    """
    dataset: xr.Dataset = catalog.test_data.atl11_test_case.to_dask()
    with tempfile.TemporaryDirectory() as tmpdir:
        parquetpath: str = os.path.join(tmpdir, "temp.parquet")
        table: pd.DataFrame = ndarray_to_parquet(
            ndarray=dataset,
            parquetpath=parquetpath,
            variables=["longitude", "latitude", "h_corr"],
        )
    return table


def test_spatiotemporal_cube(table):
    """
    Tests that creating a spatiotemporal NetCDF data cube works
    """
    grid: xr.Dataset = spatiotemporal_cube(
        table=table,
        placename="greenland",
        x_var="longitude",
        y_var="latitude",
        spacing=0.1,
        folder=tempfile.gettempdir(),
    )

    assert isinstance(grid, xr.Dataset)
    assert grid.dims == {"x": 8, "y": 16, "cycle_number": 2, "grid_mapping": 12}
    xr.testing.assert_allclose(a=grid.z.min(), b=xr.DataArray(data=1435.1884))
    xr.testing.assert_allclose(a=grid.z.max(), b=xr.DataArray(data=1972.5968))
    xr.testing.assert_allclose(
        a=grid.z.median(axis=(1, 2)),
        b=xr.DataArray(
            data=[1655.0094, 1654.4307], coords=[[1, 2]], dims=["cycle_number"]
        ),
    )
    assert "-Gh_corr_greenland_cycle_2.nc" in grid.history

    paths = [
        os.path.join(tempfile.gettempdir(), f"h_corr_greenland_cycle_{cycle}.nc")
        for cycle in (1, 2)
    ]
    assert all(os.path.exists(path=path) for path in paths)
    [os.remove(path=path) for path in paths]
