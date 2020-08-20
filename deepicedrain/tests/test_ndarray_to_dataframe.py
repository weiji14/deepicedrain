"""
Tests various conversions from n-dimensional arrays to columnar dataframe table
structures.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import dask
import zarr
from deepicedrain import array_to_dataframe, catalog, ndarray_to_parquet


@pytest.fixture(scope="module", name="dataset")
def fixture_dataset():
    """
    Load the sample ICESat-2 ATL11 data into an xarray, and clean it up to
    allow saving to other formats like Zarr
    """
    dataset: xr.Dataset = catalog.test_data.atl11_test_case.to_dask()
    for key, variable in dataset.variables.items():
        assert isinstance(dataset[key].DIMENSION_LABELS, np.ndarray)
        dataset[key].attrs["DIMENSION_LABELS"] = (
            dataset[key].attrs["DIMENSION_LABELS"].astype(str)
        )

    return dataset


@pytest.mark.parametrize("shape", [(10, 1), (10, 2)])
def test_numpy_array_to_pandas_dataframe(shape):
    """
    Test converting from a numpy.array to a pandas.Dataframe, and ensure that
    the colname argument works.
    """
    array: np.ndarray = np.ones(shape=shape)
    dataframe = array_to_dataframe(array=array)

    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe.columns) == shape[1]
    assert dataframe.columns.to_list() == [str(i) for i in range(shape[1])]


@pytest.mark.parametrize("shape", [(10, 1), (10, 2)])
def test_dask_array_to_dask_dataframe(shape):
    """
    Test converting from a dask.array to a dask.dataframe, and ensure that the
    startcol argument works.
    """
    array: dask.array.core.Array = dask.array.ones(shape=shape, name="varname")
    dataframe = array_to_dataframe(array=array, startcol=1)

    assert isinstance(dataframe, dask.dataframe.core.DataFrame)
    assert len(dataframe.columns) == shape[1]
    assert dataframe.columns.to_list() == [f"varname_{i+1}" for i in range(shape[1])]


def test_xarray_dataset_to_parquet_table(dataset):
    """
    Test converting from an xarray Dataset to a parquet table, specifying a
    list of variables to store and setting 'snappy' compression.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        parquetpath: str = os.path.join(tmpdir, "temp.parquet")
        ndarray_to_parquet(
            ndarray=dataset,
            parquetpath=parquetpath,
            variables=["longitude", "latitude", "h_corr", "h_corr_sigma"],
            compression="snappy",
        )

        df: dask.dataframe.core.DataFrame = dask.dataframe.read_parquet(
            path=parquetpath
        )
        assert len(df) == 1404
        assert list(df.columns) == [
            "longitude",
            "latitude",
            "h_corr_1",
            "h_corr_2",
            "h_corr_sigma_1",
            "h_corr_sigma_2",
        ]
        assert all(np.issubdtype(dtype, np.float64) for dtype in df.dtypes)


def test_zarr_array_to_parquet_table(dataset):
    """
    Test converting from a zarr array to a parquet table, specifying a list of
    variables to store and setting 'snappy' compression.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        zarrstore: str = os.path.join(tmpdir, "temp.zarr")
        dataset.to_zarr(store=zarrstore, consolidated=True)
        zarrarray: zarr.hierarchy.Group = zarr.open_consolidated(store=zarrstore)

        parquetpath: str = os.path.join(tmpdir, "temp.parquet")
        ndarray_to_parquet(
            ndarray=zarrarray,
            parquetpath=parquetpath,
            variables=["longitude", "latitude", "h_corr", "delta_time"],
            compression="snappy",
        )

        df: dask.dataframe.core.DataFrame = dask.dataframe.read_parquet(
            path=parquetpath
        )
        assert len(df) == 1404
        assert list(df.columns) == [
            "longitude",
            "latitude",
            "h_corr_1",
            "h_corr_2",
            "delta_time_1",
            "delta_time_2",
        ]
        assert all(np.issubdtype(dtype, np.float64) for dtype in df.dtypes)
