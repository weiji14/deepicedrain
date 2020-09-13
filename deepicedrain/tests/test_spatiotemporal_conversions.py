"""
Tests various conversions between geospatial and temporal units
"""
import datetime

import numpy as np
import numpy.testing as npt
import pandas as pd
import xarray as xr

import dask
from deepicedrain import catalog, deltatime_to_utctime, lonlat_to_xy


def test_deltatime_to_utctime_numpy_timedelta64():
    """
    Test that converting from ICESat-2 delta_time to utc_time works on a
    single numpy.timedelta object.
    """
    delta_time = np.timedelta64(24731275413287379, "ns")
    utc_time: np.datetime64 = deltatime_to_utctime(dataarray=delta_time)

    npt.assert_equal(
        actual=utc_time, desired=np.datetime64("2018-10-14T05:47:55.413287379")
    )


def test_deltatime_to_utctime_pandas_series():
    """
    Test that converting from ICESat-2 delta_time to utc_time works on a
    dask.dataframe.core.Series.
    """
    atl11_dataset: xr.Dataset = catalog.test_data.atl11_test_case.to_dask()
    atl11_dataframe: pd.DataFrame = atl11_dataset.to_dataframe()

    utc_time: pd.Series = deltatime_to_utctime(dataarray=atl11_dataframe.delta_time)

    assert utc_time.shape == (2808,)

    npt.assert_equal(
        actual=utc_time.min(), desired=pd.Timestamp("2019-05-19T20:53:51.039891534")
    )

    npt.assert_equal(
        actual=utc_time.loc[3].mean(),
        desired=pd.Timestamp("2019-05-19 20:54:00.925868800"),
    )
    npt.assert_equal(
        actual=utc_time.loc[4].mean(),
        desired=pd.Timestamp("2019-08-18 16:33:47.791226368"),
    )
    npt.assert_equal(
        actual=utc_time.max(), desired=pd.Timestamp("2019-08-18T16:33:57.834610209")
    )


def test_deltatime_to_utctime_xarray_dataarray():
    """
    Test that converting from ICESat-2 delta_time to utc_time works on an
    xarray.DataArray, and that the dimensions are preserved in the process.
    """
    atl11_dataset: xr.Dataset = catalog.test_data.atl11_test_case.to_dask()

    utc_time: xr.DataArray = deltatime_to_utctime(dataarray=atl11_dataset.delta_time)

    assert utc_time.shape == (1404, 2)
    assert utc_time.dims == ("ref_pt", "cycle_number")
    assert dask.is_dask_collection(utc_time)

    utc_time = utc_time.compute()

    npt.assert_equal(
        actual=utc_time.data.min(),
        desired=np.datetime64("2019-05-19T20:53:51.039891534"),
    )
    npt.assert_equal(
        actual=np.datetime64(pd.DataFrame(utc_time.data)[0].mean()),
        desired=np.datetime64("2019-05-19 20:54:00.925868"),
    )
    npt.assert_equal(
        actual=np.datetime64(pd.DataFrame(utc_time.data)[1].mean()),
        desired=np.datetime64("2019-08-18 16:33:47.791226"),
    )
    npt.assert_equal(
        actual=utc_time.data.max(),
        desired=np.datetime64("2019-08-18T16:33:57.834610209"),
    )


def test_lonlat_to_xy_dask_series():
    """
    Test that converting from longitude/latitude to x/y in EPSG:3031 works when
    passing them in as dask.dataframe.core.Series objects.
    """
    atl11_dataset: xr.Dataset = catalog.test_data.atl11_test_case.to_dask()
    atl11_dataframe: dask.dataframe.core.DataFrame = atl11_dataset.to_dask_dataframe()

    x, y = lonlat_to_xy(
        longitude=atl11_dataframe.longitude, latitude=atl11_dataframe.latitude
    )
    npt.assert_equal(actual=x.mean(), desired=-56900105.00307033)
    npt.assert_equal(actual=y.mean(), desired=48141607.48486084)


def test_lonlat_to_xy_xarray_dataarray():
    """
    Test that converting from longitude/latitude to x/y in EPSG:3031 works when
    passing them in as xarray.DataArray objects. Ensure that the xarray
    dimensions are preserved in the process.
    """
    atl11_dataset: xr.Dataset = catalog.test_data.atl11_test_case.to_dask()

    x, y = lonlat_to_xy(
        longitude=atl11_dataset.longitude, latitude=atl11_dataset.latitude
    )

    assert x.dims == y.dims == ("ref_pt",)
    assert x.shape == y.shape == (1404,)
    npt.assert_equal(actual=x.mean().data, desired=-56900105.00307034)
    npt.assert_equal(actual=y.mean().data, desired=48141607.48486084)
