"""
Tests various conversions between geospatial and temporal units
"""
import datetime

import dask
import numpy as np
import numpy.testing as npt
import pandas as pd
import xarray as xr

from deepicedrain import catalog, deltatime_to_utctime


def test_deltatime_to_utctime():
    """
    Test that converting from ICESat-2 delta_time to utc_time works,
    and that the xarray dimensions are preserved in the process.
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

    atl11_dataset.close()
