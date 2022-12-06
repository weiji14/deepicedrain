"""
Tests the dhdt_maxslp function.
"""
import numpy as np
import numpy.testing as npt
import xarray as xr

from deepicedrain import dhdt_maxslp, catalog


def test_dhdt_maxslp():
    """
    Check that performing dhdt_maxslp on height over time works across cycles
    for one sample data point.
    """
    atl11_dataset: xr.Dataset = catalog.test_data.atl11_test_case.to_dask()

    dhdt_maxslp_result: np.ndarray = dhdt_maxslp(
        x=atl11_dataset.delta_time.astype(np.uint64)[0], y=atl11_dataset.h_corr[0]
    )

    npt.assert_allclose(actual=dhdt_maxslp_result, desired=-5.037294e-17)


def test_dhdt_maxslp_with_nan():
    """
    Check that performing dhdt_maxslp works even with NaN values.
    """
    x = np.array([100, 200, np.NaN, 400, 500])
    y = np.array([-20, -35, np.NaN, -25, -30])

    dhdt_maxslp_result: np.ndarray = dhdt_maxslp(x=x, y=y)

    npt.assert_allclose(actual=dhdt_maxslp_result, desired=-0.15)
