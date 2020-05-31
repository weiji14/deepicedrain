"""
Tests the nan_linregress function
"""
import numpy as np
import numpy.testing as npt
import xarray as xr

from deepicedrain import nan_linregress, catalog


def test_nan_linregress():
    """
    Check that performing linear regression on height over time works
    across cycles for one sample data point.
    """
    atl11_dataset: xr.Dataset = catalog.test_data.atl11_test_case.to_dask()

    linregress_result: np.ndarray = nan_linregress(
        x=atl11_dataset.delta_time.astype(np.uint64)[0], y=atl11_dataset.h_corr[0]
    )

    slope, intercept, rvalue, pvalue, stderr = linregress_result
    npt.assert_allclose(actual=slope, desired=-5.037294e-17)
    npt.assert_allclose(actual=intercept, desired=1877.745452)
    npt.assert_allclose(actual=rvalue, desired=-1)
    npt.assert_allclose(actual=pvalue, desired=0)
    npt.assert_allclose(actual=stderr, desired=0)


def test_nan_linregress_with_nan():
    """
    Check that performing linear regression works even with NaN values.
    """
    x = np.array([100, 200, np.NaN, 400, 500])
    y = np.array([20, 35, np.NaN, 25, 30])

    linregress_result: np.ndarray = nan_linregress(x=x, y=y)

    npt.assert_allclose(
        actual=linregress_result,
        desired=[0.01, 24.5, 0.282842712, 0.717157288, 0.0239791576],
    )
