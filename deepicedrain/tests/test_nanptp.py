"""
Tests the nanptp function
"""
import numpy as np
import numpy.testing as npt
import xarray as xr

from deepicedrain import nanptp, catalog


def test_nanptp():
    """
    Check that calculating absolute height range across cycles work.
    """
    atl11_dataset: xr.Dataset = catalog.test_data.atl11_test_case.to_dask()

    height_range: np.ndarray = nanptp(a=atl11_dataset.h_corr, axis=1)

    assert isinstance(height_range, np.ndarray)
    assert height_range.shape == (1404,)
    npt.assert_allclose(actual=height_range.min(), desired=0.07718418)
    npt.assert_allclose(actual=height_range.mean(), desired=0.9836243)
    npt.assert_allclose(actual=height_range.max(), desired=9.49908442)


def test_nanptp_with_nan():
    """
    Check that calculating point to point range works even with NaN values.
    """
    a = [[123, 231, np.NaN, 312, 213]]

    height_range: np.ndarray = nanptp(a=a, axis=1)

    assert isinstance(height_range, np.ndarray)
    npt.assert_equal(actual=height_range, desired=189)
