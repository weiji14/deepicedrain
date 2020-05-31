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
