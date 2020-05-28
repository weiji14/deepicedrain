"""
Tests the calculate_delta function
"""
import intake
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from deepicedrain import calculate_delta


@pytest.fixture(scope="module")
def atl11_dataset():
    """
    Loads a sample ATL11 test dataset from the intake catalog
    """
    catalog: intake.catalog.local.YAMLFileCatalog = intake.open_catalog(
        uri="tests/test_catalog.yaml"
    )
    atl11_dataset: xr.Dataset = catalog.atl11_test_case.read()

    return atl11_dataset


def test_calculate_delta_height(atl11_dataset):
    """
    Check that calculating change in elevation works.
    """
    delta_height = calculate_delta(
        dataset=atl11_dataset, oldcyclenum=3, newcyclenum=4, variable="h_corr"
    )

    assert isinstance(delta_height, xr.DataArray)
    assert delta_height.shape == (1404,)
    npt.assert_allclose(actual=delta_height.min().data, desired=-3.10612352)
    npt.assert_allclose(actual=delta_height.mean().data, desired=-0.90124122)
    npt.assert_allclose(actual=delta_height.max().data, desired=9.49908442)


def test_calculate_delta_time(atl11_dataset):
    """
    Check that calculating change in time works.
    """
    delta_time = calculate_delta(
        dataset=atl11_dataset, oldcyclenum=3, newcyclenum=4, variable="delta_time"
    )

    assert isinstance(delta_time, xr.DataArray)
    assert delta_time.shape == (1404,)
    npt.assert_equal(
        actual=delta_time.min().data, desired=np.timedelta64(7846786703322903)
    )
    npt.assert_equal(
        actual=delta_time.mean().data, desired=np.timedelta64(7846786865357197)
    ),
    npt.assert_equal(
        actual=delta_time.max().data, desired=np.timedelta64(7846787022726588)
    )
