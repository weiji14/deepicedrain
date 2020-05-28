"""
Tests behaviour of the Region class
"""
import numpy as np
import pytest
import xarray as xr

from deepicedrain import Region


def test_region_scale():
    """
    Tests that a map scale is output based on the region.
    """
    region = Region("Antarctica", -2700000, 2800000, -2200000, 2300000)
    assert region.scale == 27500000


def test_region_bounds_lrbt():
    """
    Tests that PyGMT style bounds are given (by default).
    """
    region = Region("Siple Coast", -1000000, 250000, -1000000, -100000)
    assert region.bounds() == (-1000000, 250000, -1000000, -100000)


def test_region_bounds_lbrt():
    """
    Tests that Shapely style bounds are given
    """
    region = Region("Whillans Ice Stream", -350000, -100000, -700000, -450000)
    assert region.bounds(style="lbrt") == (-350000, -700000, -100000, -450000)


def test_region_bounds_ltrb():
    """
    Tests that error is raised when passing in a style that is not implemented.
    """
    region = Region("Kamb Ice Stream", -500000, -400000, -600000, -500000)
    with pytest.raises(NotImplementedError):
        print(region.bounds(style="ltrb"))


def test_region_subset():
    """
    Test that we can subset an xarray.Dataset based on the region's bounds
    """
    region = Region("South Pole", -100, 100, -100, 100)
    dataset = xr.Dataset(
        data_vars={"h_corr": (["x", "y"], np.random.rand(50, 50))},
        coords={
            "x": np.linspace(start=-200, stop=200, num=50),
            "y": np.linspace(start=-160, stop=160, num=50),
        },
    )
    ds_subset = region.subset(ds=dataset)
    assert isinstance(ds_subset, xr.Dataset)
    assert ds_subset.h_corr.shape == (24, 30)
