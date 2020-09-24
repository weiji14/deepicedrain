"""
Tests behaviour of the Region class's bounding box based functionality
"""
import dask.dataframe
import geopandas as gpd
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import xarray as xr

from deepicedrain import Region, catalog, lonlat_to_xy


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
    region = Region("Whillans Ice Stream", -500000, -400000, -600000, -500000)
    with pytest.raises(NotImplementedError):
        print(region.bounds(style="ltrb"))


def test_region_datashade():
    """
    Tests that we can datashade a pandas.DataFrame based on the region's bounds
    """
    region = Region("Kitaa, Greenland", -1_600_000, -1_520_000, -1_360_000, -1_300_000)

    atl11_dataset: xr.Dataset = catalog.test_data.atl11_test_case.to_dask()
    atl11_dataset["x"], atl11_dataset["y"] = lonlat_to_xy(
        longitude=atl11_dataset.longitude, latitude=atl11_dataset.latitude, epsg=3995
    )
    atl11_dataset = atl11_dataset.set_coords(["x", "y"])
    df: pd.DataFrame = atl11_dataset.h_corr.to_dataframe()

    agg_grid: xr.DataArray = region.datashade(df=df, z_dim="h_corr", plot_width=100)

    assert agg_grid.shape == (75, 100)  # check correct aspect ratio is maintained
    npt.assert_allclose(agg_grid.min(), 1426.336637)
    npt.assert_allclose(agg_grid.mean(), 1668.94741)
    npt.assert_allclose(agg_grid.max(), 1798.066285)


def test_region_from_geodataframe():
    """
    Test that we can create a deepicedrain.Region object from a single
    geodataframe row.
    """
    geodataframe: gpd.GeoDataFrame = gpd.read_file(
        filename="deepicedrain/deepicedrain_regions.geojson"
    ).iloc[1]
    region = Region.from_gdf(gdf=geodataframe, name_col="fullname", spacing=1000)

    assert region.name == "Kamb Ice Stream"
    assert region.bounds() == (-412000.0, -365000.0, -740000.0, -699000.0)


def test_region_subset_xarray_dataset():
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
    ds_subset = region.subset(data=dataset)
    assert isinstance(ds_subset, xr.Dataset)
    assert ds_subset.h_corr.shape == (24, 30)


@pytest.mark.parametrize("dataframe_type", [pd.DataFrame, dask.dataframe.DataFrame])
def test_region_subset_dataframe(dataframe_type):
    """
    Test that we can subset a pandas or dask DataFrame based on the region's
    bounds
    """
    region = Region("South Pole", -100, 100, -100, 100)
    dataframe = pd.DataFrame(
        data={
            "x": np.linspace(start=-200, stop=200, num=50),
            "y": np.linspace(start=-160, stop=160, num=50),
            "dhdt": np.random.rand(50),
        }
    )
    if dataframe_type == dask.dataframe.core.DataFrame:
        dataframe = dask.dataframe.from_pandas(data=dataframe, npartitions=2)
    df_subset = region.subset(data=dataframe)
    assert isinstance(df_subset, dataframe_type)
    assert len(df_subset.dhdt) == 24
