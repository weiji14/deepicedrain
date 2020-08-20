"""
Tests GPU accelerated spatial algorithms
"""

import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry

from deepicedrain import point_in_polygon_gpu

cudf = pytest.importorskip(modname="cudf")


def test_point_in_polygon_gpu():
    """
    Tests that the Point in Polygon GPU algorithm works
    """
    points_df: cudf.DataFrame = cudf.DataFrame(
        data={
            "x": np.linspace(start=-200, stop=200, num=50),
            "y": np.linspace(start=-160, stop=160, num=50),
        }
    )
    polygon = {
        "placename": ["South Pole"],
        "geometry": shapely.geometry.box(minx=-5, maxx=5, miny=-5, maxy=5).buffer(100),
    }
    poly_df: gpd.GeoDataFrame = gpd.GeoDataFrame(polygon)

    point_labels = point_in_polygon_gpu(
        points_df=points_df, poly_df=poly_df, poly_label_col="placename"
    )
    assert isinstance(point_labels, cudf.Series)
    assert point_labels.count() == 20  # Count non-NaN labels
    assert list(point_labels.unique().to_pandas()) == [None, "South Pole"]
