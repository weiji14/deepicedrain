"""
Geospatial and Temporal class that implements some handy tools.
Does bounding box region subsets, coordinate/time conversions, and more!
"""
import dataclasses
import datetime
import os
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

import datashader


@dataclasses.dataclass(frozen=True)
class Region:
    """
    A nice bounding box data class structure that holds the coordinates of its
    left, right, bottom and top extent, and features convenience functions for
    performing spatial subsetting and visualization based on those boundaries.
    """

    name: str  # name of region
    xmin: float  # left coordinate
    xmax: float  # right coordinate
    ymin: float  # bottom coordinate
    ymax: float  # top coordinate

    @classmethod
    def from_gdf(
        cls,
        gdf: gpd.GeoDataFrame,
        name_col: str = None,
        spacing: float = 1000.0,
        **kwargs,
    ):
        """
        Create a deepicedrain.Region instance from a geopandas GeoDataFrame
        (single row only). The bounding box will be automatically calculated
        from the geometry, rounded up and down as necessary if `spacing` is set.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            A single row geodataframe with a Polygon or Polyline type geometry.

        name_col : str
            Name of the column in the geodataframe to use for setting the name
            of the Region. If  unset, the name of the region will be
            automatically based on the first column of the geodataframe.
            Alternatively, pass in `name="Some Name"` to directly set the name.

        spacing : float
            Number to round coordinates up and down such that the bounding box
            are in nice intervals (requires PyGMT). Set to None to use exact
            bounds of input shape instead (uses Shapely only). Default is 1000m
            for rounding bounding box coordinates to nearest kilometre.

        Returns
        -------
        region : deepicedrain.Region

        """
        if "name" not in kwargs:
            try:
                kwargs["name"] = gdf[name_col]
            except KeyError:
                kwargs["name"] = gdf.iloc[0]

        try:
            import pygmt

            xmin, xmax, ymin, ymax = pygmt.info(
                table=np.vstack(gdf.geometry.exterior.coords.xy).T,
                spacing=float(spacing),
            )
        except (ImportError, TypeError):
            xmin, ymin, xmax, ymax = gdf.geometry.bounds
        kwargs.update({"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax})

        return cls(**kwargs)

    @property
    def scale(self) -> int:
        """
        Automatically set a map scale (1:scale)
        based on x-coordinate range divided by 0.2
        """
        return int((self.xmax - self.xmin) / 0.2)

    def bounds(self, style="lrbt") -> tuple:
        """
        Convenience function to get the bounding box coordinates
        of the region in two different styles, lrbt or lbrt.
        Defaults to 'lrbt', i.e. left, right, bottom, top.
        """
        if style == "lrbt":  # left, right, bottom, top (for PyGMT)
            return (self.xmin, self.xmax, self.ymin, self.ymax)
        elif style == "lbrt":  # left, bottom, right, top (for Shapely, etc)
            return (self.xmin, self.ymin, self.xmax, self.ymax)
        else:
            raise NotImplementedError(f"Unknown style type {style}")

    def datashade(
        self,
        df: pd.DataFrame,
        x_dim: str = "x",
        y_dim: str = "y",
        z_dim: str = "h_range",
        plot_width: int = 1400,
    ) -> xr.DataArray:
        """
        Convenience function to quickly datashade a table of x, y, z points
        into a grid for visualization purposes, using a mean aggregate function
        """
        # Datashade our height values (vector points) onto a grid (raster image)
        # Will maintain the correct aspect ratio according to the region bounds
        canvas: datashader.core.Canvas = datashader.Canvas(
            plot_width=plot_width,
            plot_height=int(
                plot_width * ((self.ymax - self.ymin) / (self.xmax - self.xmin))
            ),
            x_range=(self.xmin, self.xmax),
            y_range=(self.ymin, self.ymax),
        )
        return canvas.points(
            source=df, x=x_dim, y=y_dim, agg=datashader.mean(column=z_dim)
        )

    def subset(
        self, data: xr.Dataset, x_dim: str = "x", y_dim: str = "y", drop: bool = True
    ) -> xr.Dataset:
        """
        Convenience function to find datapoints in an xarray.Dataset or
        pandas.DataFrame that fit within the bounding boxes of this region.
        Note that the 'drop' boolean flag is only valid for xarray.Dataset.
        """
        cond = np.logical_and(
            np.logical_and(data[x_dim] > self.xmin, data[x_dim] < self.xmax),
            np.logical_and(data[y_dim] > self.ymin, data[y_dim] < self.ymax),
        )

        try:
            # xarray.DataArray subset method
            data_subset = data.where(cond=cond, drop=drop)
        except TypeError:
            # pandas.DataFrame subset method
            data_subset = data.loc[cond]

        return data_subset


def deltatime_to_utctime(
    dataarray: xr.DataArray,
    start_epoch: np.datetime64 = np.datetime64("2018-01-01T00:00:00.000000"),
) -> xr.DataArray:
    """
    Converts GPS time in nanoseconds from an epoch (default is 2018 Jan 1st)
    to Coordinated Universal Time (UTC).

    Note, does not account for leap seconds! There are none declared since the
    last one announced on 31/12/2016, so it should be fine for now as of 2020.
    """
    try:
        start_epoch = dataarray.__class__(start_epoch).squeeze()
    except ValueError:  # Could not convert object to NumPy timedelta
        pass

    utc_time: xr.DataArray = start_epoch + dataarray

    return utc_time


def lonlat_to_xy(
    longitude: xr.DataArray, latitude: xr.DataArray, epsg: int = 3031
) -> (xr.DataArray, xr.DataArray):
    """
    Reprojects longitude/latitude EPSG:4326 coordinates to x/y coordinates.
    Default conversion is to Antarctic Stereographic Projection EPSG:3031.
    """
    if hasattr(longitude, "__array__") and callable(longitude.__array__):
        # TODO upgrade to PyProj 3.0 to remove this workaround for passing in
        # dask.dataframe.core.Series or xarray.DataArray objects
        # Based on https://github.com/pyproj4/pyproj/pull/625
        _longitude = longitude.__array__()
        _latitude = latitude.__array__()

    x, y = pyproj.Proj(projparams=epsg)(_longitude, _latitude)

    if hasattr(longitude, "coords"):
        return (
            xr.DataArray(data=x, coords=longitude.coords),
            xr.DataArray(data=y, coords=latitude.coords),
        )
    else:
        return x, y


def point_in_polygon_gpu(
    points_df,  # cudf.DataFrame with x and y columns of point coordinates
    poly_df: gpd.GeoDataFrame,  # geopandas.GeoDataFrame with polygon shapes
    points_x_col: str = "x",
    points_y_col: str = "y",
    poly_label_col: str = None,
):
    """
    Find polygon labels for each of the input points.
    This is a GPU accelerated version that requires cuspatial!

    Parameters
    ----------
    points_df : cudf.DataFrame
        A dataframe in GPU memory containing the x and y coordinates.
    points_x_col : str
        Name of the x coordinate column in points_df. Default is "x".
    points_y_col : str
        Name of the y coordinate column in points_df. Default is "y".

    poly_df : geopandas.GeoDataFrame
        A geodataframe in CPU memory containing polygons geometries in each
        row.
    poly_label_col : str
        Name of the column in poly_df that will be used to label the points,
        e.g. "placename". Default is to automatically use the first column
        unless otherwise specified.

    Returns
    -------
    point_labels : cudf.Series
        A column of labels that indicates which polygon the points fall into.

    """
    import cudf
    import cuspatial

    poly_df_: gpd.GeoDataFrame = poly_df.reset_index()

    # Simply use first column of geodataframe as label if not provided (None)
    # See https://stackoverflow.com/a/22736342/6611055
    poly_label_col: str = poly_label_col or poly_df.columns[0]
    point_labels: cudf.Series = cudf.Series(index=points_df.index).astype(
        poly_df[poly_label_col].dtype
    )

    # Load CPU-based GeoDataFrame into a GPU-based cuspatial friendly format
    # This is a workaround until the related feature request at
    # https://github.com/rapidsai/cuspatial/issues/165 is implemented
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save geodataframe to a temporary shapefile,
        # so that we can load it into GPU memory using cuspatial
        tmpshpfile = os.path.join(tmpdir, "poly_df.shp")
        poly_df_.to_file(filename=tmpshpfile, driver="ESRI Shapefile")

        # Load polygon_offsets, ring_offsets and polygon xy points
        # from temporary shapefile into GPU memory
        poly_offsets, poly_ring_offsets, poly_points = cuspatial.read_polygon_shapefile(
            filename=tmpshpfile
        )

    # Run the actual point in polygon algorithm!
    # Note that cuspatial's point_in_polygon function has a 31 polygon limit,
    # hence the for-loop code below. See also
    # https://github.com/rapidsai/cuspatial/blob/branch-0.15/notebooks/nyc_taxi_years_correlation.ipynb
    num_poly: int = len(poly_df_)
    point_in_poly_iter: list = list(np.arange(0, num_poly, 31)) + [num_poly]
    for i in range(len(point_in_poly_iter) - 1):
        start, end = point_in_poly_iter[i], point_in_poly_iter[i + 1]
        poly_labels: cudf.DataFrame = cuspatial.point_in_polygon(
            test_points_x=points_df[points_x_col],
            test_points_y=points_df[points_y_col],
            poly_offsets=poly_offsets[start:end],
            poly_ring_offsets=poly_ring_offsets,
            poly_points_x=poly_points.x,
            poly_points_y=poly_points.y,
        )

        # Label each point with polygon they fall in
        for label in poly_labels.columns:
            point_labels.loc[poly_labels[label]] = poly_df_.loc[label][poly_label_col]

    return point_labels
