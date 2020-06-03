"""
Geospatial and Temporal class that implements some handy tools.
Does bounding box region subsets, coordinate/time conversions, and more!
"""
import dataclasses
import datetime

import datashader
import numpy as np
import pandas as pd
import pyproj
import xarray as xr


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
        self, ds: xr.Dataset, x_dim: str = "x", y_dim: str = "y", drop: bool = True
    ) -> xr.Dataset:
        """
        Convenience function to find datapoints in an xarray.Dataset
        that fit within the bounding boxes of this region
        """
        cond = np.logical_and(
            np.logical_and(ds[x_dim] > self.xmin, ds[x_dim] < self.xmax),
            np.logical_and(ds[y_dim] > self.ymin, ds[y_dim] < self.ymax),
        )

        return ds.where(cond=cond, drop=drop)


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
    start_epoch = dataarray.__class__(start_epoch).squeeze()

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
