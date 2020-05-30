"""
Geospatial and Temporal class that implements some handy tools.
Does bounding box region subsets, coordinate/time conversions, and more!
"""
import dataclasses
import datetime

import numpy as np
import xarray as xr


@dataclasses.dataclass(frozen=True)
class Region:
    """
    A nice region data structure that outputs a tuple of bounding box
    coordinates, has xarray subsetting capabilities and a map scale property.
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
    utc_time: xr.DataArray = dataarray.__class__(start_epoch) + dataarray

    return utc_time
