"""
Geographic class that implements some handy geographic tools.
"""
import dataclasses

import numpy as np
import xarray as xr


@dataclasses.dataclass(frozen=True)
class BBox:
    """
    A BoundingBox structure that outputs nice tuples of coordinates,
    includes xarray subsetting capabilities and has a map scale property.
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

    def subset(self, ds: xr.Dataset, x_dim: str = "x", y_dim: str = "y") -> xr.Dataset:
        """
        Convenience function to find datapoints in an xarray.Dataset
        that fit within the bounding boxes of this region
        """
        return np.logical_and(
            np.logical_and(ds[x_dim] > self.xmin, ds[x_dim] < self.xmax),
            np.logical_and(ds[y_dim] > self.ymin, ds[y_dim] < self.ymax),
        )
