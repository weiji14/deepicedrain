"""
Geospatial and Temporal class that implements some handy tools.
Does bounding box region subsets, coordinate/time conversions, and more!
"""
import dataclasses
import datetime
import os
import shutil
import tempfile

import datashader
import geopandas as gpd
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

    See also https://pyproj4.github.io/pyproj/latest/api/proj.html#pyproj-proj

    Parameters
    ----------
    longitude : xr.DataArray or dask.dataframe.core.Series
        Input longitude coordinate(s).

    latitude : xr.DataArray or dask.dataframe.core.Series
        Input latitude coordinate(s).

    epsg : int
        EPSG integer code for the desired output coordinate system. Default is
        3031 for Antarctic Polar Stereographic Projection.

    Returns
    -------
    x : xr.DataArray or dask.dataframe.core.Series
        The transformed x coordinate(s).

    y : xr.DataArray or dask.dataframe.core.Series
        The transformed y coordinate(s).
    """
    x, y = pyproj.Proj(projparams=epsg)(longitude, latitude)

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


def spatiotemporal_cube(
    table: pd.DataFrame,
    placename: str = "",
    x_var: str = "x",
    y_var: str = "y",
    z_var: str = "h_corr",
    spacing: int = 250,
    cycles: list = None,
    folder: str = "",
) -> xr.Dataset:
    """
    Interpolates a time-series point cloud into an xarray.Dataset data cube.
    Uses `pygmt`'s blockmedian and surface algorithms to produce individual
    NetCDF grids, and `xarray` to stack each NetCDF grid into one dataset.

    Steps are as follows:

    1. Create several xarray.DataArray grid surfaces from a table of points,
       one for each time cycle.
    2. Stacked the grids along a time cycle axis into a xarray.Dataset which is
       a spatiotemporal data cube with 'x', 'y' and 'cycle_number' dimensions.

                             _1__2__3_
            *   *           /  /  /  /|
         *   *             /  /  /  / |
       *   *    *         /__/__/__/  |  y
    *    *   *      -->   |  |  |  |  |
      *    *   *          |  |  |  | /
        *    *            |__|__|__|/  x
                             cycle

    Parameters
    ----------
    table : pandas.DataFrame
        A table containing the ICESat-2 track data from multiple cycles. It
        should ideally have geographical columns called 'x', 'y', and attribute
        columns like 'h_corr_1', 'h_corr_2', etc for each cycle time.
    placename : str
        Optional. A descriptive placename for the data (e.g. some_ice_stream),
        to be used in the temporary NetCDF filename.
    x_var : str
        The x coordinate column name to use from the table data. Default is
        'x'.
    y_var : str
        The y coordinate column name to use from the table data. Default is
        'y'.
    z_var : str
        The z column name to use from the table data. This will be the
        attribute that the surface algorithm will run on. Default is 'h_corr'.
    spacing : float or str
        The spatial resolution of the resulting grid, provided as a number or
        as 'dx/dy' increments. This is passed on to `pygmt.blockmedian` and
        `pygmt.surface`. Default is 250 (metres).
    cycles : list
        The cycle numbers to run the gridding algorithm on, e.g. [3, 4] will
        use columns 'h_corr_3' and 'h_corr_4'. Default is None which will
        automatically determine the cycles for a given z_var.
    folder : str
        The folder to keep the intermediate NetCDF file in. Default is to place
        the files in the current working directory.

    Returns
    -------
    cube : xarray.Dataset
        A 3-dimensional data cube made of digital surfaces stacked along a time
        cycle axis.

    """
    import pygmt
    import tqdm

    # Determine grid's bounding box region (xmin, xmax, ymin, ymax)
    grid_region: np.ndarray = pygmt.info(
        table=table[[x_var, y_var]], spacing=f"s{spacing}"
    )

    # Create one grid surface for each time cycle
    if cycles is None:
        cycles: list = [
            int(col[len(z_var) + 1 :]) for col in table.columns if col.startswith(z_var)
        ]
    _placename = f"_{placename}" if placename else ""
    for cycle in tqdm.tqdm(iterable=cycles):
        df_trimmed = pygmt.blockmedian(
            table=table[[x_var, y_var, f"{z_var}_{cycle}"]].dropna(),
            region=grid_region,
            spacing=f"{spacing}+e",
        )
        outfile = f"{z_var}{_placename}_cycle_{cycle}.nc"
        pygmt.surface(
            data=df_trimmed.values,
            region=grid_region,
            spacing=spacing,
            T=0.35,  # tension factor
            V="e",  # error messages only
            outfile=outfile,
        )
        # print(pygmt.grdinfo(outfile))

    # Stack several NetCDF grids into one NetCDF along the time cycle axis
    paths: list = [f"{z_var}{_placename}_cycle_{cycle}.nc" for cycle in cycles]
    dataset: xr.Dataset = xr.open_mfdataset(
        paths=paths,
        combine="nested",
        concat_dim=[pd.Index(data=cycles, name="cycle_number")],
        attrs_file=paths[-1],
    )

    # Move files into new folder if requested
    if folder:
        [shutil.move(src=path, dst=folder) for path in paths]

    return dataset
