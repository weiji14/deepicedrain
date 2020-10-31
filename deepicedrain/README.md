# DeepIceDrain Python package

Contents:

- :artificial_satellite: atlas_catalog.yaml - [intake](https://intake.readthedocs.io) data catalog for accessing ICESat-2 ATLAS datasets
  - icesat2atlasdownloader - Download Antarctic ICESat-2 ATLAS products from [NSIDC](https://nsidc.org/data/ICESat-2)
  - icesat2atl06 - Reads in ICESat-2 ATL06 data into an xarray.Dataset
  - icesat2dhdt - Preprocessed ICESat-2 height change over time (dhdt) data in a columnar format
  - test_data - Sample ICESat-2 datasets for testing purposes

- :1234: deltamath.py - Mathematical functions for calculating delta changes of some physical unit
  - calculate_delta - Calculates the change in some quantity variable between two ICESat-2 cycles
  - nanptp - Range of values (maximum - minimum) along an axis, ignoring any NaNs
  - nan_linregress - Linear Regression function that handles NaN and NaT values

- :globe_with_meridians: spatiotemporal.py - Tools for doing geospatial and temporal subsetting and conversions
  - Region - Bounding box data class structure that has xarray subsetting capabilities and more!
  - deltatime_to_utctime - Converts GPS time from an epoch (default is 2018 Jan 1st) to UTC time
  - lonlat_to_xy - Reprojects longitude/latitude EPSG:4326 coordinates to x/y EPSG:3031 coordinates
  - spatiotemporal_cube - Interpolates a time-series point cloud into an xarray.Dataset data cube

- :card_file_box: extraload.py - Convenience functions for extracting, transforming and loading data
  - array_to_dataframe - Turns a 1D/2D numpy/dask array into a tidy pandas/dask dataframe table
  - ndarray_to_parquet - Turns an n-dimensional xarray/zarr array into an a parquet columnar format
  - wide_to_long - Turns a pandas dataframe table with many columns into one with many rows

- :droplet: lakealgorithms.py - Custom algorithms for detecting and filtering active subglacial lakes
  - find_clusters - Density based clustering algorithm (DBSCAN) to group points into lakes

- :world_map: vizplots.py - Makes interactive dashboard plots and publication quality figures
  - IceSat2Explorer - Dashboard for interacting with ICESat-2 point clouds on a 2D map
  - plot_alongtrack - Makes a 2D along track figure of height measurements taken at different cycle times
  - plot_crossovers - Makes a figure showing how elevation is changing at many crossover points over time
