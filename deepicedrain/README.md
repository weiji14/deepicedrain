# DeepIceDrain Python package

Contents:

- :artificial_satellite: atlas_catalog.yaml - [intake](https://intake.readthedocs.io) data catalog for accessing ICESat-2 ATLAS datasets
  - icesat2atlasdownloader - Download Antarctic ICESat-2 ATLAS products from [NSIDC](https://nsidc.org/data/ICESat-2)
  - icesat2atl06 - Reads in ICESat-2 ATL06 data into an xarray.Dataset
  - test_data - Sample ICESat-2 datasets for testing purposes

- :1234: deltamath.py - Mathematical functions for calculating delta changes of some physical unit
  - calculate_delta - Calculates the change in some quantity variable between two ICESat-2 cycles
  - nanptp - Range of values (maximum - minimum) along an axis, ignoring any NaNs
  - nan_linregress - Linear Regression function that handles NaN and NaT values

- :globe_with_meridians: spatiotemporal.py - Tools for doing geospatial and temporal subsetting and conversions
  - Region - Bounding box data class structure that has xarray subsetting capabilities and more!
  - deltatime_to_utctime - Converts GPS time from an epoch (default is 2018 Jan 1st) to UTC time
  - lonlat_to_xy - Reprojects longitude/latitude EPSG:4326 coordinates to x/y EPSG:3031 coordinates

- :card_file_box: extraload.py - Convenience functions for extracting, transforming and loading data
  - array_to_dataframe - Turns a 1D/2D numpy/dask array into a tidy pandas/dask dataframe table
