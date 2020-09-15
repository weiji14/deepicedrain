"""
Extract, Tranform and Load (ETL) functions for handling ICESat-2 point clouds.
Copies data seamlessly between different array structures and file formats!
"""
import functools

import dask
import pandas as pd
import zarr


def array_to_dataframe(
    array: dask.array.core.Array, colname: str = None, startcol: int = 0
):
    """
    Converts a 1D or 2D data array into a tidy dataframe structure.
    An array of shape (m, n) will turn into a table with m rows and n columns.

    These are the possible conversions:
    - numpy array -> pandas DataFrame
    - dask Array -> dask DataFrame

    Pass in a colname to set the column name. By default, it will automatically
    use the array.name attribute in dask Arrays, but this can be overriden.
    For 2D arrays, columns will be formatted as 'col_0', 'col_1', 'col_2' and
    so on. The startcol argument allows adjustment of the starting column
    number, helpful if you prefer starting from 1, e.g. 'col_1', 'col_2', etc.

    See also https://github.com/dask/dask/issues/5021
    """
    if not colname:
        colname = array.name if hasattr(array, "name") else ""

    if array.ndim == 1:  # 1-dimensional arrays
        columns = colname
    elif array.ndim == 2:  # 2-dimensional arrays
        colname += "_" if colname != "" else ""  # add underscore to name
        columns = [f"{colname}{i+startcol}" for i in range(array.shape[1])]

    try:
        # Attempt dask Array to dask DataFrame conversion
        dataframe: dask.dataframe.core.DataFrame = array.to_dask_dataframe(
            columns=columns
        )
    except AttributeError:
        # Fallback to converting to pandas.DataFrame
        dataframe: pd.DataFrame = pd.DataFrame.from_records(data=array, columns=columns)

    return dataframe


def ndarray_to_parquet(
    ndarray,
    parquetpath: str,
    variables: list = None,
    dropnacols: list = None,
    engine: str = "pyarrow",
    **kwargs,
) -> pd.DataFrame:
    """
    Converts an n-dimensional xarray Dataset or Zarr Array into an Apache
    Parquet columnar file via an intermediate Dask/Pandas DataFrame format.
    This is a convenience function that wraps around array_to_dataframe,
    intended to make converting n number of arrays easier.

    Parameters
    ----------
    ndarray : xarray.Dataset or zarr.hierarchy.Group
        An n-dimensional array in xarray containing several coordinate and data
        variables, or a Zarr array containing several variables.
    parquetpath : str
        Filepath to where the resulting parquet file will be stored.
    variables : list
        Name(s) of the variables/columns that will be stored to the parquet
        file. If not provided, all the variables in the zarr group will be
        stored.
    dropnacols : list
        Drop rows containing NaN values in these fields before saving to the
        Parquet file.
    engine : str
        Parquet library to use. Choose from 'auto', 'fastparquet', 'pyarrow'.
        Default is "pyarrow".
    **kwargs : dict
        Extra options to be passed on to pandas.DataFrame.to_parquet.

    Returns
    -------
    point_labels : cudf.Series
        A column of labels that indicates which polygon the points fall into.

    """
    if variables is None:
        try:
            variables = [varname for varname, _ in ndarray.arrays()]
        except AttributeError:
            variables = [c for c in ndarray.coords] + [d for d in ndarray.data_vars]

    if isinstance(ndarray, zarr.hierarchy.Group):
        array_func = lambda varname: dask.array.from_zarr(ndarray[varname])
    else:
        array_func = lambda varname: ndarray[varname].data

    dataframes: list = [
        array_to_dataframe(array=array_func(varname), colname=varname, startcol=1)
        for varname in variables
    ]
    dataframe: dask.dataframe.core.DataFrame = dask.dataframe.concat(
        dfs=dataframes, axis="columns"
    )
    if dropnacols:
        dataframe = dataframe.dropna(subset=dropnacols)

    # Convert to pandas DataFrame first before saving to a single binary
    # parquet file, rather than going directly from a Dask DataFrame to a
    # series of parquet files in a parquet folder. This ensures that cudf can
    # read it later, see https://github.com/rapidsai/cudf/issues/1688
    df: pd.DataFrame = dataframe.compute()
    df.to_parquet(path=parquetpath, engine=engine, **kwargs)

    return df


@functools.wraps(wrapped=pd.wide_to_long)
def wide_to_long(
    df: pd.DataFrame,
    stubnames: list,
    i: str = "id",
    j: str = None,
    sep: str = "_",
    suffix: str = "\\d+",
) -> pd.DataFrame:
    """
    A wrapper around pandas.wide_to_long that wraps around pandas.melt!
    Handles setting an index (Default to "id") and resetting the second level
    index (the 'j' variable), while dropping NaN values too!

    Documentation for input arguments are the same as pd.wide_to_long. This
    convenience function just uses different default arguments for 'i' and
    'sep'.
    """
    df[i] = df.index
    df_long = pd.wide_to_long(
        df=df, stubnames=stubnames, i=i, j=j, sep=sep, suffix=suffix
    )
    df_long = df_long.reset_index(level=j)

    return df_long.dropna()
