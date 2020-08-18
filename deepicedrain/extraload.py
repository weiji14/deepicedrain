"""
Extract, Tranform and Load (ETL) functions for handling ICESat-2 point clouds.
Copies data seamlessly between different array structures and file formats!
"""
import pandas as pd


def array_to_dataframe(array, colname: str = None, startcol: int = 0):
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
