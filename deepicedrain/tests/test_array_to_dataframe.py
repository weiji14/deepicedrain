"""
Tests the array_to_dataframe function
"""
import dask
import numpy as np
import pandas as pd
import pytest

from deepicedrain import array_to_dataframe


@pytest.mark.parametrize("shape", [(10, 1), (10, 2)])
def test_numpy_array_to_pandas_dataframe(shape):
    """
    Test converting from a numpy.array to a pandas.Dataframe, and ensure that
    the colname argument works.
    """
    array: np.ndarray = np.ones(shape=shape)
    dataframe = array_to_dataframe(array=array)

    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe.columns) == shape[1]
    assert dataframe.columns.to_list() == [str(i) for i in range(shape[1])]


@pytest.mark.parametrize("shape", [(10, 1), (10, 2)])
def test_dask_array_to_dask_dataframe(shape):
    """
    Test converting from a dask.array to a dask.dataframe, and ensure that the
    startcol argument works.
    """
    array: dask.array.core.Array = dask.array.ones(shape=shape, name="varname")
    dataframe = array_to_dataframe(array=array, startcol=1)

    assert isinstance(dataframe, dask.dataframe.core.DataFrame)
    assert len(dataframe.columns) == shape[1]
    assert dataframe.columns.to_list() == [f"varname_{i+1}" for i in range(shape[1])]
