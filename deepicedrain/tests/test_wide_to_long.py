"""
Tests converting a wide dataframe with multiple columns to a long dataframe
with several more rows.
"""
import numpy as np
import pandas as pd

from deepicedrain import wide_to_long


def test_wide_to_long():
    """
    Tests that wide_to_long works to flatten height/time data stored in multiple
    columns (one for each cycle) into a few columns only.
    """
    df: pd.DataFrame = pd.DataFrame(
        data={
            "x": np.random.rand(12),
            "y": np.random.rand(12),
            "height_1": np.random.rand(12),
            "height_2": np.append(arr=np.random.rand(11), values=np.nan),
            "height_3": np.random.rand(12),
            "time_1": pd.date_range(start="2020-01-01", periods=12),
            "time_2": pd.date_range(start="2020-05-01", periods=12),
            "time_3": pd.date_range(start="2020-09-01", periods=12),
        }
    )
    df_long = wide_to_long(df=df, stubnames=["height", "time"], j="cycle_number")

    assert len(df_long) == 35  # should have 1 missing row because of the NaN
    assert not df_long.isnull().values.any()  # ensure NaN containing rows were dropped

    assert set(df_long.columns) == set(["cycle_number", "x", "y", "height", "time"])
    assert list(df_long.cycle_number.unique()) == [1, 2, 3]
