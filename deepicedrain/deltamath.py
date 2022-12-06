"""
DeepIceDrain functions for calculating delta changes, such as for ice elevation
differencing (dh), measuring lengths of time (dt), and related measures.
"""
import numpy as np
import scipy.stats
import xarray as xr


def calculate_delta(
    dataset: xr.Dataset,
    oldcyclenum: int = 5,
    newcyclenum: int = 6,
    variable: str = "h_corr",
) -> xr.DataArray:
    """
    Calculates the change in some quantity variable between two ICESat-2 cycles
    i.e. new minus old.

    Example ATL11 variables to use:

    h_corr - corrected height
    delta_time - GPS time for the segments for each pass
    """

    oldcycle: xr.Dataset = dataset.sel(cycle_number=oldcyclenum)
    newcycle: xr.Dataset = dataset.sel(cycle_number=newcyclenum)

    delta_quantity: xr.DataArray = newcycle[variable] - oldcycle[variable]

    return delta_quantity


def nanptp(a, axis=None) -> np.ndarray:
    """
    Range of values (maximum - minimum) along an axis, ignoring any NaNs.
    When slices with less than two non-NaN values are encountered,
    a ``RuntimeWarning`` is raised and Nan is returned for that slice.

    Adapted from https://github.com/numpy/numpy/pull/13220
    """
    return np.nanmax(a=a, axis=axis) - np.nanmin(a=a, axis=axis)


def nan_linregress(x, y) -> np.ndarray:
    """
    Linear Regression function that handles NaN and NaT values.
    Hardcoded so that x is expected to be the time array.

    Stacking the outputs (slope, intercept, rvalue, pvalue, stderr)
    into one numpy.ndarray to keep xarray.apply_ufuncs happy.
    Kudos to https://stackoverflow.com/a/60524715/6611055
    """
    x = np.atleast_2d(x)  # shape of at least (1, 6) instead of (6,)
    y = np.atleast_2d(y)  # shape of at least (1, 6) instead of (6,)

    mask = ~np.isnan(y)
    x = x[mask]  # .where(cond=mask, drop=True)
    y = y[mask]  # .where(cond=mask, drop=True)

    try:
        linregress_result = np.array(scipy.stats.linregress(x=x, y=y))
    except ValueError:
        # if x.size == 0 or y.size == 0:
        linregress_result = np.full(shape=(5,), fill_value=np.NaN)

    return linregress_result


def dhdt_maxslp(x, y) -> np.ndarray:
    """
    Maximum slope (i.e. steepest gradient) for any consecutive paired value
    within an elevation time-series. Hardcoded so that x is expected to be the
    time array.

    For example, in the plot below, the rate of elevation change over time is
    greatest from point B to C, so the algorithm will return the dhdt_maxslp
    value as (elev_C - elev_B) / (time_C - time_B).

             ^
             |        C
             |           D
    elev (m) |
             |     B        E
             |  A              F
             -------------------->
                     time

    Note that NaN values are ignored in the calculation. So if point E had a
    NaN value, the algorithm will calculate dhdt between point F and D.
    """
    x = np.atleast_2d(x)  # shape of at least (1, 9) instead of (9,)
    y = np.atleast_2d(y)  # shape of at least (1, 9) instead of (9,)

    mask = ~np.isnan(y)

    # Rolling difference, i.e. x_2 - x_1, x_3 - x_2, etc
    roll_xdiff = np.diff(a=x[mask])  # rolling time difference
    roll_ydiff = np.diff(a=y[mask])  # rolling elev difference

    try:
        # Get maximum absolute dhdt slope value
        dhdt_values = roll_ydiff / roll_xdiff
        maxslp_index = np.argmax(np.abs(dhdt_values))
        maxslp_result = dhdt_values[maxslp_index]
    except ValueError:  # axes don't match array
        maxslp_result = np.full(shape=(1,), fill_value=np.NaN)

    return maxslp_result
