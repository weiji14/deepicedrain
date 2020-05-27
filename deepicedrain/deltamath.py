"""
DeepIceDrain functions for calculating delta changes, such as for ice elevation
differencing (dh), measuring lengths of time (dt), and related measures.
"""
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
