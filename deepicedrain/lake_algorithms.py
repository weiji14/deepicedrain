"""
Custom algorithms for detecting active subglacial lakes and estimating ice
volume displacement over time.
"""
try:
    import cudf as xpd
except ImportError:
    import pandas as xpd

import numpy as np


def find_clusters(
    X: xpd.DataFrame,
    eps: float = 3000,
    min_samples: int = 250,
    output_colname: str = "cluster_id",
) -> xpd.Series:
    """
    Classify a point cloud into several groups, with each group being assigned
    a positive integer label like 1, 2, 3, etc. Unclassified noise points are
    labelled as NaN.

    Uses Density-based spatial clustering of applications with noise (DBSCAN).
    See also https://www.naftaliharris.com/blog/visualizing-dbscan-clustering

    ***       **         111       NN
    **    **   *         11    22   N
    *     ****     -->   1     2222
      **     **            33     22
    ******               333333

    Parameters
    ----------
    X : cudf.DataFrame or pandas.DataFrame
        A table of X, Y, Z points to run the clustering algorithm on.
    eps : float
        The maximum distance between 2 points such they reside in the same
        neighborhood. Default is 3000 (metres).
    min_samples : int
        The number of samples in a neighborhood such that this group can be
        considered as an important core point (including the point itself).
        Default is 250 (sample points).
    output_colname : str
        The name of the column for the output Series. Default is 'cluster_id'.

    Returns
    -------
    cluster_labels : cudf.Series or pd.Series
        Which cluster each datapoint belongs to. Noisy samples are labeled as
        NaN.
    """
    try:
        from cuml.cluster import DBSCAN
    except ImportError:
        from sklearn.cluster import DBSCAN

    # Run DBSCAN using {eps} m distance, and minimum of {min_samples} points
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X=X)

    cluster_labels = dbscan.labels_ + 1  # noise points -1 becomes 0
    if isinstance(cluster_labels, np.ndarray):
        cluster_labels = xpd.Series(data=cluster_labels, dtype=xpd.Int32Dtype())
    cluster_labels = cluster_labels.mask(cond=cluster_labels == 0)  # turn 0 to NaN
    cluster_labels.index = X.index  # let labels have same index as input data
    cluster_labels.name = output_colname

    return cluster_labels


def ice_volume_over_time(
    df_elev: xpd.DataFrame,
    surface_area: float,
    rolling_window: int or str = "91d",
    elev_col: str = "h_anom",
    time_col: str = "utc_time",
) -> xpd.DataFrame:
    """
    Generates a time-series of ice volume displacement. Ice volume change (dvol
    in m^3) is estimated by multiplying the lake area (in m^2) by the mean
    height anomaly (dh in m), following the methodology of Kim et al. 2016 and
    Siegfried et al., 2016. Think of it as a multiplying the circle of a
    cylinder by it's height:

             _~~~~_
           _~      ~_ <--- Lake area (m^2)
          |~__   __~ |          *
          |   ~~~    | <-- Mean height anomaly (m) from many points inside lake
          |         _|          =
           ~__   __~  <--- Ice volume displacement (m^3)
              ~~~

    More specifically, this function will:
    1) Take a set of height anomaly and time values and calculate a rolling
       mean of height change over time.
    2) Multiply the height change over time sequence in (1) by the lake surface
       area to obtain an estimate of volume change over time.

    Parameters
    ----------
    df_elev : cudf.DataFrame or pandas.DataFrame
        A table of with time and height anomaly columns to run the rolling time
        series calculation on. Ensure that the rows are sorted with time in
        ascending order from oldest to newest.
    surface_area : float
        The ice surface area (of the active lake) experiencing a change in
        height over time. Recommended to provide in unit metres.
    rolling_window : str
        Size of the moving window to calculate the rolling mean, given as a
        time period. Default is '91d' (91 days = 1 ICESat-2 cycle).
    elev_col : str
        The elevation anomaly column name to use from the table data, used to
        calculate the rolling mean height anomaly at every time interval.
        Default is 'h_anom', recommended to provide in unit metres.
    time_col : str
        The time-dimension column name to use from the table data, used in the
        rolling mean algorithm. Default is 'utc_time', ensure that this column
        is provided in a datetime64 format.

    Returns
    -------
    dvol : cudf.Series or pd.Series
        A column of delta volume changes over time. If pint metre units were
        provided in df_elev, then this output will be given in cubic metres
        with a one standard deviation uncertainty range.

    Examples
    --------
    >>> import pandas as pd
    >>> import pint
    >>> import pint_pandas
    >>> ureg = pint.UnitRegistry()
    >>> pint_pandas.PintType.ureg = ureg

    >>> h_anom = pd.Series(
    ...     data=np.random.RandomState(seed=42).rand(100), dtype="pint[metre]"
    ... )
    >>> utc_time = pd.date_range(
    ...    start="2018-10-14", end="2020-09-30", periods=100
    ... )
    >>> df_elev = pd.DataFrame(data={"h_anom": h_anom, "utc_time": utc_time})

    >>> dvol = ice_volume_over_time(
    ...     df_elev=df_elev, surface_area=123 * ureg.metre ** 2
    ... )
    >>> print(f"{dvol.iloc[len(dvol)//2]:Lx}")
    \SI[]{12+/-35}{\meter\cubed}

    References:
    - Kim, B.-H., Lee, C.-K., Seo, K.-W., Lee, W. S., & Scambos, T. A. (2016).
      Active subglacial lakes and channelized water flow beneath the Kamb Ice
      Stream. The Cryosphere, 10(6), 2971–2980.
      https://doi.org/10.5194/tc-10-2971-2016
    - Siegfried, M. R., Fricker, H. A., Carter, S. P., & Tulaczyk, S. (2016).
      Episodic ice velocity fluctuations triggered by a subglacial flood in
      West Antarctica. Geophysical Research Letters, 43(6), 2640–2648.
      https://doi.org/10.1002/2016GL067758
    """
    # Get just the elevation anomaly and time columns
    df_: pd.DataFrame = df_elev[[elev_col, time_col]]

    # Temporarily changing dtype from pint[metre] to float to avoid
    # "DataError: No numeric types to aggregate" in rolling mean calculation
    elev_dtype = df_elev[elev_col].dtype
    if "pint" in str(elev_dtype):
        df_[elev_col] = df_[elev_col].pint.magnitude  # dequantify unit

    # Calculate rolling mean of elevation
    df_roll = df_.rolling(window=rolling_window, on=time_col, min_periods=1)
    elev_mean: np.ndarray = df_roll[elev_col].mean().to_numpy()

    # Calculate elevation anomaly as elevation at time=n minus elevation at time=1
    elev_anom: np.ndarray = elev_mean - elev_mean[0]

    # Add standard deviation uncertainties to mean if pint units are used
    # Need to do it in numpy world instead of pandas to workaround issue
    # in https://github.com/hgrecco/pint-pandas/issues/45, and wait also for
    # pint.Measurements overhaul in https://github.com/hgrecco/pint/issues/350
    if "pint" in str(elev_dtype):
        import uncertainties.unumpy

        elev_std: np.ndarray = df_roll[elev_col].std().to_numpy()
        elev_anom: np.ndarray = (
            uncertainties.unumpy.uarray(nominal_values=elev_anom, std_devs=elev_std)
            * elev_dtype.units
        )

    # Calculate ice volume displacement (m^3) = area (m^2) x height (m)
    dvol: np.ndarray = surface_area * elev_anom
    dvol_dtype: str = (
        f"pint[{dvol.units}]" if "pint" in str(elev_dtype) else df_elev[elev_col].dtype
    )

    ice_dvol: pd.Series = df_[elev_col].__class__(
        data=dvol, dtype=dvol_dtype, index=df_.index
    )

    return ice_dvol
