"""
Custom algorithms for helping to detect active subglacial lakes.
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
