"""
Feature tests for finding Active Subglacial Lakes in Antactica.
"""
try:
    import cudf as xpd
except ImportError:
    import pandas as xpd
import deepicedrain
import fsspec
from pytest_bdd import given, scenario, then, when


@scenario(
    feature_name="features/subglacial_lakes.feature",
    scenario_name="Subglacial Lake Finder",
    example_converters=dict(location=str, this_many=int),
)
def test_subglacial_lake_finder():
    """Find active subglacial lakes at some place"""
    pass


@given("some altimetry data at <location>", target_fixture="dataframe")
def basin_altimetry_data(location):
    """
    Load up some pre-processed ICESat-2 ATL11 altimetry data with x, y,
    dhdt_slope and referencegroundtrack columns from a Parquet file.
    """
    # TODO use intake_parquet after https://github.com/intake/intake-parquet/issues/18
    with fsspec.open(
        f"simplecache::https://github.com/weiji14/deepicedrain/releases/download/v0.3.1/df_dhdt_{location}.parquet",
        simplecache=dict(cache_storage="ATLXI", same_names=True),
    ) as openfile:
        _dataframe: xpd.DataFrame = xpd.read_parquet(
            openfile, columns=["x", "y", "dhdt_slope", "referencegroundtrack"]
        )
        # Take only 1/4 of the data for speed
        _dataframe: xpd.DataFrame = _dataframe.loc[: len(_dataframe) / 4]

    # Filter to points > 2 * Median(dhdt)
    abs_dhdt: xpd.Series = _dataframe.dhdt_slope.abs()
    dataframe: xpd.DataFrame = _dataframe.loc[abs_dhdt > 2 * abs_dhdt.median()]

    return dataframe


@when("it is passed through an unsupervised clustering algorithm")
def run_unsupervised_clustering(dataframe, context):
    """
    Find draining and filling lake clusters by pass a point cloud through the
    DBSCAN unsupervised clustering algorithm.
    """
    X = dataframe
    cluster_vars = ["x", "y", "dhdt_slope"]

    draining_lake_labels: xpd.Series = -deepicedrain.find_clusters(
        X=X.loc[X.dhdt_slope < 0][cluster_vars]
    )
    filling_lake_labels: xpd.Series = deepicedrain.find_clusters(
        X=X.loc[X.dhdt_slope > 0][cluster_vars]
    )
    lake_labels = xpd.concat(objs=[draining_lake_labels, filling_lake_labels])
    context.lake_labels: xpd.Series = lake_labels.sort_index()

    return context.lake_labels


@then("<this_many> potential subglacial lakes are found")
def verify_subglacial_lake_labels(this_many, context):
    """
    Ensure that the lake_labels column is of int32 type, and that we are
    getting a specific number of unique lake clusters.
    """
    assert context.lake_labels.dtype.name.lower() == "int32"
    clusters: xpd.Series = context.lake_labels.unique()
    assert this_many == len(clusters) - 1

    return clusters
