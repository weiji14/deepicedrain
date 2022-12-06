import importlib.resources
import logging

import intake

import deepicedrain
from deepicedrain.deltamath import calculate_delta, dhdt_maxslp, nan_linregress, nanptp
from deepicedrain.extraload import (
    array_to_dataframe,
    ndarray_to_parquet,
    split_tracks,
    wide_to_long,
)
from deepicedrain.lake_algorithms import find_clusters, ice_volume_over_time
from deepicedrain.spatiotemporal import (
    Region,
    deltatime_to_utctime,
    lonlat_to_xy,
    point_in_polygon_gpu,
    spatiotemporal_cube,
)
from deepicedrain.vizplots import (
    IceSat2Explorer,
    plot_alongtrack,
    plot_crossovers,
    plot_icesurface,
)

__version__: str = "0.4.2"

# Loads the ICESat-2 ATLAS intake data catalog
_catalog_path = importlib.resources.path(
    package=deepicedrain, resource="atlas_catalog.yaml"
)
with _catalog_path as uri:
    logging.info(f"Loading intake catalog from {uri}")
    catalog: intake.catalog.local.YAMLFileCatalog = intake.open_catalog(uri=str(uri))
