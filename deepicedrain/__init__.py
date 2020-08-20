import importlib.resources
import logging

import deepicedrain
import intake
from deepicedrain.deltamath import calculate_delta, nan_linregress, nanptp
from deepicedrain.extraload import array_to_dataframe, ndarray_to_parquet
from deepicedrain.spatiotemporal import (
    Region,
    deltatime_to_utctime,
    lonlat_to_xy,
    point_in_polygon_gpu,
)

__version__: str = "0.2.1"

# Loads the ICESat-2 ATLAS intake data catalog
_catalog_path = importlib.resources.path(
    package=deepicedrain, resource="atlas_catalog.yaml"
)
with _catalog_path as uri:
    logging.info(f"Loading intake catalog from {uri}")
    catalog: intake.catalog.local.YAMLFileCatalog = intake.open_catalog(uri=str(uri))
