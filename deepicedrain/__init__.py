import importlib.resources
import logging

import intake

import deepicedrain
from deepicedrain.deltamath import calculate_delta
from deepicedrain.spatiotemporal import Region, deltatime_to_utctime, lonlat_to_xy

__version__: str = "0.1.0"

# Loads the ICESat-2 ATLAS intake data catalog
_catalog_path = importlib.resources.path(
    package=deepicedrain, resource="atlas_catalog.yaml"
)
with _catalog_path as uri:
    logging.info(f"Loading intake catalog from {uri}")
    catalog: intake.catalog.local.YAMLFileCatalog = intake.open_catalog(uri=str(uri))
