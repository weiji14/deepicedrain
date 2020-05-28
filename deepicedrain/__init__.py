import intake
from deepicedrain.deltamath import calculate_delta
from deepicedrain.spatiotemporal import Region

__version__: str = "0.1.0"

# Loads the ICESat-2 ATLAS intake data catalog
catalog: intake.catalog.local.YAMLFileCatalog = intake.open_catalog(
    uri="atlas_catalog.yaml"
)
