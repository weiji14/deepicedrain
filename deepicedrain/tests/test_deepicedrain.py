import intake

from deepicedrain import __version__, catalog


def test_version():
    assert __version__ == "0.4.0"


def test_deepicedrain_catalog():
    """
    Test that the intake ATLAS data catalog can be loaded via both
    `deepicedrain.catalog` and `intake.cat.atlas_cat`
    """
    catalog_entries = [
        "icesat2atlasdownloader",
        "icesat2atl06",
        "icesat2atl11",
        "icesat2dhdt",
        "measures_antarctic_boundaries",
        "subglacial_lakes",
        "test_data",
    ]
    assert list(catalog) == catalog_entries
    assert list(intake.cat.atlas_cat) == catalog_entries
