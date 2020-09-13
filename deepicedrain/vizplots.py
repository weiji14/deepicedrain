"""
Creates interactive visualizations for Exploratory Data Analysis using PyViz
and produce publication quality figures using PyGMT!
"""

import os
import warnings

import numpy as np

import holoviews as hv
import intake
import panel as pn
import param

warnings.filterwarnings(
    action="ignore",
    message="The global colormaps dictionary is no longer considered public API.",
)


class IceSat2Explorer(param.Parameterized):
    """
    ICESat-2 rate of height change over time (dhdt) interactive dashboard.
    Built using HvPlot and Panel.

    Adapted from the "Panel-based Datashader dashboard" at
    https://examples.pyviz.org/datashader_dashboard/dashboard.html.
    See also https://github.com/holoviz/datashader/pull/676.
    """

    plot_variable = param.Selector(
        default="dhdt_slope", objects=["referencegroundtrack", "dhdt_slope", "h_corr"]
    )
    cycle_number = param.Integer(default=7, bounds=(2, 7))
    dhdt_range = param.Range(default=(1.0, 10.0), bounds=(0.0, 20.0))
    rasterize = param.Boolean(default=True)
    datashade = param.Boolean(default=False)

    # Load from intake data source
    # catalog = intake.cat.atlas_cat
    catalog: intake.catalog.local.YAMLFileCatalog = intake.open_catalog(
        os.path.join(os.path.dirname(__file__), "atlas_catalog.yaml")
    )
    placename: str = "whillans_upstream"
    source = catalog.icesat2dhdt(placename=placename)
    if os.path.exists(f"ATLXI/df_dhdt_{placename}.parquet"):
        try:
            import cudf
            import hvplot.cudf

            df_ = cudf.read_parquet(source._urlpath)
        except ImportError:
            df_ = source.to_dask()
        plot: hv.core.spaces.DynamicMap = source.plot.dhdt_slope()  # default plot
        startX, endX = plot.range("x")
        startY, endY = plot.range("y")

    def keep_zoom(self, x_range, y_range):
        self.startX, self.endX = x_range
        self.startY, self.endY = y_range

    @param.depends(
        "cycle_number", "plot_variable", "dhdt_range", "rasterize", "datashade"
    )
    def view(self) -> hv.core.spaces.DynamicMap:
        # Filter/Subset data to what's needed. Wait for
        # https://github.com/holoviz/hvplot/issues/72 to do it properly
        cond = np.logical_and(
            float(self.dhdt_range[0]) < abs(self.df_.dhdt_slope),
            abs(self.df_.dhdt_slope) < float(self.dhdt_range[1]),
        )
        if self.plot_variable == "h_corr":
            df_subset = self.df_.loc[cond].dropna(
                subset=[f"h_corr_{self.cycle_number}"]
            )
        else:
            df_subset = self.df_.loc[cond]

        # Create the plot! Uses plot_kwargs from catalog metdata
        # self.plot = getattr(source.plot, self.plot_variable)()
        self.source = self.catalog.icesat2dhdt(cycle=self.cycle_number)
        plot_kwargs = {
            "xlabel": self.source.metadata["fields"]["x"]["label"],
            "ylabel": self.source.metadata["fields"]["y"]["label"],
            **self.source.metadata["plot"],
            **self.source.metadata["plots"][self.plot_variable],
        }
        plot_kwargs.update(
            rasterize=self.rasterize, datashade=self.datashade, dynspread=self.datashade
        )
        self.plot = df_subset.hvplot(
            title=f"ICESat-2 Cycle {self.cycle_number} {self.plot_variable}",
            **plot_kwargs,
        )

        # Keep zoom level intact when changing the plot_variable
        self.plot = self.plot.redim.range(
            x=(self.startX, self.endX), y=(self.startY, self.endY)
        )
        self.plot = self.plot.opts(active_tools=["pan", "wheel_zoom"])
        rangexy = hv.streams.RangeXY(
            source=self.plot,
            x_range=(self.startX, self.endX),
            y_range=(self.startY, self.endY),
        )
        rangexy.add_subscriber(self.keep_zoom)

        return self.plot

    def widgets(self):
        _widgets = pn.Param(
            self.param,
            widgets={
                "plot_variable": pn.widgets.RadioButtonGroup,
                "cycle_number": pn.widgets.IntSlider,
                "dhdt_range": {"type": pn.widgets.RangeSlider, "name": "dhdt_range_±"},
                "rasterize": pn.widgets.Checkbox,
                "datashade": pn.widgets.Checkbox,
            },
        )
        return pn.Row(
            pn.Column(_widgets[0], _widgets[1], align="center"),
            pn.Column(_widgets[2], _widgets[3], align="center"),
            pn.Column(_widgets[4], _widgets[5], align="center"),
        )
