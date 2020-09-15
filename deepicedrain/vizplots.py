"""
Creates interactive visualizations for Exploratory Data Analysis using PyViz
and produce publication quality figures using PyGMT!
"""

import os
import warnings

import holoviews as hv
import intake
import numpy as np
import pandas as pd
import panel as pn
import param
import pygmt

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

    # Param Widgets that interactively control plot settings
    plot_variable = param.Selector(
        default="dhdt_slope", objects=["referencegroundtrack", "dhdt_slope", "h_corr"]
    )
    cycle_number = param.Integer(default=7, bounds=(2, 7))
    dhdt_range = param.Range(default=(1.0, 10.0), bounds=(0.0, 20.0))
    rasterize = param.Boolean(default=True)
    datashade = param.Boolean(default=False)

    def __init__(self, placename: str = "whillans_upstream", **kwargs):
        super().__init__(**kwargs)
        self.placename = placename

        # Load from intake data source
        # catalog = intake.cat.atlas_cat
        self.catalog: intake.catalog.local.YAMLFileCatalog = intake.open_catalog(
            os.path.join(os.path.dirname(__file__), "atlas_catalog.yaml")
        )
        self.source = self.catalog.icesat2dhdt(placename=self.placename)

        try:
            import cudf
            import hvplot.cudf

            self.df_ = cudf.read_parquet(self.source._urlpath)
        except ImportError:
            self.df_ = self.source.to_dask()

        # Setup default plot (dhdt_slope) and x/y axis limits
        self.plot: hv.core.spaces.DynamicMap = self.source.plot.dhdt_slope()
        self.startX, self.endX = self.plot.range("x")
        self.startY, self.endY = self.plot.range("y")

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
        self.source = self.catalog.icesat2dhdt(
            cycle=self.cycle_number, placename=self.placename
        )
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


def plot_alongtrack(
    df: pd.DataFrame,
    rgtpair: str,
    regionname: str,
    x_var: str = "x_atc",
    y_var: str = "h_corr",
    time_var: str = "utc_time",
    cycle_var: str = "cycle_number",
    spacing: str = "1000/5",
    oldtonew: bool = True,
) -> pygmt.Figure:
    """
    Plot 2D along track view of Ice Surface Height Changes over Time.
    Uses PyGMT to produce the figure.

    Parameters
    ----------
    df : pandas.DataFrame
        A table containing the ICESat-2 track data from multiple cycles. It
        should ideally have columns called 'x_atc', 'h_corr', 'utc_time' and
        'cycle_number'.
    x_var : str
        The x-dimension column name to use from the table data, plotted
        on the horizontal x-axis. Default is 'x_atc'.
    y_var : str
        The y-dimension column name to use from the table data, plotted
        on the vertical x-axis. Default is 'h_corr'.
    time_var : str
        The time-dimension column name to use from the table data, used to
        calculate the mean datetime for each track in every cycle. Default is
        'utc_time'.
    cycle_var : str
        The column name from the table which is used to determine which time
        cycle each row/observation falls into. Default is 'cycle_number'.
    spacing : str
        Provide as 'dx/dy' increments, this is passed directly to `pygmt.info`
        and used to round up and down the x and y axis limits for a nicer plot
        frame. Default is '1000/5'.
    oldtonew : bool
        Determine the plot order (True: Cycle 1 -> Cycle n; False: Cycle n ->
        Cycle 1), useful when you want the legend to go one way or the other.
        For example, the default `oldtonew=True` is recommended when plotting
        decreasing elevation over time (i.e. lake draining). Set to False
        instead to reverse the order, recommended when plotting increasing
        elevation over time (i.e. lake filling).

    Returns
    -------
    fig : pygmt.Figure
        A pygmt Figure instance containing the along track plot which can be
        viewed using fig.show() or saved to a file using fig.savefig()
    """
    fig = pygmt.Figure()
    # Setup map frame, title, axis annotations, etc
    fig.basemap(
        projection="X30c/10c",
        region=pygmt.info(table=df[[x_var, y_var]], spacing=spacing),
        frame=[
            rf'WSne+t"ICESat-2 Change in Ice Surface Height over Time at {regionname}"',
            'xaf+l"Along track x (m)"',
            'yaf+l"Height (m)"',
        ],
    )
    fig.text(
        text=f"Reference Ground Track {rgtpair}",
        position="TC",
        offset="jTC0c/0.2c",
        V="q",
    )
    # Colors from https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=9
    cycle_colors: dict = {
        1: "#999999",
        2: "#f781bf",
        3: "#a65628",
        4: "#ffff33",
        5: "#ff7f00",
        6: "#984ea3",
        7: "#4daf4a",
        8: "#377eb8",
        9: "#e41a1c",
    }
    # Choose only cycles that need to be plotted, reverse order if requested
    cycles: list = sorted(df[cycle_var].unique(), reverse=not oldtonew)
    cycle_colors: dict = {cycle: cycle_colors[cycle] for cycle in cycles}

    # For each cycle, plot the height values (y_var) along the track (x_var)
    for cycle, color in cycle_colors.items():
        df_ = df.query(expr=f"{cycle_var} == @cycle").copy()
        # Get x, y, time
        data = np.column_stack(tup=(df_[x_var], df_[y_var]))
        time_nsec = df_[time_var].mean()
        time_sec = np.datetime_as_string(arr=time_nsec.to_datetime64(), unit="s")
        label = f'"Cycle {cycle} at {time_sec}"'

        # Plot data points
        fig.plot(data=data, style="c0.05c", color=color, label=label)
        # Plot line connecting points
        # fig.plot(data=data, pen=f"faint,{color},-", label=f'"+g-1l+s0.15c"')

    fig.legend(S=3, position="JMR+JMR+o0.2c", box="+gwhite+p1p")
    return fig
