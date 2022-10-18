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
import tqdm
import xarray as xr

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
    cycle_number = param.Integer(default=7, bounds=(2, 8))
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
    regionname: str,
    rgtpair: str,
    elev_var: str = "h_corr",
    xatc_var: str = "x_atc",
    time_var: str = "utc_time",
    cycle_var: str = "cycle_number",
    spacing: str = "1000/5",
    oldtonew: bool = True,
) -> pygmt.Figure:
    """
    Plot 2D along track cross-section view of Ice Surface Elevation over Time.
    Uses PyGMT to produce the figure. The input table should look something like
    below (more columns can be present too).

    | cycle_number | x_atc | h_corr |       utc_time      |
    |--------------|-------|--------|---------------------|
    |      1       |  500  |   14   | 2020-01-01T01:12:34 |
    |      2       |  500  |   12   | 2020-04-01T12:23:45 |
    |      3       |  500  |   10   | 2020-07-01T23:34:56 |

    which will produce a plot similar to the following:

      Ice Surface Elevation over each ICESat-2 cycle at Some Ice Stream
         ^
         |  Reference Ground Track 1234_pt3
         |
         | -----------------------------    --- Cycle 1 at 2020-01-01T01:12:34
    Elev | -.-.-.-.-.-.-.-.-.-.-.-.-.-.-    -.- Cycle 2 at 2020-04-01T12:23:45
         | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    ~~~ Cycle 3 at 2020-07-01T23:34:56
         |____________________________________________________________________>
                   Along track x

    Parameters
    ----------
    df : pandas.DataFrame
        A table containing the ICESat-2 track data from multiple cycles. It
        should ideally have columns called 'x_atc', 'h_corr', 'utc_time' and
        'cycle_number'.
    regionname : str
        A descriptive placename for the data (e.g. Some Ice Stream), to be used
        in the figure's main title.
    rgtpair : str
        The name of the referencegroundtrack pair being plotted (e.g. 1234_pt3),
        to be used in the figure's subtitle.
    elev_var : str
        The elevation column name to use from the table data, plotted on the
        vertical y-axis. Default is 'h_corr'.
    xatc_var : str
        The x along-track column name to use from the table data, plotted on the
        horizontal x-axis. Default is 'x_atc'.
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
        region=pygmt.info(table=df[[xatc_var, elev_var]], spacing=spacing),
        frame=[
            rf'WSne+t"Ice Surface Elevation over each ICESat-2 cycle at {regionname}"',
            'xaf+l"Along track x (m)"',
            'yaf+l"Elevation (m)"',
        ],
    )
    fig.text(
        text=f"Reference Ground Track {rgtpair}",
        position="TC",
        offset="jTC0c/0.2c",
        verbose="q",
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

    # For each cycle, plot the height values (elev_var) along the track (xatc_var)
    for cycle, color in cycle_colors.items():
        df_ = df.query(expr=f"{cycle_var} == @cycle").copy()
        # Get x, y, time
        data = np.column_stack(tup=(df_[xatc_var], df_[elev_var]))
        time_nsec = df_[time_var].mean()
        time_sec = np.datetime_as_string(arr=time_nsec.to_datetime64(), unit="s")
        label = f'"Cycle {cycle} at {time_sec}"'

        # Plot data points
        fig.plot(data=data, style="c0.05c", color=color, label=label)
        # Plot line connecting points
        # fig.plot(data=data, pen=f"faint,{color},-", label=f'"+g-1l+s0.15c"')

    fig.legend(S=3, position="jBL+jBL+o0.2c", box="+gwhite+p1p")
    return fig


def _plot_crossover_area(
    outline_points: str or pd.DataFrame,
    df: np.ndarray,
    fig: pygmt.Figure = None,
    plotsize: float = 8,
    cmap: str = "vik",
    elev_range: list = None,
    wsne: str = "SE",
):
    """"""
    fig = fig or pygmt.Figure()
    plotregion: np.ndarray = pygmt.info(outline_points, per_column=True, spacing=1000)

    # Map frame in metre units
    fig.basemap(frame="+n", region=plotregion, projection=f"X{plotsize}c")
    # Plot lake boundary in cyan
    fig.plot(data=outline_points, region=plotregion, pen="thin,cyan2,-.")
    # Plot crossover point locations
    pygmt.makecpt(cmap=cmap, series=elev_range, reverse=True)
    fig.plot(data=df, style="d0.1c", cmap=True)
    # Map frame in kilometre units
    with pygmt.config(FONT_ANNOT_PRIMARY=f"{plotsize+2}p", FONT_LABEL=f"{plotsize+2}p"):
        fig.basemap(
            frame=[
                wsne,
                'xa+l"Polar Stereographic X (km)"',
                'ya+l"Polar Stereographic Y (km)"',
            ],
            region=plotregion / 1000,
            projection=f"X{plotsize}c",
        )
    return fig


def plot_crossovers(
    df: pd.DataFrame,
    regionname: str,
    elev_var: str = "h",
    time_var: str = "t",
    track_var: str = "track1_track2",
    spacing: float = 2.5,
    elev_filter: float = 0.2,
    outline_points: str or pd.DataFrame = None,
) -> pygmt.Figure:
    """
    Plot to show how elevation is changing at many crossover points over time.
    Uses PyGMT to produce the figure. The input table should look something like
    below (more columns can be present too).

    |  track1_track2   |  h  |          t          |
    |------------------|-----|---------------------|
    | 0111_pt1x0222pt2 | 111 | 2020-01-01T01:12:34 |
    | 0222_pt2x0333pt3 | 110 | 2020-04-01T12:23:45 |
    | 0333_pt3x0111pt1 | 101 | 2020-07-01T23:34:56 |

    which will produce a plot similar to the following:

                                           ^
         | --- * |            Ice Stream W |
         | \ * \ | y                       |
         |* ---- |          -a---a---a---a |
         |_______|         /               |  Xover
         |   x         -a-/      -b---b    |  Elev (m)
         | -a---a---a-/      -b-/          |
         |   -b---b---b---b-/  -c-         |
         |  -c---c---c---c----/   \-c---c  |
         |_________________________________|
                       Date

    Parameters
    ----------
    df : pandas.DataFrame
        A table containing the ICESat-2 track data from multiple cycles. It
        should ideally have columns called 'h', 't', and 'track1_track2'.
    regionname : str
        A descriptive placename for the data (e.g. Some Ice Stream), to be used
        in the figure's main title.
    elev_var : str
        The elevation column name to use from the table data, plotted on the
        vertical y-axis. Default is 'h'.
    time_var : str
        The time-dimension column name to use from the table data, plotted on
        the horizontal x-axis. Default is 't'.
    track_var : str
        The track column name to use from the table data, containing variables
        in the form of track1xtrack2 (note that 'x' is a hardcoded delimiter),
        e.g. 0111_pt1x0222pt2. Default is 'track1_track2'.
    spacing : str or float
        Provide as a 'dy' increment, this is passed on to `pygmt.info` and used
        to round up and down the y axis (elev_var) limits for a nicer plot
        frame. Default is 2.5.
    elev_filter : float
        Minimum elevation change required for the crossover point to show up
        on the plot. Default is 1.0 (metres).
    outline_points : str or pd.DataFrame
        A set of nodes making up a polygon to be plotted in a 2D inset map at
        one corner of the figure, provided as a pandas.DataFrame table with xyz
        columns or a path to an OGR GMT file (necessary for multi-polygons).
        Optional.

    Returns
    -------
    fig : pygmt.Figure
        A pygmt Figure instance containing the crossover plot which can be
        viewed using fig.show() or saved to a file using fig.savefig()
    """
    fig = pygmt.Figure()

    # Setup map frame, title, axis annotations, etc
    with pygmt.config(
        FONT_ANNOT_PRIMARY="9p",
        FORMAT_TIME_PRIMARY_MAP="abbreviated",
        FORMAT_DATE_MAP="o",
    ):
        # Get plot region, spaced out into nice intervals
        # Note that passing time columns into pygmt.info doesn't work well yet,
        # see https://github.com/GenericMappingTools/pygmt/issues/597
        plotregion = np.array(
            [
                df[time_var].min() - pd.Timedelta(1, unit="W"),
                df[time_var].max() + pd.Timedelta(1, unit="W"),
                *pygmt.info(table=df[[elev_var]], spacing=spacing)[:2],
            ]
        )
        # pygmt.info(table=df[[time_var, elev_var]], spacing=f"1W/{spacing}", f="0T")
        _y_label = "Elevation anomaly" if elev_var == "h_anom" else "Elevation"
        fig.basemap(
            projection="X12c/12c",
            region=plotregion,
            frame=[
                rf"wSnE",
                "pxa1Of1o",  # primary time axis, 1 mOnth annotation and minor axis
                "sx1Y",  # secondary time axis, 1 Year intervals
                f'yaf+l"{_y_label} at crossover (m)"',
            ],
        )
        fig.text(
            text=regionname, position="TR", justify="TR", offset="-0.2c", font="14p"
        )

    crossovers = df.groupby(by=track_var)
    dh_max = df[elev_var].abs().max()
    elev_range = [-dh_max, +dh_max]
    pygmt.makecpt(cmap="vik", series=elev_range, reverse=True)
    xover: dict = {"x": [], "y": [], "dhdt": []}
    for track1_track2, indexes in tqdm.tqdm(crossovers.indices.items()):
        df_ = df.loc[indexes].sort_values(by=time_var)
        dhdt, _ = np.polyfit(
            x=df_[time_var].astype(np.int64), y=df_[elev_var], deg=1
        ) * (365.25 * 24 * 60 * 60 * 1_000_000_000)
        if abs(dhdt) > elev_filter:  # Plot only > 1 metre height change
            track1, track2 = track1_track2.split("x")
            fig.plot(
                x=df_[time_var],
                y=df_[elev_var],
                zvalue=dhdt,
                style="c0.1c",
                cmap=True,
                pen="thin+z",
            )
            # Plot line connecting points
            fig.plot(
                x=df_[time_var],
                y=df_[elev_var],
                zvalue=dhdt,
                pen=f"faint,+z,-",
                cmap=True,
            )
            # Get x, y and color (elev) for inset map
            if outline_points:
                _x, _y = df_.iloc[0][["x", "y"]]
                xover["x"].append(_x)
                xover["y"].append(_y)
                xover["dhdt"].append(dhdt)

    # 2D inset map showing lake outline and crossover point locations
    if outline_points:
        if df[elev_var].mean() > 0:  # uptrend
            position, wsne = "TL", "SE"
        else:  # downtrend
            position, wsne = "BL", "NE"

        with pygmt.clib.Session() as session:
            session.call_module(module="inset", args=f"begin -Dj{position}+w4c -N")

        ## Plot insets
        fig = _plot_crossover_area(
            outline_points=outline_points,
            df=pd.DataFrame(data=xover).to_numpy(),
            fig=fig,
            plotsize=4,
            cmap="vik",
            elev_range=elev_range,
            wsne=wsne,
        )

        with pygmt.clib.Session() as session:
            session.call_module(module="inset", args="end")

    return fig


def plot_icesurface(
    grid: str or xr.DataArray = None,
    grid_region: tuple or np.ndarray = None,
    diff_grid: str or xr.DataArray = None,
    diff_grid_region: tuple or np.ndarray = None,
    track_points: pd.DataFrame = None,
    outline_points: str or pd.DataFrame = None,
    azimuth: float = 157.5,
    elevation: float = 45,
    title: str = "",
) -> pygmt.Figure:
    """
    Plot to show a 3D perspective of an elevation grid surface on the top, and
    the differenced grid on the bottom. Also allows for ovelaying track points
    and a polygon outline. This function is custom designed for showcasing
    ICESat-2 altimetry data of an active subglacial lake surface changing over
    time. The resulting plot will be similar to the right plot below:

         ___________                     Subglacial Lake X at YYYYMMDD
        |__|__|__|__|                          ^
        |__|__|__|__|                        z |
      y |__|__z__|__|                          |  ^~^~~^~      ^
        |__|__|__|__|   ___________  -->        \ ~~^~~~^^~~    \  Elev (m)
        |__|__|__|__|  |__|__|__|__|             \  ~^^~~^~~~    \
              x        |__|__|__|__|           ^  \__ __ __ __    v
                     y |__|_dz__|__|        dz |
                       |__|__|__|__|           |  ^~^~~^~       ^
                       |__|__|__|__|            \ ~~^~~~^^~~     \  Diff (m)
                             x                 y \  ~^^~~^~~~     \
                                                  \__ __ __ __     v
                                                        x

    Uses `pygmt.grdview` to produce the figure. The main input grid can be a
    NetCDF filename or an xarray.DataArray with x, y, z variables, while the
    diff_grid must be an xarray.DataArray. Note that there are several
    hardcoded defaults like the vertical exaggeration (0.1x) and axis labels.

    Parameters
    ----------
    grid : str or xr.DataArray
        The main digital surface elevation model to plot, provided as a file
        name or xarray.DataArray grid.
    grid_region : tuple or np.ndarray
        The bounding cube of the grid given as (xmin, xmax, ymin, ymax, zmin,
        zmax).
    diff_grid : str or xr.DataArray
        A differenced elevation grid as an xarray.DataArray.
    diff_grid_region : tuple or np.ndarray
        The bounding cube of the diff_grid given as (xmin, xmax, ymin, ymax,
        zmin, zmax).
    track_points : pd.DataFrame
        Altimetry track points to plot on top of the main grid surface,
        provided as a pandas.DataFrame table with xyz columns. Optional.
    outline_points : str or pd.DataFrame
        A set of nodes making up a polygon to be plotted on top of the main
        grid surface, provided as a pandas.DataFrame table with xyz columns or
        a path to an OGR GMT file (necessary for multi-polygons). Optional.
    azimuth : float
        Angle of viewpoint in degrees from 0-360. Default is 157.5 (SSE),
    elevation : float
        Angle from horizon in degrees from 0-90. Default is 45.
    title : str
        Main heading text (e.g. "Subglacial Lake X at YYYYMMDD"). Default is ""
        (blank).

    Returns
    -------
    fig : pygmt.Figure
        A pygmt Figure instance containing the 3D perspective grid plot which
        can be viewed using fig.show() or saved to a file using fig.savefig()
    """
    assert len(grid_region) == 6  # (xmin, xmax, ymin, ymax, zmin, zmax)

    fig = pygmt.Figure()

    ## Bottom plot
    # Normalized ice surface elevation change grid
    try:
        if diff_grid.min() == diff_grid.max():
            # add some tiny random noise to make plot work
            np.random.seed(seed=int(elevation))
            diff_grid = diff_grid + abs(
                np.random.normal(scale=1e-32, size=diff_grid.shape)
            )
    except AttributeError:
        pass
    try:
        series = diff_grid_region[-2:]
    except TypeError:
        series = pygmt.grdinfo(grid=diff_grid, nearest_multiple="1+s")[2:-3]
    finally:
        pygmt.makecpt(cmap="roma", series=series)
    fig.grdview(
        grid=diff_grid,
        projection="X10c",
        region=diff_grid_region,
        shading=False,
        frame=[
            f"SWZ",  # plot South, West axes, and Z-axis
            'xaf+l"Polar Stereographic X (m)"',  # add x-axis annotations and minor ticks
            'yaf+l"Polar Stereographic Y (m)"',  # add y-axis annotations and minor ticks
            f"zaf",  # add z-axis annotations, minor ticks and axis label
        ],
        cmap=True,
        zscale=0.1,  # zscaling factor, hardcoded to 0.1x vertical exaggeration
        # zsize="5c",  # z-axis size, hardcoded to be 5 centimetres
        surftype="sim",  # surface, image and mesh plot
        perspective=[azimuth, elevation],  # perspective using azimuth/elevation
        # W="c0.05p,black,solid",  # draw contours
    )
    fig.colorbar(
        cmap=True,
        position="JMR+o1c/0c+w7c/0.5c+n",
        frame=[
            'x+l"Elevation Trend"',
            "y+lm/yr",
        ]
        if "?dhdt" in str(diff_grid)
        else ['x+l"Elevation Change"', "y+lm"],
        perspective=True,
    )

    ## Top plot
    fig.shift_origin(yshift="9c")
    # Ice surface elevation grid
    pygmt.makecpt(cmap="lapaz", series=grid_region[-2:])
    fig.grdview(
        grid=grid,
        projection="X10c",
        region=grid_region,
        shading=True,
        frame=[
            f'SWZ+t"{title}"',  # plot South, West axes, and Z-axis
            "xf",  # add x-axis minor ticks
            "yf",  # add y-axis minor ticks
            f"zaf",  # add z-axis annotations, minor ticks and axis label
        ],
        cmap=True,
        zscale=0.1,  # zscaling factor, hardcoded to 0.1x vertical exaggeration
        # zsize="5c",  # z-axis size, hardcoded to be 5 centimetres
        surftype="sim",  # surface, image and mesh plot
        perspective=[azimuth, elevation],  # perspective using azimuth/elevation
        # W="c0.05p,black,solid",  # draw contours
    )
    fig.colorbar(
        cmap=True,
        shading=True,
        position="JMR+o1c/0c+w7c/0.5c+n",
        frame=['x+l"Elevation"', "y+lm"],
        perspective=True,
    )

    # Plot satellite track line points in green
    if track_points is not None:
        fig.plot3d(
            data=track_points,
            color="green",
            style="c0.02c",
            zscale=True,
            perspective=True,
        )
    # Plot lake boundary outline as cyan dashed line
    if outline_points is not None:
        with pygmt.helpers.GMTTempFile() as tmpfile:
            pygmt.grdtrack(
                points=outline_points,
                grid=grid,
                region="/".join(map(str, grid_region[:-2])),
                outfile=tmpfile.name,
                verbose="e",
            )
            _df = pd.read_csv(tmpfile.name, sep="\t", names=["x", "y", "z"])
            pygmt.grdtrack(
                points=outline_points,
                grid=grid,
                region="/".join(map(str, grid_region[:-2])),
                outfile=tmpfile.name,
                d=f"o{_df.z.median()}",  # fill NaN points with median height
                verbose="e",
            )
            fig.plot3d(
                data=tmpfile.name,
                region=grid_region,
                pen="thicker,cyan2,-.",
                zscale=True,
                perspective=True,
            )
    return fig
