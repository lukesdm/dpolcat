import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import bokeh.models
    import geopandas as gpd
    import geoviews as gv
    import holoviews as hv
    import hvplot.xarray
    import hvplot.pandas
    import numpy as np
    import odc.stac
    import pandas as pd
    import planetary_computer
    import pystac_client
    import rasterio.features
    import rioxarray
    import shapely.geometry as sg
    import shapely.ops as so
    import xarray

    import dpolcat
    import dpolcat.modelling
    import dpolcat.utils


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## *dpolcat* Demo 04: Urban Change

    In this notebook, we'll develop a basic model for detecting urban change, based on a *dpolcat* time series.

    In `demo_03_batch_processing.py`, we created a *dpolcat* datacube in .zarr format for the Graz area. We'll use a similar one covering a much longer period here, prepared with `scripts/graz_prep.py`.
    """)
    return


@app.cell
def _():
    aoi = gpd.read_file("gis/graz/graz_4326.gpkg")

    date_range = "2020-01-01/2025-12-31"

    dpolcat_zarr = "data/dpolcat_graz_full.zarr"
    sat_direction = "descending"
    sat_rel_orbit = 22
    return aoi, dpolcat_zarr, sat_direction, sat_rel_orbit


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prepare data
    """)
    return


@app.cell
def _(dpolcat_zarr, sat_direction, sat_rel_orbit):
    z = xarray.open_groups(
        dpolcat_zarr, engine="zarr", chunks={}, decode_coords="all"
    )

    da_dpolcat = z[f"/{sat_direction}_{sat_rel_orbit}/dpolcat"]["dpolcat"]

    # Data may not be stored in chronological order, so sort here.
    da_dpolcat = da_dpolcat.sortby("time")

    # CRS may also not be picked up automatically, so set it here.
    da_dpolcat = da_dpolcat.rio.write_crs(da_dpolcat._CRS["wkt"])
    return (da_dpolcat,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Modelling

    Due to the structure and hard materials of buildings, urban areas typically feature a high concentration of double/triple-bounce scattering (referred to in *dpolcat* as `MultiBounce`). So, can we use this knowledge to detect new buildings on previously flat ground over time? Buildings are not the only source of double/triple bounce. Steep natural terrain, and trees, especially without their leaves in winter, can also exhibit this type of scattering.  To address this, and the natural variations in series of radar images (see Inverness EDA Part B/C), let's calculate the modal scattering type of each pixel, by year.
    """)
    return


@app.cell
def _():
    from dpolcat.modelling import MultiBounce

    return (MultiBounce,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Xarray doesn't have a built-in mode operation, but we can use the one from [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html). Apply it with a yearly [`resample`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.resample.html) and [`reduce`](https://docs.xarray.dev/en/stable/generated/xarray.core.resample.DataArrayResample.reduce.html#xarray.core.resample.DataArrayResample.reduce) combination.
    """)
    return


@app.cell
def _():
    import scipy.stats

    return (scipy,)


@app.cell
def _(da_dpolcat, scipy):
    yearly_mode = da_dpolcat.resample(time="YE").reduce(
        lambda cat, axis: scipy.stats.mode(cat, axis=axis).mode, dim="time"
    )
    return (yearly_mode,)


@app.cell
def _(MultiBounce, yearly_mode):
    multibounce = yearly_mode.isin(MultiBounce).astype(np.int8)

    # Tidy metadata
    multibounce = multibounce.rename("multibounce")
    multibounce = multibounce.assign_coords(
        year=multibounce.time.dt.year
    ).swap_dims(time="year")
    return (multibounce,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Visualize the presence of multibounce, year by year.
    """)
    return


@app.cell
def _(multibounce):
    # Using hvplot(`geo=True`) is very slow, but a reproject here instead is fast.
    _mb = multibounce.rio.reproject(3857)

    # Initial map extent, EPSG:3857
    init_extent_xs = (1710000, 1728000)
    init_extent_ys = (5944000, 5962000)

    # Colourmap:
    # Absent (0) -> Transparent
    # Present (1) -> Red
    _mb_cmap = cmap = ["#00000000", "#ff0000"]

    gv.tile_sources.OSM * _mb.hvplot(x="x", y="y").opts(
        cmap=_mb_cmap,
        color_levels=2,
        clim=(-0.5, 1.5),
        cticks=[0, 1],
        title="Multi-bounce presence",
        active_tools=["wheel_zoom"],
        xlim=init_extent_xs,
        ylim=init_extent_ys,
        frame_width=400,
        frame_height=400,
        framewise=False,  # Prevent reset on year change
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You will notice slight differences over the area between years, but we can make clearer the nature of the changes.

    For comparing the presence of `MultiBounce` over two years (MB$_1$ and MB$_2$), we create a new `mb_change` variable, with the definition:

    `mb_change` = MB$_{2}$ + (MB$_2$ - MB$_1$)


    | MB$_1$ | MB$_2$ | `mb_change` | Description | Legend
    | - | - | -  |  -       | - |
    | 1 | 0 | -1 |  Removed | <span style="background-color: #000000;">&nbsp;&nbsp;&nbsp;</span> |
    | 0 | 0 | 0  |  Neither | - |
    | 1 | 1 | 1  |  Remain    | <span style="background-color: #aaaaaa;">&nbsp;&nbsp;&nbsp;</span> |
    | 0 | 1 | 2  |  New     | <span style="background-color: #ff00ff;">&nbsp;&nbsp;&nbsp;</span> |


     This can be calculated simply using the Xarray [`shift`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.shift.html) function.
    """)
    return


@app.cell
def _(multibounce):
    mb_change = (
        multibounce + multibounce - multibounce.shift(year=1, fill_value=127)
    )

    # Drop the first year, which has no change information.
    mb_change = mb_change[1:]
    return (mb_change,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A summary of the numbers:
    """)
    return


@app.cell
def _(mb_change):
    ds_mbch_counts = xarray.Dataset(
        {
            "removed": (mb_change == -1).sum(dim=["y", "x"]),
            # [Skip] # "neither": (mb_change == 0).sum(dim=["y", "x"]),
            "remain": (mb_change == 1).sum(dim=["y", "x"]),
            "new": (mb_change == 2).sum(dim=["y", "x"]),
        }
    )

    ds_mbch_counts.hvplot.bar(
        x="year",
        cmap={"new": "#ff00ff", "remain": "#aaaaaa", "removed": "#000000"},
    ).opts(
        xrotation=30,
        yformatter="%d",
        title="MultiBounce presence, year-on-year",
        ylabel="Count (pixels)",
        xlabel="",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Visualization

    Styria has regular aerial imagery campaigns, the results of which are available as open data (see [here](https://www.landesentwicklung.steiermark.at/cms/beitrag/12866797/145230171) for further details). There are WMS endpoints for accessing these. For the city of Graz area in the time range of our data cube, there is imagery from 11 May 2021 and 19 June 2024, which we can use for comparison. (The dates were found by a corresponding metadata service, per the previously linked page).
    """)
    return


@app.function
# Load WMS as HoloViews Tiles layer
# Credit: Anita Graser - https://github.com/anitagraser/ogd-at-lab/blob/main/notebooks/webmapping.ipynb
def load_wms(server, layer, extents, tile_size=256):
    url = f"{server}?&service=WMS&request=GetMap&layers={layer}&styles=&format=image/png&transparent=true&version=1.3.0&width={tile_size}&height={tile_size}&crs=EPSG:3857"
    url = url + "&bbox={XMIN},{YMIN},{XMAX},{YMAX}"
    return hv.Tiles(url, extents=extents).opts(xlabel="Longitude", ylabel="Latitude")


@app.cell
def _():
    ui_hires = mo.ui.switch(label="Fetch aerial imagery at higher resolution (slower).")
    ui_hires
    return (ui_hires,)


@app.cell
def _(aoi, ui_hires):
    ortho_2021 = load_wms(
        server="https://gis.stmk.gv.at/image/services/OGD_DOP/Flug_2019_2021_RGB/ImageServer/WMSServer",
        layer="Flug_2019_2021_RGB",
        extents=tuple(aoi.to_crs(3857).total_bounds),
        tile_size=512 if ui_hires.value else 256,
    )

    ortho_2024 = load_wms(
        server="https://gis.stmk.gv.at/image/services/OGD_DOP/Flug_2022_2024_RGB/ImageServer/WMSServer",
        layer="Flug_2022_2024_RGB",
        extents=tuple(aoi.to_crs(3857).total_bounds),
        tile_size=512 if ui_hires.value else 256,
    )
    return ortho_2021, ortho_2024


@app.cell
def _():
    # Setup linked crosshairs
    # Credit: https://discourse.holoviz.org/t/linked-crosshair/4326/2
    linked_crosshair = bokeh.models.CrosshairTool(
        overlay=[
            bokeh.models.Span(dimension="width"),
            bokeh.models.Span(dimension="height"),
        ]
    )
    return (linked_crosshair,)


@app.cell
def _(linked_crosshair, mb_change):
    # Set nodata and reproject for visualization
    _da = mb_change.rio.write_nodata(0)
    _da = _da.rio.reproject(3857)

    # Clear coordinates to match WMS layers (otherwise plots won't be linked)
    _da.x.attrs = {}
    _da.y.attrs = {}

    change_plot = _da.hvplot(
        x="x",
        y="y",
    ).opts(
        cmap=["#000000", "#00000000", "#aaaaaa", "#ff00ff"],
        clim=(-1.5, 2.5),
        color_levels=4,
        cticks=[(-1, "Removed"), (-0, "None"), (1, "Both"), (2, "New")],
        xticks=bokeh.models.MercatorTicker(dimension="lon"),
        yticks=bokeh.models.MercatorTicker(dimension="lat"),
        xformatter=bokeh.models.MercatorTickFormatter(dimension="lon"),
        yformatter=bokeh.models.MercatorTickFormatter(dimension="lat"),
        xlabel="Longitude",
        ylabel="Latitude",
        tools=[linked_crosshair],
        active_tools=["wheel_zoom"],
        framewise=False,  # Prevent reset on year change
    )

    change_plot = hv.element.tiles.OSM() * change_plot
    return (change_plot,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Assemble the final plot. Viewing the cell output in full-screen mode is recommended.
    """)
    return


@app.cell
def _(change_plot, linked_crosshair, ortho_2021, ortho_2024):
    plot_all = (
        change_plot.opts(
            tools=[linked_crosshair],
            title="Multibounce change",
            frame_width=400,
            frame_height=400,
            # These don't seem to work (SHOULDDO: fix)
            # xlim=init_extent_xs,
            # ylim=init_extent_ys,
        )
        + ortho_2021.opts(
            tools=[linked_crosshair],
            title="Aerial: 11-05-2021",
            frame_width=400,
            frame_height=400,
        )
        + ortho_2024.opts(
            tools=[linked_crosshair],
            title="Aerial: 19-06-2024",
            frame_width=400,
            frame_height=400,
        )
    ).opts(title="")

    hv.output(widget_location="top_left")

    plot_all
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A QGIS project has been created for further exploring the results - see `gis/graz/graz.qgis`. There you will also find some example points.

    The button below will export `mb_change` to a `.zarr` store with a band for each year. Due to a current software limitation, the bands are named *Band 1*, *Band 2* etc.. These correspond accordingly:

    | Band | Change (from-) to |
    | - | - |
    | Band 1 | (2020-) 2021 |
    | Band 2 | (2021-) 2022 |
    | Band 3 | (2022-) 2023 |
    | Band 4 | (2023-) 2024 |
    | Band 5 | (2024-) 2025 |
    """)
    return


@app.cell
def _():
    btn_export = mo.ui.run_button(label="Export results")
    btn_export
    return (btn_export,)


@app.cell
def _(btn_export, mb_change):
    mo.stop(not btn_export.value, "Click 👆 to export.")

    mb_change.to_zarr("gis/graz/mb_change.zarr", "w")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conclusion

    Hopefully, you will have seen a few instances of urban change highlighted by our model, when exploring above. But, you will also have noticed plenty of typically small areas (1-2 pixels) where there is no obvious change. Some common caveats and limitations of working with SAR apply here - georeferencing errors and noise. The model could likely be improved with some further spatial filtering.
    """)
    return


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
