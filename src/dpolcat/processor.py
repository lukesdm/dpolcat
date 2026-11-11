import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    from collections import defaultdict
    from functools import partial
    import importlib.metadata
    from pathlib import Path
    import xml.etree.ElementTree as ET

    import bokeh.models
    import dask
    from dask.diagnostics import ProgressBar
    import geopandas as gpd
    import numpy as np
    import odc.stac
    
    import pandas as pd
    import pystac_client
    import planetary_computer
    import shapely.geometry as sg
    import xarray
    import holoviews as hv
    import hvplot.pandas
    import hvplot.xarray
    import rioxarray
    import zarr

    from dpolcat.pc_rio_driver import PCRioDriver
    import dpolcat as dp
    import dpolcat.utils

    hv.extension("bokeh")
    
    try:
        from osgeo import gdal
        gdal.UseExceptions()
    except ImportError:
        print("WARNING: GDAL is not installed. Some features will not work.")

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Luke McQuade, October 2026
    
    # *dpolcat*: Processor

    This is a notebook/module for creating Sentinel-1 *dpolcat* time series/datacubes for a user-supplied area of interest (AoI) and time period. The main steps of the pipeline:
    - Find STAC items matching the AoI, time period, and selected orbit.
    - Create a lazy Xarray dataset, grouped by datatake
    - Apply speckle filtering (optional)
    - Perform dpolcat categorization

    This is written directly to a ZARR store. This can be appended to, for example, when new items are available. Lazy Dask processing is employed, meaning no data loading or processing is performed until the final writing step. I.e., between this the STAC search and this, the steps in the notebook just build up a Dask task graph.

    Convention: This is a parameterized notebook intended to be run externally (see [marimo.App.run]( https://docs.marimo.io/api/app/#marimo.App.run)). Variables beginning with `p_` can be injected when running the notebook this way, e.g., `dpolcat.processor.app.run({"p_aoi": my_aoi, "p_date_range": my_date_range, ...})`. Be careful renaming/changing these! The most important are:

    | Parameter | Description |
    | - | - |
    | `p_headless` | Whether to enable headless execution. Should be `True` when running externally. |
    | `p_aoi` | The area of interest as a GeoDataFrame containing a single polygon. |
    | `p_date_range` | The date range as a string with format `{from_date}/{to_date}`, e.g., "2020-01-01/2020-03-31" |
    | `p_sat_direction` | Selected orbit direction ("ascending" or "descending") |
    | `p_sat_rel_orbit` | Selected relative orbit number (integer) |
    | `p_speckle_filter` | Speckle filter to apply, "Frost" (default) or "Unfiltered". |
    | `p_speckle_filter_window` | Speckle filter window size, default 5. |
    | `p_out_zarr_path` | Path to the ZARR store. |
    | `p_write_intermediate_data` | Whether to write intermediate backscatter data also (defaults to False). |
    """)
    return


@app.cell
def _():
    p_headless = False
    return (p_headless,)


@app.cell
def _():
    mo.md(r"""
    ## Search and Select Sentinel-1 RTC data
    """)
    return


@app.cell
def _():
    # Sample AoI bounding boxes, with format:
    # [lon_min, lat_min, lon_max, lat_max]
    sample_aois = {
        "SalzburgCity": [13.02, 47.76, 13.09, 47.83],
        # Salzburg and Salzach Valley
        "SalzburgSouth": [12.92, 47.50, 13.19, 47.86],
        "Tenerife": [-16.96, 27.92, -16.05, 28.61],
        "GranCanaria": [-15.84, 27.72, -15.34, 28.19],
        # Eastern part of Lake Titicaca, La Paz, and some of the Cordillera Real
        "Titicaca": [-69.41, -16.71, -67.87, -15.61],
        "LaPaz": [-68.224, -16.603, -68.0542, -16.430],
        # Inverness and surroundings to the East
        "Inverness": [-4.32, 57.34, -3.63, 57.63],
        "London": [-0.599, 51.231, 0.347, 51.734],
        # Sentinel-2 footprint around Graz
        "33TWN": [14.999, 46.856, 16.467, 47.853],
        "Duisburg": [6.695, 51.407, 6.847, 51.462],
    }

    ui_aoi = mo.ui.dropdown(options=sample_aois.keys(), value=list(sample_aois.keys())[0])
    ui_date_range = mo.ui.date_range(
        start="2015-01-01", value=("2025-05-01", "2025-05-31")
    )

    mo.md(f"""
    Area of interest: {ui_aoi} <br>
    Search dates: {ui_date_range} <br>
    """)
    return sample_aois, ui_aoi, ui_date_range


@app.cell
def _(ui_aoi):
    p_aoi_name = ui_aoi.value
    return (p_aoi_name,)


@app.cell
def _(p_aoi_name, sample_aois):
    _bbox = sample_aois[p_aoi_name]
    p_aoi = gpd.GeoDataFrame(geometry=[sg.box(*_bbox)], crs="EPSG:4326")
    return (p_aoi,)


@app.cell
def _(p_aoi):
    bbox = p_aoi.total_bounds
    return (bbox,)


@app.cell
def _(ui_date_range):
    _d1, _d2 = ui_date_range.value
    p_date_range = f"{_d1.year}-{_d1.month:02}-{_d1.day:02}/{_d2.year}-{_d2.month:02}-{_d2.day:02}"
    return (p_date_range,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Find best matching acquisitions - those which share the same orbit/track* and, when combined, cover the entirety of the AoI.

    *This is crucial, as different acquisition geometries give incomparable results.
    """)
    return


@app.cell
def _(p_aoi, p_date_range):
    stac_items_gdf = dpolcat.utils.find_stac_items(p_aoi, date_range=p_date_range)
    df_coverages = dpolcat.utils.calc_coverages(stac_items_gdf, aoi=p_aoi)
    return df_coverages, stac_items_gdf


@app.cell
def _(df_coverages):
    ui_orbit_selector = mo.ui.table(df_coverages.reset_index(), selection="single")
    ui_orbit_selector
    return (ui_orbit_selector,)


@app.cell
def _(ui_orbit_selector):
    mo.stop(
        len(ui_orbit_selector.value) == 0
        or ui_orbit_selector.value["full_scenes"].values[0] == 0,
        "Selected an entry with at least 1 full scene.",
    )
    p_sat_direction = ui_orbit_selector.value["sat:orbit_state"].values[0]
    p_sat_rel_orbit = ui_orbit_selector.value["sat:relative_orbit"].values[0]
    return p_sat_direction, p_sat_rel_orbit


@app.cell
def filter_items_by_orbit(p_sat_direction, p_sat_rel_orbit, stac_items_gdf):
    sel_items_gdf = dpolcat.utils.select_by_orbit(
        stac_items_gdf,
        orbit_direction=p_sat_direction,
        relative_orbit=p_sat_rel_orbit,
    )
    sel_items = sel_items_gdf["_item"].values
    return sel_items, sel_items_gdf


@app.cell
def _(sel_items_gdf):
    sel_items_gdf
    return


@app.cell
def get_items_epsg(sel_items):
    _epsgs = set([int(item.properties["proj:code"][5:]) for item in sel_items])
    assert len(_epsgs) == 1, "items are of different CRSs, this is not supported."
    epsg_num = int(_epsgs.pop())
    return (epsg_num,)


@app.cell
def plot_footprints(p_aoi, sel_items_gdf):
    # Visualize the AoI and item footprints. Note: AoI geometry may appear a little off due to reprojection compat issue (https://github.com/oceanmodeling/stofs-event-dashboard/pull/36).
    (
        sel_items_gdf.hvplot(geo=True, alpha=0.2, color="orange")
        * p_aoi.hvplot(geo=True, tiles=True, alpha=0.2, color="blue")
    ).opts(tools=["wheel_zoom"])
    return


@app.cell
def _():
    mo.md(r"""
    ## Processing
    """)
    return


@app.cell
def _():
    ui_existing_zarr_path = mo.ui.text(
        label="Use existing Zarr store (optional):",
        full_width=True,
        placeholder="data/my_existing_dpolcat_store.zarr",
    )
    ui_existing_zarr_path
    return (ui_existing_zarr_path,)


@app.cell
def _(ui_existing_zarr_path):
    p_out_zarr_path = ui_existing_zarr_path.value
    return (p_out_zarr_path,)


@app.cell
def _():
    ui_speckle_filter = mo.ui.dropdown(options=["Unfiltered", "Frost"], value="Frost")
    ui_speckle_filter
    return (ui_speckle_filter,)


@app.cell
def _(ui_speckle_filter):
    p_speckle_filter = ui_speckle_filter.value
    return (p_speckle_filter,)


@app.cell
def _():
    p_speckle_filter_window = 5
    return (p_speckle_filter_window,)


@app.cell
def _():
    ui_write_intermediate_data = mo.ui.switch(label="Write intermediate (backscatter) data")
    ui_write_intermediate_data
    return (ui_write_intermediate_data,)


@app.cell
def _(ui_write_intermediate_data):
    p_write_intermediate_data = ui_write_intermediate_data.value
    return (p_write_intermediate_data,)


@app.cell
def _(p_headless, sel_items):
    # Only show run button once valid items have been chosen
    _ = sel_items

    if not p_headless:
        ui_run = mo.ui.run_button(label="⚙️ Perform processing")
    else:
        # In headless mode, trigger processing with a fake pressed button.
        ui_run = mo.ui.button(value=True)

    mo.md(f"""
    ### Begin processing
    {ui_run}
    """
    )
    # ui_run
    return (ui_run,)


@app.cell
def _():
    resolution = 10  # Native to data source
    return (resolution,)


@app.cell
def _():
    # Chunk size, for both reading (Dask) and writing (Zarr).
    # Chosen for a mix of time series and single-scene operations.
    DASK_CHUNKS = {"time": 10, "y": 1024, "x": 1024}
    ZARR_CHUNKS = (10, 1024, 1024)  # ⚠️ be very careful changing this!
    return DASK_CHUNKS, ZARR_CHUNKS


@app.cell
def _(DASK_CHUNKS, bbox, epsg_num, resolution, sel_items, ui_run):
    mo.stop(
        not ui_run.value,
        mo.md(
            "Click the [button above](#begin-processing) to start processing."
        ),
    )

    # Create lazy-loading dataset
    ds = odc.stac.load(
        sel_items,
        bands=["vv", "vh"],
        bbox=bbox,
        crs=f"EPSG:{epsg_num}",
        resolution=resolution,
        chunks={},  # Use default for now, we set this properly after re-sorting.
        groupby="s1:datatake_id",
        nodata=np.nan,
        driver=PCRioDriver(),
        with_properties=[
            {
                "key": "s1:datatake_id",
                "dtype": "uint32",
                "fuser": lambda s: int(s[0]),
            }
        ],
    )

    # Sort by time (until now, it's sorted by datatake_id, which isn't chronologically ordered).
    # We also need to re-chunk because of this.
    ds = ds.sortby("time").chunk(DASK_CHUNKS)

    # Add original item IDs as coordinates
    _ids_by_dt = defaultdict(list)
    for _item in sel_items:
        _ids_by_dt[int(_item.properties["s1:datatake_id"])].append(_item.id)
    _item_ids = [
        ",".join(sorted(_ids_by_dt[d]))
        for d in ds["s1_datatake_id"].values.astype(int)
    ]
    # Convert item id string to an object, otherwise it gets a fixed-length type
    # in the Zarr that breaks if appended data isn't exactly the same shape.
    _item_ids = np.array(_item_ids, dtype=object)

    ds = ds.assign_coords(item_ids=("time", _item_ids))

    vv_lin = ds.vv
    vh_lin = ds.vh
    return vh_lin, vv_lin


@app.cell
def preprocess(p_speckle_filter, p_speckle_filter_window, vh_lin, vv_lin):
    # Apply speckle filtering and scaling

    if p_speckle_filter == "Frost":
        FROST_DAMPING_FACTOR = 2.0  # COULDDO: Parameterize if needed
        vv_filt = dpolcat.utils.frost_xr(
            vv_lin,
            nanny=True,
            damping_factor=FROST_DAMPING_FACTOR,
            win_size=p_speckle_filter_window,
        )
        vh_filt = dpolcat.utils.frost_xr(
            vh_lin,
            nanny=True,
            damping_factor=FROST_DAMPING_FACTOR,
            win_size=p_speckle_filter_window,
        )
        vv_db = dp.linear_to_db(vv_filt)
        vh_db = dp.linear_to_db(vh_filt)
    else:
        vv_db = dp.linear_to_db(vv_lin)
        vh_db = dp.linear_to_db(vh_lin)
    return FROST_DAMPING_FACTOR, vh_db, vv_db


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Metadata
    """)
    return


@app.function
def get_dpolcat_version():
    return importlib.metadata.version("dpolcat")


@app.cell
def _(
    FROST_DAMPING_FACTOR,
    p_sat_direction,
    p_sat_rel_orbit,
    p_speckle_filter,
    p_speckle_filter_window,
):
    def add_common_metadata(da, epsg_num):
        _da = da.rio.write_crs(epsg_num)
        _wkt = _da.rio.crs.to_wkt()
        _dpolcat_version = get_dpolcat_version()
        _speckle_filter_string = {
            "Frost": f"{p_speckle_filter} (win_size={p_speckle_filter_window}, damping_factor={FROST_DAMPING_FACTOR})",
            "Unfiltered": "Unfiltered",
        }[p_speckle_filter]
        _da = _da.assign_attrs(
            {
                "sat:orbit_state": p_sat_direction,
                "sat:relative_orbit": p_sat_rel_orbit,
                "_CRS": {"wkt": _wkt},  # Required for GDAL compat.
                "dpolcat:speckle_filter": _speckle_filter_string,
                "dpolcat:version": _dpolcat_version,
            }
        )

        return _da

    return (add_common_metadata,)


@app.cell
def _():
    mo.md(r"""
    ### Categorization
    """)
    return


@app.function
def new_zarr_path(aoi_name):
    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S")
    path = f"data/dpolcat-{aoi_name}_{timestamp}.zarr"
    return path


@app.cell
def _(
    ZARR_CHUNKS,
    add_common_metadata,
    epsg_num,
    p_aoi_name,
    p_out_zarr_path,
    p_sat_direction,
    p_sat_rel_orbit,
    p_write_intermediate_data,
    ui_run,
    vh_db,
    vv_db,
):
    mo.stop(
        not ui_run.value,
        mo.md(
            "Click the [button above](#begin-processing) to start processing."
        ),
    )

    # Categorization Dask pipeline; write to Zarr
    # --------------
    categorized = dp.xr_categorize(vv_db, vh_db)
    categorized = categorized.rio.write_crs(epsg_num)
    categorized.attrs.pop("nodata", None)  # Remove old 'nan' nodata
    # For some reason this breaks plotting later:
    # categorized = categorized.rio.write_nodata(0)

    out_zarr_path = p_out_zarr_path or new_zarr_path(p_aoi_name)
    zarr_group_path = f"/{p_sat_direction}_{p_sat_rel_orbit}"

    layers = {"dpolcat": categorized}
    if p_write_intermediate_data:
        layers |= {"vv_db": vv_db, "vh_db": vh_db}

    try:
        zarr.open_group(out_zarr_path, path=zarr_group_path, mode="r")
        _group_exists = True
    except (FileNotFoundError, zarr.errors.GroupNotFoundError):
        _group_exists = False

    _time_encoding = {
        "time": {
            "units": "microseconds since 1970-01-01",
            "dtype": "int64",
        }
    }

    writes = [
        add_common_metadata(da, epsg_num=epsg_num)
        .to_dataset(name=name)
        .to_zarr(
            out_zarr_path,
            group=f"{zarr_group_path}/{name}",
            mode="a",
            append_dim="time" if _group_exists else None,
            zarr_format=3,
            consolidated=False,
            encoding=(
                None
                if _group_exists
                else _time_encoding | {name: {"chunks": ZARR_CHUNKS}}
            ),
            compute=False,
            align_chunks=True,
        )
        for name, da in layers.items()
    ]
    with ProgressBar(minimum=1):
        dask.compute(*writes)

    # Add overviews
    # -------------
    OVERVIEW_LEVELS = [2, 4, 8, 16]

    # Use this even for continuous data (better physicality and nodata handling)
    OVERVIEW_RESAMPLING = "NEAREST"

    _gdal_ds = gdal.OpenEx(
        out_zarr_path,
        gdal.OF_MULTIDIM_RASTER | gdal.OF_UPDATE,
        open_options=["USE_CONSOLIDATED_METADATA=NO"],
    )
    _gdal_root = _gdal_ds.GetRootGroup()

    for _layer in layers:
        _gdal_arr = _gdal_root.OpenMDArrayFromFullname(
            f"{zarr_group_path}/{_layer}/{_layer}"
        )
        _gdal_arr.BuildOverviews(OVERVIEW_RESAMPLING, OVERVIEW_LEVELS)

    # A note on the Zarr structure: the way multiscales/overviews work means
    # another level of nesting is needed whenever more than 1 variable is present
    # Expect something like, e.g.:
    # /{direction}_{relorbit}/dpolcat/dpolcat        (native resolution)
    # /{direction}_{relorbit}/dpolcat/ovr_2x/dpolcat (half resolution)
    # /{direction}_{relorbit}/dpolcat/ovr_4x/dpolcat (quarter resolution)
    # /{direction}_{relorbit}/dpolcat/ovr_*x/dpolcat (etc)
    # /{direction}_{relorbit}/vv_db/vv_db
    # /{direction}_{relorbit}/vv_db/ovr_2x/vv_db
    # etc

    # Close GDAL items & finalize
    # ----------
    del _gdal_arr, _gdal_root, _gdal_ds

    zarr.consolidate_metadata(out_zarr_path)

    result = xarray.open_groups(
        out_zarr_path, engine="zarr", chunks={}, decode_coords="all"
    )
    return OVERVIEW_LEVELS, layers, out_zarr_path, result, zarr_group_path


@app.cell
def _(result, zarr_group_path):
    # Results as a more friendly format
    result_ds = xarray.Dataset(
        {"dpolcat": result[f"{zarr_group_path}/dpolcat"]["dpolcat"]}
    )
    if result.get(f"{zarr_group_path}/vv_db") and result.get(
        f"{zarr_group_path}/vh_db"
    ):
        result_ds["vv_db"] = result[f"{zarr_group_path}/vv_db"]["vv_db"]
        result_ds["vh_db"] = result[f"{zarr_group_path}/vh_db"]["vh_db"]
    return (result_ds,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Visualization
    """)
    return


@app.cell
def _():
    ui_viz_enabled = mo.ui.switch(label="Viz enabled")
    ui_viz_enabled
    return (ui_viz_enabled,)


@app.cell
def _(ui_viz_enabled):
    mo.stop(not ui_viz_enabled.value)

    preview_resolution = 40
    return (preview_resolution,)


@app.cell
def _(preview_resolution, resolution, result_ds):
    _step = preview_resolution // resolution
    result_preview = result_ds.isel(x=slice(None, None, _step), y=slice(None, None, _step))
    # SHOULDDO: load using multiscale, and maybe use datashader rasterize

    # result_preview = rioxarray.open_rasterio(
    #     out_zarr_path
    # )

    # # Have to re-add time coordinates
    # result_preview = result_preview.rename(band="time").assign_coords(
    #     time=result.time.values
    # )
    return (result_preview,)


@app.cell
def _(p_write_intermediate_data, result_preview):
    # Result preview.

    def plot_backscatter(da):
        return da.hvplot.image(
            x="x", y="y", cmap="gray", data_aspect=1.0, clim=(-35, 15)
        )

    plot_width = 280

    # Setup linked crosshairs,
    # via https://discourse.holoviz.org/t/linked-crosshair/4326/2
    _width = bokeh.models.Span(dimension="width")
    _height = bokeh.models.Span(dimension="height")
    linked_crosshair = bokeh.models.CrosshairTool(overlay=[_width, _height])

    _dpolcat = result_preview["dpolcat"].astype("str")
    _dpolcat_plot = _dpolcat.hvplot(
        x="x",
        y="y",
        data_aspect=1.0,
        cmap=dpolcat.utils.make_cmap(),
        frame_width=plot_width,
        tools=[linked_crosshair],
    )

    if p_write_intermediate_data:
        _plot = (
            _dpolcat_plot
            + dpolcat.utils.make_legend(size=250).opts(width=350)
            + plot_backscatter(result_preview["vv_db"]).opts(
                frame_width=plot_width, tools=[linked_crosshair]
            )
            + plot_backscatter(result_preview["vh_db"]).opts(
                frame_width=plot_width, tools=[linked_crosshair]
            )
        ).cols(2)
    else:
        _plot = (_dpolcat_plot + dpolcat.utils.make_legend(size=250)).cols(2)
    _plot
    return


@app.cell
def _(result_preview, ui_viz_enabled):
    mo.stop(not ui_viz_enabled.value)

    _bins = list(range(0, dp.NUM_CATEGORIES + 1))
    result_preview["dpolcat"].hvplot.hist(bins=_bins).opts(
        tools=["wheel_zoom"]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Create .VRTs

    Create .VRT files for the Zarr assets that provide overview (multiscale) support, e.g., for QGIS.

    /// attention

    These point to the absolute path of the .zarr store.
    So, if this is moved, the VRTs will have to be updated/recreated.
    """)
    return


@app.cell
def _():
    ui_vrt = mo.ui.run_button(label="Create .VRTs")
    ui_vrt
    return (ui_vrt,)


@app.cell
def _(OVERVIEW_LEVELS, layers, out_zarr_path, ui_vrt, zarr_group_path):
    mo.stop(not ui_vrt.value, "Click the button above to create .VRT files.")

    def zarr_layer_vrt(zarr_path, layer, levels, vrt_path=None):
        """Creates .VRT containing overviews for each layer."""
        # Mostly Claude (Opus 5.5)
        zarr_path = Path(zarr_path).absolute()
        src = f'ZARR:"{zarr_path}":{zarr_group_path}/{layer}'
        vrt_path = vrt_path or f"data/{zarr_path.stem}_{layer}.vrt"
        gdal.Translate(
            vrt_path,
            f"{src}/{layer}",
            format="VRT",
        )
        tree = ET.parse(vrt_path)
        for band in tree.getroot().iter("VRTRasterBand"):
            for f in levels:
                ovr = ET.SubElement(band, "Overview")
                ET.SubElement(
                    ovr, "SourceFilename"
                ).text = f"{src}/ovr_{f}x/{layer}"
                ET.SubElement(ovr, "SourceBand").text = band.get("band")
        tree.write(vrt_path)
        return vrt_path

    vrts = {
        layer: zarr_layer_vrt(out_zarr_path, layer, OVERVIEW_LEVELS)
        for layer in layers
    }
    return (vrts,)


@app.cell
def _(vrts):
    vrts
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### COG export
    """)
    return


@app.cell
def _():
    ui_export = mo.ui.run_button(label="Export COGs")
    ui_export
    return (ui_export,)


@app.cell
def _(aoi_name, result_ds, ui_export):
    mo.stop(not ui_export.value, "Click the button above to export.")

    exported_cogs = []
    for timeslice in result_ds["dpolcat"]:
        _date = str(timeslice["time"].values)[:10]
        _filename = f"data/dpolcat-{aoi_name}-{_date}.tif"
        _da = timeslice
        # _da = timeslice.astype(np.uint8).rename("dpolcat")
        _da = _da.rio.write_nodata(0)
        _da.rio.to_raster(_filename, compute=True)
        exported_cogs.append(_filename)
    return (exported_cogs,)


@app.cell
def _(exported_cogs):
    exported_cogs
    return


if __name__ == "__main__":
    app.run()
